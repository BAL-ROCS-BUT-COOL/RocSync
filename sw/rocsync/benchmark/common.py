"""Shared utilities for RocSync benchmark tools."""

import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from rocsync.dataset import VIDEO_SUFFIXES

STEP_ORDER = [
    "aruco_detection",
    "corner_detection",
    "fine_rectification",
    "counter_reading",
    "ring_reading",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

FRAME_KEY_SEPARATOR = "#"
FRAME_INDEX_DIGITS = 6  # zero-padded so keys sort in frame order
FRAME_CACHE_SIZE = 4  # a decoded 4K frame is ~25 MB
FORWARD_GRAB_LIMIT = 12  # a seek re-decodes from the preceding keyframe anyway

REFERENCE_RESIDUAL_THRESHOLD_MS = 2.0  # one ring LED is 1 ms; more is a bad annotation
MIN_REFERENCE_FRAMES = 5  # a two-point fit is exact by construction and proves nothing


@dataclass(frozen=True)
class FrameRef:
    """One benchmark frame: a still image, or one frame of a video.

    `key` is the identity used in both ground_truth.json and the results JSON, so those
    two files associate through it alone and nothing downstream has to know which kind
    of input produced a frame.
    """

    path: Path
    index: int | None  # None for a still image
    key: str


def frame_key(rel_path, index=None):
    """Ground truth key for a frame: the input's relative path, plus an index for video.

    The index is the frame's absolute position in the file, not a presentation timestamp:
    a timestamp shifts when a decoder reports it differently, while the index is a
    property of the file. Zero padding keeps a video's keys sorting in frame order.
    """
    rel_path = str(rel_path)
    if index is None:
        return rel_path
    return f"{rel_path}{FRAME_KEY_SEPARATOR}{index:0{FRAME_INDEX_DIGITS}d}"


def parse_frame_key(key):
    """Split a frame key back into (relative path, index), with None for a still image.

    Only the last separator counts, so a path that contains one itself round-trips.
    """
    head, sep, tail = str(key).rpartition(FRAME_KEY_SEPARATOR)
    if not sep or not tail.isdigit():
        return str(key), None
    return head, int(tail)


def count_video_frames(path):
    """Number of frames a video contributes, or 0 if it cannot be read.

    Enumeration defines the keys, so an over-reported count would invent keys that no
    frame can ever fill -- and that the annotator's jump-to-unannotated would then land
    on forever. The container's own count is checked by grabbing the frame it claims is
    last, and only when that fails is the file demuxed to count for real.
    """
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return 0
    try:
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, reported - 1)
            if cap.grab():
                return reported

        # grab() demuxes without decoding pixels, so counting for real stays cheap
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        counted = 0
        while cap.grab():
            counted += 1
        return counted
    finally:
        cap.release()


def collect_frames(root_dir):
    """Every benchmark frame under `root_dir`, sorted, images and videos alike.

    Every frame of a video is benchmark material, so all of them are enumerated. A video
    that will not open contributes no frames at all, which keeps the annotator and the
    validator from disagreeing about what the dataset contains.
    """
    root_dir = Path(root_dir)
    frames = []
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        rel_path = str(path.relative_to(root_dir))
        if suffix in IMAGE_EXTENSIONS:
            frames.append(FrameRef(path, None, rel_path))
        elif suffix in VIDEO_SUFFIXES:
            n_frames = count_video_frames(path)
            if not n_frames:
                print(f"WARNING: could not read any frame of {path}", file=sys.stderr)
                continue
            frames.extend(FrameRef(path, i, frame_key(rel_path, i)) for i in range(n_frames))
    frames.sort(key=lambda ref: (str(ref.path), -1 if ref.index is None else ref.index))
    return frames


class FrameSource:
    """Reads the image behind a `FrameRef`, for both sequential and random access.

    The two tools walk the same list differently -- the validator front to back, the
    annotator jumping around it -- and in both the frames of a video are interleaved with
    still images, so a per-video iterator like `clips.read_frames_at_indices` does not
    fit. One capture is held open across calls instead.

    Reaching a nearby later frame grabs forward rather than seeking, because a seek makes
    the decoder restart from the preceding keyframe and re-decode the span anyway. So the
    validator, walking in order, never seeks, and the annotator's step forward costs one
    grab while its step back is a cache hit. Frames are not copied: a caller keeping one
    past the next `read` must copy it.
    """

    def __init__(self):
        self._cache = OrderedDict()
        self._cap = None
        self._cap_path = None
        self._next_index = -1  # index the open capture would read next; -1 forces a seek

    def read(self, ref):
        """Decoded BGR image for `ref`, or None if it could not be read."""
        cached = self._cache.pop(ref.key, None)
        if cached is not None:
            self._cache[ref.key] = cached
            return cached

        frame = cv2.imread(str(ref.path)) if ref.index is None else self._read_video(ref)
        if frame is not None:
            self._cache[ref.key] = frame
            while len(self._cache) > FRAME_CACHE_SIZE:
                self._cache.popitem(last=False)
        return frame

    def _read_video(self, ref):
        if self._cap is None or self._cap_path != ref.path:
            self.close()
            cap = cv2.VideoCapture(str(ref.path), cv2.CAP_FFMPEG)
            if not cap.isOpened():
                return None
            self._cap, self._cap_path, self._next_index = cap, ref.path, 0
        cap = self._cap

        ahead = ref.index - self._next_index
        if 0 <= ahead <= FORWARD_GRAB_LIMIT:
            for _ in range(ahead):
                if not cap.grab():
                    self._next_index = -1
                    return None
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, ref.index)

        success, frame = cap.read()
        self._next_index = ref.index + 1 if success else -1
        return frame if success else None

    def close(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._cap_path = None
        self._next_index = -1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def corner_positions_in_image(stats):
    """Detected corner LED positions in original image coordinates.

    The pipeline detects corners in the rough-rectified grid, whose scale is a
    property of the branch under test. Un-warping through that grid's own
    homography yields the annotated quantity, comparable across branches.
    """
    positions = stats.get("corner_positions")
    rough_H = stats.get("rough_homography")
    if positions is None or rough_H is None:
        return [None] * 4

    inv_rough = np.linalg.inv(np.array(rough_H, dtype=np.float64))
    pts = np.array([positions], dtype=np.float64)
    return cv2.perspectiveTransform(pts, inv_rough).reshape(-1, 2).tolist()


def ring_visible(image_data):
    """Ring is visible when start != end (half-open interval has nonzero length)."""
    ring = image_data.get("ring", {})
    return ring.get("start", 0) != ring.get("end", 0)


def reconstruct_timestamp(image_data, board):
    """Reconstruct [start, end] timestamp from counter value and ring position.

    Returns [start, end] list or None if counter or ring is not visible.
    """
    counter_value = image_data.get("counter", {}).get("value")
    ring = image_data.get("ring", {})
    if counter_value is None or ring.get("start", 0) == ring.get("end", 0):
        return None
    start = ring["start"] + counter_value * board.period
    end = ring["end"] + counter_value * board.period
    return [start, end]


def descriptive_stats(values):
    """Compute mean, median, std, min, max, n for a list of numbers."""
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "n": 0}
    a = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "n": len(values),
    }


def confusion_metrics(tp, fp, fn, tn):
    """Derive precision, recall, FPR, and F1 from confusion matrix counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
    }


@dataclass(frozen=True)
class ReferenceClock:
    """Affine map from container presentation time to annotated board time.

    The benchmark's own least-squares fit rather than `rocsync.timeline.fit_timeline`.
    A reference the evaluation scores against has to be a pure function of the
    annotations: fitting it with the production estimator would let a change to that
    estimator move the ground truth, and with it every error measured against it. No
    outlier is tolerated here anyway, which is all RANSAC would have bought.
    """

    clock_rate: float  # board ms per container ms
    clock_offset_ms: float  # board ms at pts == 0
    pts_min_ms: float  # span of the annotated frames, not of the file
    pts_max_ms: float
    n_frames_fitted: int
    rmse_ms: float
    max_residual_ms: float

    def predict(self, pts_ms):
        """Board time in ms for one or many container timestamps in ms."""
        return self.clock_rate * np.asarray(pts_ms, dtype=float) + self.clock_offset_ms

    def to_dict(self):
        return {
            **asdict(self),
            "residual_threshold_ms": REFERENCE_RESIDUAL_THRESHOLD_MS,
        }

    @classmethod
    def from_dict(cls, data):
        fields = (
            "clock_rate",
            "clock_offset_ms",
            "pts_min_ms",
            "pts_max_ms",
            "n_frames_fitted",
            "rmse_ms",
            "max_residual_ms",
        )
        return cls(**{k: data[k] for k in fields})


def fit_reference_clock(starts_by_index, pts_by_index, exclude=None):
    """Least-squares clock over annotated frames, or None with too few of them.

    `starts_by_index` maps a video's frame index to its annotated board time in ms,
    `pts_by_index` to the frame's presentation timestamp. Passing `exclude` drops one
    index, which measures a frame against a fit that does not contain it.
    """
    order = sorted(k for k in starts_by_index if k != exclude and pts_by_index.get(k) is not None)
    if len(order) < MIN_REFERENCE_FRAMES:
        return None

    x = np.array([pts_by_index[k] for k in order], dtype=float)
    y = np.array([starts_by_index[k] for k in order], dtype=float)
    if x.min() == x.max():  # a vertical line has no slope to fit
        return None

    clock_rate, clock_offset_ms = (float(v) for v in np.polyfit(x, y, 1))
    residuals = clock_rate * x + clock_offset_ms - y
    return ReferenceClock(
        clock_rate=clock_rate,
        clock_offset_ms=clock_offset_ms,
        pts_min_ms=float(x.min()),
        pts_max_ms=float(x.max()),
        n_frames_fitted=len(order),
        rmse_ms=float(np.sqrt(np.mean(residuals**2))),
        max_residual_ms=float(np.max(np.abs(residuals))),
    )


def reference_residual(clock, index, starts_by_index, pts_by_index):
    """Signed residual of one frame in ms, predicted minus annotated, or None."""
    pts = pts_by_index.get(index)
    if clock is None or pts is None or index not in starts_by_index:
        return None
    return float(clock.predict(pts)) - float(starts_by_index[index])


def reference_outliers(clock, starts_by_index, pts_by_index):
    """[(index, signed residual ms)] for frames beyond the reference threshold."""
    if clock is None:
        return []
    outliers = []
    for index in sorted(starts_by_index):
        residual = reference_residual(clock, index, starts_by_index, pts_by_index)
        if residual is not None and abs(residual) > REFERENCE_RESIDUAL_THRESHOLD_MS:
            outliers.append((index, residual))
    return outliers
