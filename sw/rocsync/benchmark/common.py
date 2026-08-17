"""Shared utilities for RocSync benchmark tools."""

import sys
from collections import OrderedDict
from dataclasses import dataclass
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
