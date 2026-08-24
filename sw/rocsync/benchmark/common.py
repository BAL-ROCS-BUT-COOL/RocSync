"""Shared utilities for RocSync benchmark tools."""

import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from rocsync.board_profiles import PROFILES_BY_ARUCO
from rocsync.dataset import VIDEO_SUFFIXES
from rocsync.timeline import frame_pts, median_frame_period, probe_packet_field, run_ffprobe

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
SEEK_BACKOFF_FRAMES = 32  # retry margin for a seek that landed past the frame wanted

MIN_REFERENCE_FRAMES = 5  # a two-point fit is exact by construction and proves nothing

# A synthesized timeline is built from the annotations themselves, so the only slack it
# needs covers reading the ring arc one LED out at either end. A measured one has to
# absorb the camera: the container stores a nominal frame rate while the sensor exposes
# when it pleases, which on the dataset's 30 fps clips scatters by up to 9 ms.
SYNTHESIZED_RESIDUAL_THRESHOLD_MS = 2.0
MEASURED_RESIDUAL_FRACTION = 1 / 2  # of a source frame
MEASURED_RESIDUAL_MIN_MS = 2.0  # never tighter than the board itself resolves
MEASURED_RESIDUAL_MAX_MS = 50.0  # below the ring period, so a counter step still shows

RETIMED_MARKER = ".retimed"  # `clip.retimed.mp4` is `clip.mp4` on a synthesized timeline


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
    on forever. The stream's packets are therefore counted rather than the count the
    container claims taken at its word.
    """
    counted = _probe_packet_count(path)
    if counted:
        return counted

    # ffprobe could not answer, so demux the file here instead
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return 0
    try:
        # grab() demuxes without decoding pixels, so counting for real stays cheap
        counted = 0
        while cap.grab():
            counted += 1
        return counted
    finally:
        cap.release()


def _probe_packet_count(path):
    """Packets the video stream holds, or None if ffprobe could not count them."""
    output = run_ffprobe(
        path, "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0"
    )
    counted = (output or "").strip().rstrip(",")
    return int(counted) if counted.isdigit() else None


def source_frame_period_ms(video_path):
    """Frame period of the recording this file was cut from, in ms, or None.

    Read from the packet duration, which survives decimation: a clip holding every 16th
    frame of a 30 fps recording still says 33.3 ms per packet while its timestamps sit
    533 ms apart. The timestamp spacing would describe the subsampling instead, and a
    tolerance scaled from that would be far too loose to catch anything.
    """
    durations = [d for d in probe_packet_field(video_path, "duration_time") if d > 0]
    if durations:
        # The mode, so one odd packet at a cut cannot stand for the whole recording
        return float(max(set(durations), key=durations.count)) * 1000

    # Without a packet duration the spacing is all there is, subsampled or not
    try:
        return median_frame_period(frame_pts(video_path))
    except OSError:
        return None  # a file that will not open has no period to report


def measured_residual_threshold_ms(source_period_ms):
    """How far an annotated frame may sit from a clock fitted to a recorded timeline."""
    if not source_period_ms:
        return MEASURED_RESIDUAL_MAX_MS
    scaled = source_period_ms * MEASURED_RESIDUAL_FRACTION
    return min(max(scaled, MEASURED_RESIDUAL_MIN_MS), MEASURED_RESIDUAL_MAX_MS)


def residual_threshold_ms(video_entry):
    """The tolerance that applies to one video, from how its timeline came about.

    An entry records the tolerance its reference was checked against, and that stored
    number wins: a frozen reference stays valid under the rule it was frozen by, not
    whichever rule the code holds today. Deriving is the fallback for an entry written
    before the tolerance was recorded.
    """
    entry = video_entry or {}
    stored = entry.get("residual_threshold_ms")
    if stored is not None:
        return float(stored)
    if entry.get("timeline") == "synthesized":
        return SYNTHESIZED_RESIDUAL_THRESHOLD_MS
    return measured_residual_threshold_ms(entry.get("source_frame_period_ms"))


def is_retimed(path):
    """Whether this file is a retimed clip rather than something recorded."""
    return Path(path).stem.endswith(RETIMED_MARKER)


def retimed_path(path):
    """Where the retimed clip cut from `path` belongs."""
    path = Path(path)
    return path.with_name(f"{path.stem}{RETIMED_MARKER}{path.suffix}")


def has_retimed_sibling(path):
    """Whether a retimed clip cut from `path` sits beside it, in any video container."""
    path = Path(path)
    return any(
        path.with_name(f"{path.stem}{RETIMED_MARKER}{suffix}").is_file()
        for suffix in VIDEO_SUFFIXES
    )


def collect_frames(root_dir, sources_only=False):
    """Every benchmark frame under `root_dir`, sorted, images and videos alike.

    Every frame of a video is benchmark material, so all of them are enumerated. A video
    that will not open contributes no frames at all, which keeps the annotator and the
    validator from disagreeing about what the dataset contains.

    A retimed clip stands in for the video it was cut from, so scoring a dataset holding
    both counts each frame once. `sources_only` walks what was recorded instead, which is
    what the annotator wants: a retimed clip is trimmed to the frames already annotated,
    and annotating the ones outside that window is how the window grows.
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
            if is_retimed(path) if sources_only else has_retimed_sibling(path):
                continue
            n_frames = count_video_frames(path)
            if not n_frames:
                print(f"WARNING: could not read any frame of {path}", file=sys.stderr)
                continue
            frames.extend(FrameRef(path, i, frame_key(rel_path, i)) for i in range(n_frames))
    frames.sort(key=lambda ref: (str(ref.path), -1 if ref.index is None else ref.index))
    return frames


@dataclass(frozen=True)
class Orphans:
    """Ground truth entries no frame on disk stands behind, split by what is wrong.

    The split is what makes pruning safe to automate: a file that is gone is gone, while
    a file that is merely unreadable today -- a truncated download, a codec the machine
    lacks -- must not cost the annotation work behind it.
    """

    missing_images: list  # keys whose file is not there at all
    missing_videos: list  # `videos` paths whose recording is not there at all
    out_of_range_images: list  # keys past the last frame of a file that did decode
    unreadable_videos: list  # present but contributed no frame; their entries are kept

    def prunable(self):
        """Whether anything here can be removed without losing recoverable work."""
        return bool(self.missing_images or self.missing_videos or self.out_of_range_images)

    def is_empty(self):
        return not (self.prunable() or self.unreadable_videos)


def orphaned_entries(ground_truth, data_dir, frames):
    """Ground truth entries with no frame behind them, as an `Orphans`.

    `frames` is the caller's own `collect_frames(data_dir, sources_only=True)` result:
    annotations are keyed to the recording and never to a retimed clip, so the sources
    walk is the one whose keys they are supposed to match.

    Existence is decided by a plain directory listing rather than by that walk, because
    the walk drops an unreadable video too, and a file that cannot be decoded right now
    is a different thing from one that was deleted.
    """
    data_dir = Path(data_dir)
    on_disk = {str(p.relative_to(data_dir)) for p in data_dir.rglob("*") if p.is_file()}
    valid_keys = {ref.key for ref in frames}
    enumerated = {parse_frame_key(ref.key)[0] for ref in frames}

    unreadable = {
        rel
        for rel in on_disk
        if Path(rel).suffix.lower() in VIDEO_SUFFIXES
        and rel not in enumerated
        and not is_retimed(rel)
    }

    missing_images, out_of_range_images = [], []
    for key in ground_truth.get("images") or {}:
        if key in valid_keys:
            continue
        rel_path, index = parse_frame_key(key)
        if rel_path not in on_disk:
            missing_images.append(key)
        elif index is not None and rel_path not in unreadable:
            out_of_range_images.append(key)  # the file decoded and stops before this frame

    missing_videos = []
    for rel_path, entry in (ground_truth.get("videos") or {}).items():
        # A retimed clip is cut again on demand, so its source is what has to exist
        if (entry.get("source") or rel_path) not in on_disk:
            missing_videos.append(rel_path)

    return Orphans(
        missing_images=sorted(missing_images),
        missing_videos=sorted(missing_videos),
        out_of_range_images=sorted(out_of_range_images),
        unreadable_videos=sorted(unreadable),
    )


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
        self._pts: dict[Path, list[float] | None] = {}  # every frame's presentation timestamp

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
        pts = self._frame_pts(ref.path)

        ahead = ref.index - self._next_index
        frame = (
            self._grab_forward(cap, ahead, pts, ref.index)
            if 0 <= ahead <= FORWARD_GRAB_LIMIT
            else None
        )
        if frame is None:
            frame = self._seek_and_read(cap, pts, ref.index)
        self._next_index = ref.index + 1 if frame is not None else -1
        return frame

    def _frame_pts(self, path):
        """Presentation timestamps for `path`, or None when they cannot be read."""
        if path not in self._pts:
            self._pts[path] = frame_pts(path) or None
        return self._pts[path]

    def _window(self, pts, index):
        """The timestamp span that belongs to frame `index` and to no other."""
        low = (pts[index - 1] + pts[index]) / 2 if index > 0 else pts[index] - 1.0
        high = (pts[index] + pts[index + 1]) / 2 if index + 1 < len(pts) else pts[index] + 1.0
        return low, high

    def _is_frame(self, cap, pts, index):
        """Whether the frame just read off `cap` is the one `index` names."""
        if pts is None or index >= len(pts):
            return True
        low, high = self._window(pts, index)
        return low < cap.get(cv2.CAP_PROP_POS_MSEC) < high

    def _grab_forward(self, cap, ahead, pts, index):
        """Frame `index`, reached by reading on from where the capture stands.

        None when it is not what came back, which leaves the caller to seek: an earlier
        read may have been served a neighbouring frame, and counting on from there lands
        every later frame one out.
        """
        for _ in range(ahead):
            if not cap.grab():
                return None
        success, frame = cap.read()
        return frame if success and self._is_frame(cap, pts, index) else None

    def _seek_and_read(self, cap, pts, index):
        """Frame `index`, reached by seeking, or None if it could not be read.

        `CAP_PROP_POS_FRAMES` converts the index into a timestamp through the stream's
        average frame rate, so a recording whose frames are not spaced at that rate --
        anything variable-rate, or cut from a faster source -- lands on a neighbour,
        while the capture still reports the index that was asked for. The frame is
        identified by its own timestamp instead: read on when the seek falls short, and
        seek further back when it overshoots.
        """
        if pts is None or index >= len(pts):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = cap.read()
            return frame if success else None

        low, high = self._window(pts, index)
        for start in (index, max(index - SEEK_BACKOFF_FRAMES, 0), 0):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            while True:
                if not cap.grab():
                    return None
                now = cap.get(cv2.CAP_PROP_POS_MSEC)
                if now > low:
                    break
            if now < high:
                success, frame = cap.retrieve()
                return frame if success else None
        return None

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


def corner_positions_in_image(stats, n_leds=4):
    """Detected corner LED positions in original image coordinates, one slot per LED.

    The pipeline detects corners in the rough-rectified grid, whose scale is a
    property of the branch under test. Un-warping through that grid's own
    homography yields the annotated quantity, comparable across branches.

    The pipeline reports one row per always-on LED, NaN where it found none; this is
    where that becomes the None the annotation format uses, so position i always names
    LED i. `n_leds` only sizes the all-None result for a frame that reported nothing.
    """
    positions = stats.get("corner_positions")
    rough_H = stats.get("rough_homography")
    if positions is None or rough_H is None:
        return [None] * n_leds

    positions = np.array(positions, dtype=np.float64).reshape(-1, 2)
    found = np.isfinite(positions).all(axis=1)
    slots = [None] * len(positions)
    if found.any():
        inv_rough = np.linalg.inv(np.array(rough_H, dtype=np.float64))
        pts = positions[found].reshape(1, -1, 2)
        mapped = cv2.perspectiveTransform(pts, inv_rough).reshape(-1, 2).tolist()
        for i, point in zip(np.flatnonzero(found), mapped, strict=True):
            slots[i] = point
    return slots


def ring_visible(image_data):
    """Ring is visible when start != end (half-open interval has nonzero length)."""
    ring = image_data.get("ring", {})
    return ring.get("start", 0) != ring.get("end", 0)


def reconstruct_timestamp(image_data, board):
    """Reconstruct [start, end] timestamp from counter value and ring position.

    Returns None when the counter or ring is not visible, and equally when the board
    calls the pair undecodable -- an arc wrapping the end of the period, or ending
    within a LED of it, means the counter changed mid-exposure and no longer says
    which period the arc belongs to. The decision is the board's rather than a rule
    of our own: annotating what is on screen is not the same as it being decodable,
    and a benchmark that reconstructed a time here would score the pipeline against
    timestamps it is designed never to produce.
    """
    counter_value = image_data.get("counter", {}).get("value")
    ring = image_data.get("ring", {})
    start, end = ring.get("start", 0), ring.get("end", 0)
    if counter_value is None or start == end:
        return None
    # Annotations hold a half-open interval; the board decodes an inclusive one
    timestamp = board.board_time_from_ring(counter_value, (start, (end - 1) % board.period))
    return list(timestamp) if timestamp is not None else None


def annotated_board_time(entry):
    """Board time in ms an annotation pins down, or None when it pins down none.

    Resolves the board from the annotated marker, so a caller only needs the entry.
    """
    board = PROFILES_BY_ARUCO.get(entry.get("aruco", {}).get("id"))
    if board is None:
        return None
    timestamp = reconstruct_timestamp(entry, board)
    return None if timestamp is None else timestamp[0]


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

    def to_dict(self, threshold_ms):
        """Serialized form, recording the tolerance the reference was checked against."""
        return {**asdict(self), "residual_threshold_ms": threshold_ms}

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


def reference_outliers(clock, starts_by_index, pts_by_index, threshold_ms):
    """[(index, signed residual ms)] for frames the clock disagrees with.

    The tolerance is passed in rather than fixed, because what counts as too far
    depends on where the timeline came from -- see `residual_threshold_ms`.
    """
    if clock is None:
        return []
    outliers = []
    for index in sorted(starts_by_index):
        residual = reference_residual(clock, index, starts_by_index, pts_by_index)
        if residual is not None and abs(residual) > threshold_ms:
            outliers.append((index, residual))
    return outliers


@dataclass(frozen=True)
class RetimedVideo:
    """A retimed clip and the recording it was cut from.

    Annotations stay keyed to the source, so the clip can be regenerated or thrown away
    without touching them; this is what turns a prediction about a retimed frame back
    into the annotation it should be scored against.
    """

    path: str  # relative path of the retimed clip
    source: str  # relative path of the clip it was cut from
    frame_offset: int  # retimed frame j is source frame j + frame_offset


def retimed_videos(ground_truth):
    """{retimed path: RetimedVideo} for every retimed clip the ground truth describes."""
    videos = {}
    for path, entry in (ground_truth.get("videos") or {}).items():
        if entry.get("source") is not None:
            videos[path] = RetimedVideo(
                path=path,
                source=entry["source"],
                frame_offset=int(entry.get("source_frame_offset", 0)),
            )
    return videos


def source_key(key, retimed):
    """The annotation key a frame key refers to, unchanged for anything not retimed."""
    path, index = parse_frame_key(key)
    video = retimed.get(path)
    if video is None or index is None:
        return key
    return frame_key(video.source, index + video.frame_offset)
