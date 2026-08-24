"""Mapping between a video's container clock and the RocSync board clock.

Every frame in an MP4 carries its own presentation timestamp, so the container
clock is a direct measurement of when each frame was shown. Fitting board time
against that timestamp -- rather than against the frame index -- keeps the fit
correct when frames are missing: a dropped span is a gap in the timestamps, not
a constant shift of every later index that the fit would have to absorb into
its clock_rate and offset.

The result is a plain affine map, board_ms = clock_rate * pts_ms + clock_offset_ms, which
is all any consumer needs in order to time a frame it has just decoded.
"""

import subprocess
from dataclasses import dataclass, field
from itertools import pairwise
from typing import cast

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import root_mean_squared_error

from rocsync.video_statistics import VideoStatistics


@dataclass
class TimelineFit:
    """Affine map from container presentation time to RocSync board time."""

    clock_rate: float  # board ms per container ms
    clock_offset_ms: float  # board ms at pts == 0
    order: list  # frame indices, in the order they entered the fit
    inlier_mask: np.ndarray = field(repr=False)
    r2_before: float
    rmse_before: float
    r2_after: float
    rmse_after: float
    residual_threshold: float
    source_time_min: float  # smallest source-clock value offered to the fit
    source_time_max: float  # largest source-clock value offered to the fit

    def predict(self, pts_ms):
        """Board time in ms for one or many container timestamps in ms."""
        return self.clock_rate * np.asarray(pts_ms, dtype=float) + self.clock_offset_ms

    def to_dict(self) -> dict:
        """Fit-derived fields shared by every clock-fit producer (video, FTK)."""
        return {
            "clock_rate": self.clock_rate,
            "clock_offset_ms": self.clock_offset_ms,
            "r2_before": self.r2_before,
            "rmse_before": self.rmse_before,
            "r2_after": self.r2_after,
            "rmse_after": self.rmse_after,
            "n_considered_frames": int(np.sum(self.inlier_mask)),
            "n_rejected_frames": int(np.sum(~self.inlier_mask)),
            "first_frame": float(self.predict(self.source_time_min)),
            "last_frame": float(self.predict(self.source_time_max)),
        }


def median_frame_period(pts, fallback=None):
    """Median interval between consecutive presentation timestamps, in ms.

    Robust against dropped frames, since a gap only ever adds a few large
    intervals. Returns `fallback` when there is nothing to measure -- a period
    of zero would make RANSAC's residual threshold zero and reject everything.
    """
    values = np.sort(np.asarray([p for p in pts if p is not None], dtype=float))
    if values.size < 2:
        return fallback

    deltas = np.diff(values)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return fallback

    period = float(np.median(deltas))
    return period if period > 0 else fallback


def detect_dropouts(pts, period):
    """Locate gaps in the presentation timeline.

    Returns (n_gaps, n_dropped, largest_gap_ms, gaps), where `gaps` holds one
    (pts_before, pts_after, n_missing) triple per gap.
    """
    values = np.sort(np.asarray([p for p in pts if p is not None], dtype=float))
    if values.size < 2 or not period or period <= 0:
        return 0, 0, 0.0, []

    gaps = []
    for before, after in pairwise(values):
        missing = round((after - before) / period) - 1
        if missing >= 1:
            gaps.append((float(before), float(after), missing))

    n_dropped = sum(missing for _, _, missing in gaps)
    largest_gap = max((after - before for before, after, _ in gaps), default=0.0)
    return len(gaps), n_dropped, float(largest_gap), gaps


def fit_timeline(
    frame_times,
    timestamps,
    residual_threshold=None,
    fallback_period=None,
    max_trials=1000,
):
    """Robustly fit board time against a source clock.

    frame_times: {frame_index: source_ms} for every frame that was read.
    timestamps:  {frame_index: (start_ms, end_ms)} as measured off the board.

    For video, the source clock is the container's presentation timestamp and
    the keys are frame indices. For a device that reports its own clock, pass
    an identity map and an explicit `residual_threshold`; without one the
    threshold is the median frame period, i.e. at most one frame of deviation.

    Only frames present in both dicts are used. Raises ValueError if fewer than
    two such frames exist, if no usable threshold can be determined, or if the
    fit is degenerate (a non-positive or non-finite clock rate).
    """
    order = sorted(k for k in timestamps if frame_times.get(k) is not None)
    if len(order) < 2:
        raise ValueError(
            f"Need at least 2 timestamped frames with a known source timestamp, got {len(order)}."
        )

    valid_source_times = [v for v in frame_times.values() if v is not None]
    source_time_min, source_time_max = min(valid_source_times), max(valid_source_times)

    x = np.array([frame_times[k] for k in order], dtype=float).reshape(-1, 1)
    y = np.array([timestamps[k][0] for k in order], dtype=float)

    threshold = residual_threshold
    if threshold is None:
        threshold = median_frame_period(frame_times.values(), fallback=fallback_period)
    if not threshold or threshold <= 0:
        raise ValueError("Unable to determine a usable residual threshold.")

    model = RANSACRegressor(
        residual_threshold=threshold,  # max one frame deviation
        max_trials=max_trials,  # more trials for more consistent results
        random_state=0,  # deterministic results
    )
    model.fit(x, y)

    # RANSACRegressor defaults to a LinearRegression base estimator
    estimator = cast(LinearRegression, model.estimator_)
    clock_rate = float(estimator.coef_[0])
    clock_offset_ms = float(estimator.intercept_)
    if not np.isfinite(clock_rate) or clock_rate <= 0:
        raise ValueError(
            f"Fitted clock_rate is {clock_rate}, but board time must advance with the source clock."
        )
    if not np.isfinite(clock_offset_ms):
        raise ValueError(f"Fitted clock_offset_ms is {clock_offset_ms}.")

    inlier_mask = model.inlier_mask_
    inlier_x, inlier_y = x[inlier_mask], y[inlier_mask]
    return TimelineFit(
        clock_rate=clock_rate,
        clock_offset_ms=clock_offset_ms,
        order=order,
        inlier_mask=inlier_mask,
        r2_before=model.score(x, y),
        rmse_before=root_mean_squared_error(y, model.predict(x)),
        r2_after=model.score(inlier_x, inlier_y),
        rmse_after=root_mean_squared_error(inlier_y, model.predict(inlier_x)),
        residual_threshold=threshold,
        source_time_min=source_time_min,
        source_time_max=source_time_max,
    )


def summarize_timeline(
    timestamps,
    frame_times,
    n_frames,
    fps,
    window_frame_times=None,
    timeline_windowed=False,
):
    """Fit board time against the container clock and describe the result.

    Everything between a set of decoded timestamps and a finished `VideoStatistics`:
    the frame period, the fit, the dropouts, each frame's residual and the split into
    inliers and outliers. Shared with the benchmark, so what it measures is the
    summary a real run produces rather than a copy of it.

    `window_frame_times` lists the frames of each search window separately, because a
    gap between two disjoint windows is not a dropout. Callers that scanned the whole
    file can leave it out.

    Returns (statistics, fit, considered, rejected, gaps). Raises ValueError when the
    timeline cannot be fitted.
    """
    if len(timestamps) < 2:
        raise ValueError("Insufficient number of timestamped frames.")
    if not frame_times:
        raise ValueError("No frames could be read.")

    # Last-resort frame period only; the span is measured off the frames themselves
    nominal_period = 1000 / fps if fps > 0 else None
    period = median_frame_period(frame_times.values(), fallback=nominal_period)
    if not period:
        raise ValueError("Unable to determine the frame period.")

    try:
        fit = fit_timeline(frame_times, timestamps, fallback_period=nominal_period)
    except ValueError as e:
        raise ValueError(f"Unable to fit the frame timeline: {e}") from e

    # Counted per window, so the span between disjoint windows is not a dropout
    n_gaps = n_dropped_frames = 0
    largest_gap_ms = 0.0
    gaps = []
    for window_times in window_frame_times or [frame_times]:
        window_gaps, window_dropped, window_largest, found = detect_dropouts(
            window_times.values(), period
        )
        n_gaps += window_gaps
        n_dropped_frames += window_dropped
        largest_gap_ms = max(largest_gap_ms, window_largest)
        gaps.extend(found)

    # Add error to timestamps, following the order the fit used
    x = np.array([frame_times[k] for k in fit.order])
    y = np.array([timestamps[k][0] for k in fit.order])
    errors = fit.predict(x) - y
    annotated_timestamps = {
        frame_number: (*timestamps[frame_number], error)
        for frame_number, error in zip(fit.order, errors, strict=True)
    }

    # Remove outliers
    considered = {
        k: annotated_timestamps[k]
        for k, is_inlier in zip(fit.order, fit.inlier_mask, strict=True)
        if is_inlier
    }
    rejected = {
        k: annotated_timestamps[k]
        for k, is_inlier in zip(fit.order, fit.inlier_mask, strict=True)
        if not is_inlier
    }

    # Span anchored on frames actually present, so nothing is extrapolated
    fit_stats = fit.to_dict()
    pts_min, pts_max = min(frame_times.values()), max(frame_times.values())
    exposure_times = [end - start for start, end, _ in considered.values()]
    statistics = VideoStatistics(
        n_frames=n_frames,
        container_duration=pts_max - pts_min,
        board_duration=fit_stats["last_frame"] - fit_stats["first_frame"],
        nominal_fps=fps,
        measured_fps=1000 / period,
        median_frame_period=period,
        n_gaps=n_gaps,
        n_dropped_frames=n_dropped_frames,
        largest_gap_ms=largest_gap_ms,
        timeline_windowed=timeline_windowed,
        mean_exposure_time=float(np.mean(exposure_times)),
        min_exposure_time=float(np.min(exposure_times)),
        max_exposure_time=float(np.max(exposure_times)),
        std_exposure_time=float(np.std(exposure_times)),
        considered_timestamps=considered,
        rejected_timestamps=rejected,
        **fit_stats,
    )
    return statistics, fit, considered, rejected, gaps


def run_ffprobe(video_path, *args):
    """stdout of an ffprobe query about a video's first video stream, or None.

    None means ffprobe is missing or refused the file, which is what makes every caller
    keep a way of its own to answer the question it asked.
    """
    try:
        return subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", *args, str(video_path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def probe_packet_field(video_path, field):
    """Values of one ffprobe packet field, in packet order, as floats.

    A packet carries the numbers the container already stores, so reading it needs no
    decoder: ffprobe's `frame=` entries decode the file and cost seconds per video,
    `packet=` does not.
    """
    output = run_ffprobe(video_path, "-show_entries", f"packet={field}", "-of", "csv=p=0")
    # A packet the container left blank prints as 'N/A', which is not a number
    fields = (line.strip().rstrip(",") for line in (output or "").splitlines())
    return [float(f) for f in fields if _is_float(f)]


def _is_float(text):
    try:
        float(text)
    except ValueError:
        return False
    return True


def parse_ratio(text):
    """An ffprobe 'num/den' ratio as a float, or None if it is not a number at all.

    A plain number is accepted too, so a field ffprobe prints as a ratio and a human
    writes as a decimal both read the same.
    """
    num, sep, den = (text or "").partition("/")
    try:
        if not sep:
            return float(num)
        num, den = float(num), float(den)
    except ValueError:
        return None
    return num / den if den else None


def frame_pts(video_path):
    """Presentation timestamp in ms of every frame, indexed by frame number.

    Read off the container's packets, so no frame is decoded -- which is what makes this
    cheap enough to run before picking frames out of a video. Decoding the whole file,
    the fallback for a container ffprobe cannot read the packet timestamps of, costs
    seconds per video.

    The timestamps are stored as integer ticks of the stream's time base, and are turned
    into ms here exactly the way the decoder does it, so the two agree bit for bit rather
    than to within the microsecond that ffprobe prints seconds to.

    Timestamps are relative to the stream's start time, so a clip whose container starts
    at 47 s still begins at 0.0 here. `clock_offset_ms` is defined against this view,
    which is also the one `read_frames_async` feeds the pipeline; an absolute container
    timestamp has to have the start time subtracted first.
    """
    output = run_ffprobe(
        video_path,
        "-show_entries",
        "stream=time_base,start_pts:packet=pts",
        "-of",
        "default=noprint_wrappers=1",
    )
    ticks, stream = [], {}
    for line in (output or "").splitlines():
        name, _, value = line.partition("=")
        if name == "pts" and _is_float(value):
            ticks.append(float(value))
        elif value and value != "N/A":
            stream[name] = value

    time_base = parse_ratio(stream.get("time_base"))
    if ticks and time_base is not None:
        # Packets arrive in decode order, which B-frames make differ from display order
        ticks.sort()
        start = float(stream.get("start_pts", ticks[0]))
        return [(t - start) * time_base * 1000 for t in ticks]
    return _decoded_frame_pts(video_path)


def _decoded_frame_pts(video_path):
    """`frame_pts` the slow way, for a file ffprobe could not read the packets of."""
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    pts = []
    try:
        while cap.grab():
            pts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    finally:
        cap.release()
    return pts


def affine_from_statistics(statistics):
    """(clock_rate, clock_offset_ms) out of one entry of the JSON written by `rocsync`."""
    clock_rate = statistics.get("clock_rate")
    clock_offset_ms = statistics.get("clock_offset_ms")
    if clock_rate is None or clock_offset_ms is None:
        raise KeyError(
            "Time-sync entry has no clock_rate/clock_offset_ms; re-run rocsync to regenerate it."
        )
    return float(clock_rate), float(clock_offset_ms)


def per_frame_times(video_path, statistics):
    """Board time in ms for every frame of a video, indexed by frame number.

    Frame times are read from the video itself, so dropped frames stay where
    they actually are instead of being smeared across a constant-rate timeline.
    """
    clock_rate, clock_offset_ms = affine_from_statistics(statistics)
    return [clock_rate * p + clock_offset_ms for p in frame_pts(video_path)]
