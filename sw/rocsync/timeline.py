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

from dataclasses import dataclass, field

import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import root_mean_squared_error


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
    source_time_min: float  # smallest source-clock value among all measurements offered to the fit
    source_time_max: float  # largest source-clock value among all measurements offered to the fit

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
    for before, after in zip(values[:-1], values[1:]):
        missing = int(round((after - before) / period)) - 1
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
    two such frames exist, or if no usable threshold can be determined.
    """
    order = sorted(k for k in timestamps if frame_times.get(k) is not None)
    if len(order) < 2:
        raise ValueError(
            f"Need at least 2 timestamped frames with a known source "
            f"timestamp, got {len(order)}."
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

    inlier_mask = model.inlier_mask_
    inlier_x, inlier_y = x[inlier_mask], y[inlier_mask]
    return TimelineFit(
        clock_rate=float(model.estimator_.coef_[0]),
        clock_offset_ms=float(model.estimator_.intercept_),
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


def frame_pts(video_path):
    """Presentation timestamp in ms of every frame, indexed by frame number.

    Uses grab() so frames are demuxed without decoding pixels, which makes this
    cheap enough to run before picking frames out of a video.
    """
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
            "Time-sync data has no clock_rate/clock_offset_ms; it predates the "
            "presentation-timestamp fit. Re-run rocsync to regenerate it."
        )
    return float(clock_rate), float(clock_offset_ms)


def per_frame_times(video_path, statistics):
    """Board time in ms for every frame of a video, indexed by frame number.

    Frame times are read from the video itself, so dropped frames stay where
    they actually are instead of being smeared across a constant-rate timeline.
    """
    clock_rate, clock_offset_ms = affine_from_statistics(statistics)
    return [clock_rate * p + clock_offset_ms for p in frame_pts(video_path)]
