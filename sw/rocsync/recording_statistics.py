from dataclasses import asdict, dataclass

from rocsync.printer import print, printresult, warnprint


@dataclass
class RecordingStatistics:
    # Frame counts
    n_frames: int
    n_considered_frames: int
    n_rejected_frames: int

    # Scores
    r2_before: float
    rmse_before: float
    r2_after: float
    rmse_after: float

    # Duration and FPS
    source_duration: float  # source-clock span of the analyzed frames
    board_duration: float  # board time between the first and last frame
    nominal_fps: float | None  # rate the source declares, or None if it declares none
    measured_fps: float  # 1000 / median frame period

    # Affine map from source clock ticks to board time, in ms:
    # board_ms = clock_rate * source_ticks + clock_offset_ms
    clock_rate: float
    clock_offset_ms: float
    source_tick_ms: float  # ms per source-clock tick; 1.0 for a container's own pts

    # Start and end
    first_frame: float
    last_frame: float

    # Source timeline
    median_frame_period: float
    n_gaps: int
    n_dropped_frames: int
    largest_gap_ms: float
    timeline_windowed: bool  # True if only part of the recording was analyzed

    # Exposure
    mean_exposure_time: float
    min_exposure_time: float
    max_exposure_time: float
    std_exposure_time: float

    # Timestamps
    considered_timestamps: dict
    rejected_timestamps: dict

    def to_dict(self):
        d = asdict(self)
        del d["considered_timestamps"], d["rejected_timestamps"]
        return d


def warn_about_statistics(statistics: RecordingStatistics):
    """The sanity warnings every clock fit gets, regardless of source type."""
    if statistics.n_considered_frames < 0.8 * (
        statistics.n_considered_frames + statistics.n_rejected_frames
    ):
        fraction = statistics.n_considered_frames / (
            statistics.n_considered_frames + statistics.n_rejected_frames
        )
        warnprint(f"WARNING: Estimated model has fewer than 80% inliers ({fraction:.2%}).")

    drift = statistics.clock_rate / statistics.source_tick_ms
    if abs(drift - 1) > 0.05:
        warnprint(
            f"WARNING: Source clock runs at {drift:.4f}x board time; expected approximately 1x."
        )

    if statistics.n_dropped_frames:
        warnprint(
            f"WARNING: {statistics.n_dropped_frames} frames missing from the source in "
            f"{statistics.n_gaps} gap(s), largest {statistics.largest_gap_ms / 1000:.3f} s."
        )


def print_statistics(statistics: RecordingStatistics):
    format_str = "{:<40} {:>30}"
    print(71 * "-")
    # TODO: find proper thresholds
    printresult(
        "Number of considered frames",
        statistics.n_considered_frames,
        statistics.n_considered_frames > 10,
    )
    printresult(
        "Number of rejected outliers",
        statistics.n_rejected_frames,
        statistics.n_rejected_frames < 0.1 * statistics.n_frames,
    )
    printresult(
        "R2 (before/after outlier rejection)",
        f"{statistics.r2_before:.4f}/{statistics.r2_after:.4f}",
        statistics.r2_after > 0.99,
    )
    printresult(
        "RMSE (before/after outlier rejection)",
        f"{statistics.rmse_before:.2f}/{statistics.rmse_after:.2f} ms",
        statistics.rmse_after < 2,
    )
    printresult(
        "Dropped frames",
        f"{statistics.n_dropped_frames} in {statistics.n_gaps} gap(s), "
        f"max {statistics.largest_gap_ms / 1000:.3f} s",
        statistics.n_dropped_frames == 0,
    )
    print(format_str.format("First frame:", f"{statistics.first_frame / 1000:.3f} s"))
    print(format_str.format("Last frame:", f"{statistics.last_frame / 1000:.3f} s"))
    nominal = f"{statistics.nominal_fps:.3f}" if statistics.nominal_fps is not None else "n/a"
    print(
        format_str.format(
            "Framerate (nominal/measured):",
            f"{nominal}/{statistics.measured_fps:.3f} fps",
        )
    )
    print(
        format_str.format(
            "Clock rate (board/source):",
            f"{statistics.clock_rate / statistics.source_tick_ms:.6f}x",
        )
    )
    scope = "analyzed window" if statistics.timeline_windowed else "source"
    print(
        format_str.format(
            f"Duration ({scope}/board):",
            f"{statistics.source_duration / 1000:.3f}/{statistics.board_duration / 1000:.3f} s "
            f"(Δ={statistics.board_duration - statistics.source_duration:.2f} ms)",
        )
    )
    print(
        format_str.format(
            "Exposure time (mean/min/max/std):",
            f"{statistics.mean_exposure_time:.2f}/{statistics.min_exposure_time:.2f}/"
            f"{statistics.max_exposure_time:.2f}/{statistics.std_exposure_time:.2f} ms",
        )
    )
    print(71 * "-")
