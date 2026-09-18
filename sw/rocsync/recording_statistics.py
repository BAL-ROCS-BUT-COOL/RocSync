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
    clock_rate_stderr: float  # 1 sigma on clock_rate, in board ms per source tick
    extrapolation_stderr_ms: float  # 3 sigma on board time at the ends of the fitted span
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
    inlier_span_ms: float  # board-clock lever arm the rate was measured over
    inlier_mid_ms: float  # board time the rate pivots about

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


CLOCK_DRIFT_WARN_PPM = 1000  # 0.1%: a crystal is good to tens of ppm, a misdeclared fps is worse
CLOCK_DRIFT_BAD_PPM = 50000  # 5%: not a clock at all any more
RATE_STDERR_WARN_MS = 5.0  # extrapolated ends this uncertain are not worth reporting a rate for


def warn_about_statistics(statistics: RecordingStatistics):
    """The sanity warnings every clock fit gets, regardless of source type."""
    if statistics.n_considered_frames < 0.8 * (
        statistics.n_considered_frames + statistics.n_rejected_frames
    ):
        fraction = statistics.n_considered_frames / (
            statistics.n_considered_frames + statistics.n_rejected_frames
        )
        warnprint(f"WARNING: Estimated model has fewer than 80% inliers ({fraction:.2%}).")

    if statistics.extrapolation_stderr_ms > RATE_STDERR_WARN_MS:
        warnprint(
            f"WARNING: Clock rate is not reliably measurable: {statistics.n_considered_frames} "
            f"inliers span only {statistics.inlier_span_ms / 1000:.3f} s of a "
            f"{statistics.source_duration / 1000:.1f} s recording, leaving the fit uncertain by "
            f"±{statistics.extrapolation_stderr_ms:.0f} ms at the ends of the analyzed span. "
            "Widen --window or improve board visibility."
        )

    drift_ppm = (statistics.clock_rate / statistics.source_tick_ms - 1) * 1e6
    if abs(drift_ppm) > CLOCK_DRIFT_BAD_PPM:
        warnprint(
            f"WARNING: Source clock runs {drift_ppm:+.0f} ppm off board time; "
            "that is not drift, something is wrong with the fit."
        )
    elif abs(drift_ppm) > CLOCK_DRIFT_WARN_PPM:
        warnprint(
            f"WARNING: Source clock runs {drift_ppm:+.0f} ppm off board time; "
            f"expected within {CLOCK_DRIFT_WARN_PPM} ppm."
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
    drift = statistics.clock_rate / statistics.source_tick_ms
    drift_ppm = (drift - 1) * 1e6
    printresult(
        "Clock rate (board/source)",
        f"{drift:.6f}x ({drift_ppm:+.0f} ppm)",
        abs(drift_ppm) <= CLOCK_DRIFT_WARN_PPM,
    )
    stderr_ppm = statistics.clock_rate_stderr / statistics.source_tick_ms * 1e6
    printresult(
        "Clock rate uncertainty (3σ at span ends)",
        f"±{statistics.extrapolation_stderr_ms:.1f} ms (±{stderr_ppm:.0f} ppm)",
        statistics.extrapolation_stderr_ms <= RATE_STDERR_WARN_MS,
    )
    if statistics.source_duration > 0:
        printresult(
            "Inlier span / analyzed span",
            f"{statistics.inlier_span_ms / 1000:.1f}/{statistics.source_duration / 1000:.1f} s "
            f"({statistics.inlier_span_ms / statistics.source_duration:.0%})",
            statistics.inlier_span_ms / statistics.source_duration >= 0.5,
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
