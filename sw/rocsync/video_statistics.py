from dataclasses import asdict, dataclass


@dataclass
class VideoStatistics:
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
    container_duration: float  # container span of the analyzed frames
    board_duration: float  # board time between the first and last frame
    nominal_fps: float  # rate reported by the container
    measured_fps: float  # 1000 / median frame period

    # Affine map from source clock time to board time, in ms:
    # board_ms = clock_rate * source_ms + clock_offset_ms
    clock_rate: float
    clock_offset_ms: float

    # Start and end
    first_frame: float
    last_frame: float

    # Container timeline
    median_frame_period: float
    n_gaps: int
    n_dropped_frames: int
    largest_gap_ms: float
    timeline_windowed: bool  # True if only part of the file was analyzed

    # Exposure
    mean_exposure_time: float
    min_exposure_time: float
    max_exposure_time: float
    std_exposure_time: float

    # Timestamps
    considered_timestamps: dict
    rejected_timestamps: dict

    def to_dict(self):
        return asdict(self)
