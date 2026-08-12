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
    expected_duration: float  # container span of the analyzed frames
    measured_duration: float  # board time between the first and last frame
    expected_fps: float  # nominal rate reported by the container
    measured_fps: float  # 1000 / median frame period

    # Affine map from container presentation time to board time, in ms:
    # board_ms = speed_factor * pts_ms + intercept
    speed_factor: float
    intercept: float

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
