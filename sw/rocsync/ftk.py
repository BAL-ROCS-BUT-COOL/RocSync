import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from tqdm import tqdm

from rocsync.board_profiles import PROFILES_BY_FTK, BoardProfile
from rocsync.camera import CameraType
from rocsync.printer import errprint
from rocsync.timeline import detect_dropouts, fit_timeline, median_frame_period

DISTANCE_THRESHOLD = 3


maker_format = [
    "host_timestamp",
    "ftk_timestamp",
    "type",
    "marker_id",
    "x_position",
    "y_position",
    "z_position",
    "r00",
    "r01",
    "r02",
    "r10",
    "r11",
    "r12",
    "r20",
    "r21",
    "r22",
    "registration_error",
]

fiducial_format = [
    "host_timestamp",
    "ftk_timestamp",
    "type",
    "x_position",
    "y_position",
    "z_position",
    "triangulation_error",
]


class FTKPipeline:
    def __init__(self) -> None:
        self.debug_directory = None


def read_ring(
    fiducials: list[tuple[float, float]],
    board: BoardProfile,
    ax: plt.Axes | None = None,
) -> tuple[int, int] | None:
    leds = np.zeros(board.period, dtype=bool)
    for i, led in enumerate(board.ring_led_coords(CameraType.INFRARED)):
        # Check if any fiducial matches this LED
        for fiducial in fiducials:
            distance = np.linalg.norm(fiducial - led)
            if distance < DISTANCE_THRESHOLD:
                leds[i] = True
                break

        if ax is not None:
            color = "red" if leds[i] else "blue"
            ax.add_patch(Circle(led, DISTANCE_THRESHOLD, color=color, fill=False))

    potential_starts = []
    potential_ends = []
    for i in range(board.period):
        if not leds[(i - 1) % board.period] and leds[i]:
            potential_starts.append(i)
        if leds[i] and not leds[(i + 1) % board.period]:
            potential_ends.append(i)

    # Make sure that there is exactly ONE of enabled
    if len(potential_starts) == 1 and len(potential_ends) == 1:
        return (potential_starts[0], potential_ends[0])


def read_counter(
    fiducials: list[tuple[float, float]],
    board: BoardProfile,
    ax: plt.Axes | None = None,
):
    counter = 0
    # Counter LEDs are listed most significant bit first, matching vision.read_counter.
    for i, led in enumerate(board.counter_led_coords[CameraType.INFRARED]):
        enabled = False
        for fiducial in fiducials:
            distance = np.linalg.norm(fiducial - led)
            if distance < DISTANCE_THRESHOLD:
                enabled = True
                break
        if enabled:
            counter += 2 ** (board.counter_bits - 1 - i)

        if ax is not None:
            color = "red" if enabled else "blue"
            ax.add_patch(Circle(led, DISTANCE_THRESHOLD, color=color, fill=False))

    return counter


def process_frame(
    position: np.ndarray,
    rotation_matrix: np.ndarray,
    fiducials: list[dict],
    board: BoardProfile,
    ax: plt.Axes | None = None,
) -> tuple[int, int] | None:
    # Transform fiducials into local coordinate system
    transformed_fiducials = []
    inv_rotation = np.linalg.inv(rotation_matrix)
    for fiducial in fiducials:
        fid_pos_world = np.array(
            [
                float(fiducial["x_position"]),
                float(fiducial["y_position"]),
                float(fiducial["z_position"]),
                1.0,
            ]
        )
        fid_pos_marker = inv_rotation @ (fid_pos_world - position)
        fid_pos_marker[:2] += np.array([5, 5])  # Adjust for PCB origin

        # Filter fiducials within the PCB area
        if (
            abs(fid_pos_marker[2]) < 5
            and 0 < fid_pos_marker[0] < 250
            and 0 < fid_pos_marker[1] < 250
        ):
            transformed_fiducials.append(fid_pos_marker[:2])

    # Rotate until counter is readable
    # TODO: not required for Rev2
    for i in range(4):
        counter = read_counter(transformed_fiducials, board, ax)
        if counter > 0:
            break

        # Rotate 90 degrees arround center
        rot90 = np.array([[0, -1], [1, 0]])
        rotated_fiducials = []
        center = np.array([board.centre_mm, board.centre_mm])
        for f in transformed_fiducials:
            v = f - center
            rotated = rot90 @ v + center
            rotated_fiducials.append(rotated)
        transformed_fiducials = rotated_fiducials

    if ax is not None:
        for fid in transformed_fiducials:
            ax.scatter(fid[0], fid[1], color="green")

    if counter == 0:
        return None

    ring = read_ring(transformed_fiducials, board, ax)
    if ring is not None:
        start, end = ring
        # Check if the values are valid
        if (
            start > end
            or min(start, board.period - start) <= 2
            or min(board.period, 100 - end) <= 2
        ):
            return None

        start += counter * board.period
        end += counter * board.period
        return start, end
    return None


def plot_timechart(x, y, x_range, y_pred, debug_dir):
    plt.figure()
    plt.scatter(x, y, color="blue", label="Measurements", marker=".")
    plt.plot(x_range, y_pred, color="r", label="Fitted Model")
    plt.xlabel("Device timestamp")
    plt.ylabel("RocSync timestamp [ms]")
    plt.title("Frame timing")
    plt.gca().ticklabel_format(style="plain", useOffset=False)
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.savefig(f"{debug_dir}/timestamps.png")


def fit_ftk_timestamps(
    timestamps: dict[int, tuple[int, int]],
    frame_times: dict[int, int],
    debug_dir=None,
) -> dict:
    # The device reports its own clock, so regress board time directly on it.
    # Not frame-periodic, hence an explicit residual threshold.
    fit = fit_timeline(
        frame_times,
        timestamps,
        residual_threshold=10,
        max_trials=10000,  # more trials for more consistent results
    )

    period = median_frame_period(frame_times.values())
    n_gaps, n_dropped_frames, largest_gap_ms, _ = detect_dropouts(
        frame_times.values(), period
    )

    considered = {
        k: timestamps[k]
        for k, is_inlier in zip(fit.order, fit.inlier_mask)
        if is_inlier
    }
    exposure_times = [end - start for start, end in considered.values()]

    results = fit.to_dict()
    results.update(
        {
            "n_frames": len(frame_times),
            "median_frame_period": period,
            "n_gaps": n_gaps,
            "n_dropped_frames": n_dropped_frames,
            "largest_gap_ms": largest_gap_ms,
            "mean_exposure_time": float(np.mean(exposure_times)),
            "min_exposure_time": float(np.min(exposure_times)),
            "max_exposure_time": float(np.max(exposure_times)),
            "std_exposure_time": float(np.std(exposure_times)),
        }
    )

    if debug_dir is not None:
        x = np.array(fit.order).reshape(-1, 1)
        y = np.array([timestamps[k][0] for k in fit.order])
        x_range = np.array([np.min(x), np.max(x)]).reshape(-1, 1)
        plot_timechart(x, y, x_range, fit.predict(x_range), debug_dir)
    return results


def process_ftk_recording(filename: str, debug_dir=None) -> dict:
    with open(filename, "r") as file:
        total_lines = sum(1 for _ in file)

    timestamps = {}
    frame_times = {}
    with open(filename, "r") as file:
        with tqdm(total=total_lines, desc="Processing lines") as pbar:
            while True:
                line = file.readline()
                if not line:
                    break
                pbar.update(1)
                fields = line.strip().split(",")

                # Find frame with detected marker
                if len(fields) >= len(maker_format) and fields[2] == "m":
                    marker = dict(zip(maker_format, fields))

                    # Get board profile for the PCB associated with this marker
                    board = PROFILES_BY_FTK.get(int(marker["marker_id"]))
                    if board is None:
                        continue

                    # Record every frame the tracker reported for this board,
                    # whether or not the counter/ring below decodes: the frame
                    # count and any dropouts are measured off this map.
                    ftk_timestamp = int(marker["ftk_timestamp"])
                    frame_times[ftk_timestamp] = ftk_timestamp

                    # Read and collect all related fiducials (type "f") immediately after this marker
                    fiducials = []
                    current_pos = file.tell()
                    while True:
                        fid_line = file.readline()
                        if not fid_line:
                            break
                        pbar.update(1)
                        fid_fields = fid_line.strip().split(",")
                        if (
                            fid_fields[2] != "f"
                            or fid_fields[1] != marker["ftk_timestamp"]
                        ):
                            # Not a fiducial or not part of the marker; stop collecting and restore cursor
                            file.seek(current_pos)
                            break

                        fiducial = dict(
                            zip(fiducial_format, fid_fields[: len(fiducial_format)])
                        )
                        fiducials.append(fiducial)
                        current_pos = file.tell()

                    position = np.array(
                        [
                            float(marker["x_position"]),
                            float(marker["y_position"]),
                            float(marker["z_position"]),
                            1.0,
                        ]
                    )

                    # Make sure z-axis is pointing towards the camera
                    if float(marker["r22"]) < 0:
                        rotation_matrix = np.array(
                            [
                                [
                                    float(marker["r01"]),
                                    float(marker["r00"]),
                                    float(marker["r02"]),
                                    0.0,
                                ],
                                [
                                    float(marker["r11"]),
                                    float(marker["r10"]),
                                    float(marker["r12"]),
                                    0.0,
                                ],
                                [
                                    float(marker["r21"]),
                                    float(marker["r20"]),
                                    float(marker["r22"]),
                                    0.0,
                                ],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    else:
                        rotation_matrix = np.array(
                            [
                                [
                                    float(marker["r00"]),
                                    float(marker["r01"]),
                                    float(marker["r02"]),
                                    0.0,
                                ],
                                [
                                    float(marker["r10"]),
                                    float(marker["r11"]),
                                    float(marker["r12"]),
                                    0.0,
                                ],
                                [
                                    float(marker["r20"]),
                                    float(marker["r21"]),
                                    float(marker["r22"]),
                                    0.0,
                                ],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )

                    # Plot debug info if enabled
                    if debug_dir is not None:
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.set_title("Detected Fiducials")
                        ax.invert_yaxis()
                        ax.grid(True)
                        ax.set_aspect("equal")

                        result = process_frame(
                            position, rotation_matrix, fiducials, board, ax
                        )
                        # if result is not None and result[0] > 100:
                        fig.savefig(
                            f"{debug_dir}/{marker['ftk_timestamp']}.png",
                            bbox_inches="tight",
                        )
                        plt.close(fig)
                    else:
                        result = process_frame(
                            position, rotation_matrix, fiducials, board
                        )

                    if result is not None:
                        timestamps[int(marker["ftk_timestamp"])] = result
    if len(timestamps) > 0:
        try:
            return fit_ftk_timestamps(timestamps, frame_times, debug_dir)
        except ValueError as e:
            errprint(f"Error: Unable to fit the FTK timeline: {e}")
    return None
