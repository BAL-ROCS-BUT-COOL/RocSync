"""Read an Atracsys FusionTrack CSV recording and fit its board timeline.

The counter/ring readers and the 3D-fiducial decode entry points used here now live in
``rocsync.fiducial_decode`` and ``rocsync.fiducials`` -- they have nothing
FusionTrack-specific about them, only this file's CSV parsing and plotting do. Keeping
this module import-light was not a goal when it held both; it is one now, so matplotlib
and tqdm are imported only where they are actually used, and ``rocsync.timeline``
(which pulls in scikit-learn) only by ``fit_ftk_timestamps``.

A frame decodes one of two ways, tried in that order:

1. **A registered rigid geometry.** If the frame carries a marker whose id is in
   ``PROFILES_BY_FTK``, its reported position and rotation fix the board plane
   directly.
2. **The corner constellation search**, on that frame's raw fiducials, when (1) is
   absent or fails to decode. Measured live this carries the large majority of
   decodes -- see ``rocsync.fiducials``'s module docstring -- so it is not a rare
   fallback, and it always gets a chance even when a marker was matched.

Either way the result is a plane, projected and handed to
``board_detection.find_board`` by ``rocsync.fiducials.decode_fiducials``, which is why
a frame with no matched marker at all is no longer invisible to this reader: previously
only marker rows were even parsed for their trailing fiducials.
"""

import numpy as np

from rocsync.board_profiles import PROFILES_BY_FTK, BoardProfile
from rocsync.fiducials import MAX_FIDUCIALS, decode_fiducials, plane_from_rotation
from rocsync.printer import errprint, print, warnprint
from rocsync.recording_statistics import print_statistics, warn_about_statistics

FIT_RESIDUAL_THRESHOLD_MS = 10  # RANSAC inlier band; the tracker is not frame-periodic
FTK_TICK_MS = 0.001  # the FusionTrack reports its frame clock in microseconds


marker_format = [
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


def _iter_frames(file, pbar):
    """Group a FusionTrack CSV into frames: ``(ftk_timestamp, markers, fiducials)``.

    The tracker writes one record per detected object per acquisition, in timestamp
    order, so a group is flushed as soon as the timestamp changes. Unlike the marker
    lookahead this replaces, a frame is recognised whether or not any marker in it
    matches a registered geometry -- fiducials no longer live only in a matched
    marker's shadow.
    """
    seen = set()
    ts, markers, fiducials = None, [], []
    while True:
        line = file.readline()
        if not line:
            break
        pbar.update(1)
        fields = [f.strip() for f in line.strip().split(",")]
        if len(fields) < 3:
            continue

        if fields[2] == "m" and len(fields) >= len(marker_format):
            record = dict(zip(marker_format, fields[: len(marker_format)], strict=True))
        elif fields[2] == "f" and len(fields) >= len(fiducial_format):
            record = dict(zip(fiducial_format, fields[: len(fiducial_format)], strict=True))
        else:
            continue

        record_ts = int(record["ftk_timestamp"])
        if record_ts != ts:
            if ts is not None:
                yield ts, markers, fiducials
            if record_ts in seen:
                warnprint(f"Non-contiguous ftk_timestamp {record_ts}; frame may be split")
            seen.add(record_ts)
            ts, markers, fiducials = record_ts, [], []

        (markers if fields[2] == "m" else fiducials).append(record)

    if ts is not None:
        yield ts, markers, fiducials


def _frame_plane(marker):
    """The board plane from a registered marker's own reported position and rotation."""
    position = np.array(
        [float(marker["x_position"]), float(marker["y_position"]), float(marker["z_position"])]
    )
    rotation = np.array(
        [
            [marker["r00"], marker["r01"], marker["r02"]],
            [marker["r10"], marker["r11"], marker["r12"]],
            [marker["r20"], marker["r21"], marker["r22"]],
        ],
        dtype=float,
    )
    return plane_from_rotation(position, rotation)


def plot_timechart(x, y, x_range, y_pred, debug_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(x, y, color="blue", label="Measurements", marker=".")
    ax.plot(x_range, y_pred, color="r", label="Fitted Model")
    ax.set_xlabel("Device timestamp")
    ax.set_ylabel("RocSync timestamp [ms]")
    ax.set_title("Frame timing")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.legend(loc="upper left")
    ax.grid(True)
    fig.savefig(f"{debug_dir}/timestamps.png")
    plt.close(fig)


def fit_ftk_timestamps(
    timestamps: dict[int, tuple[int, int]],
    frame_times: dict[int, int],
    debug_dir=None,
):
    """Fit board time against the tracker's own clock and describe the result.

    Thin wrapper around `summarize_timeline` -- the fit, dropout detection and
    reporting are exactly what the video path uses, just fed the tracker's raw
    microsecond ticks (`FTK_TICK_MS`) and a fixed inlier band instead of one derived
    from a frame period, since the tracker is not frame-periodic the way a container
    is. Raises ValueError when the timeline cannot be fitted.
    """
    from rocsync.timeline import summarize_timeline

    statistics, fit, _, _, _ = summarize_timeline(
        timestamps,
        frame_times,
        n_frames=len(frame_times),
        fps=None,
        source_tick_ms=FTK_TICK_MS,
        residual_threshold=FIT_RESIDUAL_THRESHOLD_MS,
        max_trials=10000,  # more trials for more consistent results
    )

    if debug_dir is not None:
        x = np.array(fit.order).reshape(-1, 1)
        y = np.array([timestamps[k][0] for k in fit.order])
        x_range = np.array([np.min(x), np.max(x)]).reshape(-1, 1)
        plot_timechart(x, y, x_range, fit.predict(x_range), debug_dir)
    return statistics


def process_ftk_recording(
    filename: str,
    debug_dir=None,
    board: BoardProfile | None = None,
    max_fiducials: int = MAX_FIDUCIALS,
) -> dict | None:
    """Fit a board timeline from a FusionTrack CSV recording.

    ``board`` names the profile to use for a frame whose marker id is not a registered
    geometry (``PROFILES_BY_FTK``); a recognised marker always overrides it, since the
    registration names the revision authoritatively. Pass it (``--board-version`` on
    the CLI) for a recording that never registers a geometry at all, which is the
    common case -- see the module docstring.
    """
    from tqdm import tqdm

    with open(filename) as file:
        total_lines = sum(1 for _ in file)

    timestamps = {}
    frame_times = {}
    n_marker_frames = 0
    n_pose_decodes = 0
    n_constellation_decodes = 0
    n_rejects = {}

    with open(filename) as file, tqdm(total=total_lines, desc="Processing lines") as pbar:
        for ftk_timestamp, markers, fiducials in _iter_frames(file, pbar):
            # Record every frame the tracker reported, whether or not it decodes: the
            # frame count and any dropouts are measured off this map.
            frame_times[ftk_timestamp] = ftk_timestamp

            registered = next((m for m in markers if int(m["marker_id"]) in PROFILES_BY_FTK), None)
            if registered is not None:
                n_marker_frames += 1
                frame_board = PROFILES_BY_FTK[int(registered["marker_id"])]
            else:
                frame_board = board
            if frame_board is None:
                continue

            points_3d = np.array(
                [
                    [float(f["x_position"]), float(f["y_position"]), float(f["z_position"])]
                    for f in fiducials
                ]
            )

            ax = None
            fig = None
            if debug_dir is not None and len(points_3d) >= 4:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.invert_yaxis()
                ax.grid(True)
                ax.set_aspect("equal")

            result = None
            if registered is not None:
                plane = _frame_plane(registered)
                result = decode_fiducials(
                    points_3d, frame_board, plane, max_fiducials=max_fiducials, ax=ax
                )
                if result.reject is None:
                    n_pose_decodes += 1

            # Always give the constellation search a chance -- measured live it carries
            # the large majority of decodes, so a pose reject (or no marker at all)
            # must not skip it.
            if result is None or result.reject is not None:
                result = decode_fiducials(
                    points_3d, frame_board, None, max_fiducials=max_fiducials, ax=ax
                )
                if result.reject is None:
                    n_constellation_decodes += 1

            if fig is not None:
                ax.set_title(f"{frame_board.name} {result.reject or result.plane_source}")
                fig.savefig(f"{debug_dir}/{ftk_timestamp}.png", bbox_inches="tight")
                plt.close(fig)

            if result.reject is None:
                timestamps[ftk_timestamp] = (result.ring_start, result.ring_end)
            else:
                n_rejects[result.reject] = n_rejects.get(result.reject, 0) + 1

    stats = {
        "n_marker_frames": n_marker_frames,
        "n_decoded": len(timestamps),
        "n_pose_decodes": n_pose_decodes,
        "n_constellation_decodes": n_constellation_decodes,
        "n_rejects": n_rejects,
    }
    if board is not None:
        stats["board_version"] = board.name

    if len(timestamps) > 0:
        try:
            statistics = fit_ftk_timestamps(timestamps, frame_times, debug_dir)
        except ValueError as e:
            errprint(f"Error: Unable to fit the FTK timeline: {e}")
            _print_decode_stats(stats)
            return None

        warn_about_statistics(statistics)
        print_statistics(statistics)
        return {**statistics.to_dict(), **stats}

    if board is None and n_marker_frames == 0:
        errprint(
            "Error: no registered geometry found in this recording; "
            "pass --board-version to decode from raw fiducials"
        )
    else:
        errprint("Error: no frame in this recording decoded a timestamp.")
    _print_decode_stats(stats)
    return None


def _print_decode_stats(stats: dict):
    """Why a recording did or didn't decode -- printed on every path so a run that
    ends in `Unable to time-sync` still says whether frames decoded at all."""
    print(
        f"Marker frames: {stats['n_marker_frames']}, decoded: {stats['n_decoded']} "
        f"(pose: {stats['n_pose_decodes']}, constellation: {stats['n_constellation_decodes']})"
    )
    for reason, count in stats["n_rejects"].items():
        print(f"  Rejected {count} frame(s): {reason}")
