import math
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from rocsync.printer import *
from rocsync.timeline import detect_dropouts, fit_timeline, median_frame_period
from rocsync.video_statistics import VideoStatistics
from rocsync.vision import CameraType, process_frame


def read_frames_async(
    cap, frame_queue, skip_before_pts_ms=None, stop_after_pts_ms=None, stop_event=None
):
    def put(item):
        # Blocking put, but wake up regularly so a consumer that went away
        # (e.g. because it raised) cannot wedge this thread on a full queue.
        while stop_event is None or not stop_event.is_set():
            try:
                frame_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    # Frames before the window are grabbed but never retrieved, which skips
    # decoding them into an image. Seeking is deliberately not used: OpenCV maps
    # a requested time onto a frame index through the container's average frame
    # rate, which is exactly the quantity a dropped span invalidates -- on a file
    # with a 1.5 s hole, seeking to 2 s lands 8 frames past it.
    while stop_event is None or not stop_event.is_set():
        if not cap.grab():
            put((None, None, None))
            break

        # Read directly after grabbing, where this is the presentation timestamp
        # of the frame just grabbed, already relative to the stream start
        pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)

        if stop_after_pts_ms is not None and pts_ms > stop_after_pts_ms:
            put((None, None, None))
            break
        if skip_before_pts_ms is not None and pts_ms < skip_before_pts_ms:
            continue

        ret, frame = cap.retrieve()
        if not ret:
            put((None, None, None))
            break

        if not put((frame, frame_number, pts_ms)):
            break


def export_frames(video_path, output_path, fit, n_frames=None):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    # cap.set(cv2.CAP_PROP_FFMPEG_HWACCEL, cv2.CAP_FFMPEG_HWACCEL_NVDEC)  # try to use
    if not cap.isOpened():
        errprint(f"Error: Could not open video: {video_path}")
        return
    if n_frames is None:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(output_path, exist_ok=True)

    # Read frames in separate thread
    frame_queue = queue.Queue(maxsize=100)
    thread = threading.Thread(target=read_frames_async, args=(cap, frame_queue))
    thread.daemon = True
    thread.start()

    # Export frames concurrently using multiple threads
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        pbar = tqdm(total=n_frames, desc="Exporting frames", position=1)
        while True:
            frame, frame_number, pts_ms = frame_queue.get()  # blocking wait
            if frame is None:
                break
            timestamp = fit.clock_rate * pts_ms + fit.clock_offset_ms
            futures.append(
                executor.submit(
                    cv2.imwrite,
                    f"{output_path}/f{frame_number}_s{timestamp:.0f}.png",
                    frame,
                )
            )
            pbar.update(1)
        pbar.close()
        for future in futures:
            future.result()
    cap.release()


def process_video_window(
    video_path: str,
    camera_type: CameraType,
    window_start: int,
    window_end: int,
    stride=None,
    debug_dir: str = None,
    brightness_boost: int = None,
    board=None,
):
    cap = cv2.VideoCapture(video_path)

    # Extract video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # The window is a time span; presentation timestamps are in milliseconds
    window_start_ms = window_start * 1000.0
    window_end_ms = window_end * 1000.0

    # Read frames in separate thread. Each frame's own timestamp decides whether
    # it belongs to the window.
    frame_queue = queue.Queue(maxsize=100)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=read_frames_async,
        args=(cap, frame_queue, window_start_ms, window_end_ms, stop_event),
    )
    thread.daemon = True
    thread.start()

    timestamps = {}
    frame_times = {}
    scan_window = 0
    if stride is None:
        # One analyzed frame per second, or every frame if the container reports
        # no usable frame rate
        stride = int(fps) if fps >= 1 else 1

    expected_frames = n_frames
    if fps > 0 and math.isfinite(window_end - window_start):
        expected_frames = min(
            n_frames, int(math.ceil((window_end - window_start) * fps)) + 1
        )
    window_label = f"[{window_start:.3f}s, " + (
        "end]" if math.isinf(window_end) else f"{window_end:.3f}s]"
    )
    pbar = tqdm(
        total=expected_frames,
        desc=f"Analyzing frames in time window {window_label} --> Found {len(timestamps)} timestamps",
        position=1,
    )
    try:
        while True:
            frame, frame_number, pts_ms = frame_queue.get()  # blocking wait
            if frame is None:
                break
            pbar.update(1)

            # Record every frame that was read, whether or not it is analyzed:
            # the frame period and any dropouts are measured off this map.
            frame_times[frame_number] = pts_ms

            if not window_start_ms <= pts_ms <= window_end_ms:
                continue

            if scan_window > 0 or frame_number % stride == 0:
                rocsync_detected, timestamp = process_frame(
                    frame, camera_type, frame_number, board, debug_dir, brightness_boost
                )
                scan_window -= 1
                if timestamp is not None:
                    timestamps[frame_number] = timestamp
                if rocsync_detected:
                    scan_window = 5
                    pbar.set_description(
                        f"Analyzing frames in time window {window_label} --> Found {len(timestamps)} timestamps"
                    )
    finally:
        pbar.close()
        stop_event.set()
        thread.join(timeout=5)
        cap.release()

    return timestamps, frame_times


def process_video(
    video_path,
    camera_type,
    export_dir=None,
    stride=None,
    debug_dir=None,
    window1_start=None,
    window1_end=None,
    window2_start=None,
    window2_end=None,
    brightness_boost=None,
    board=None,
):
    # Get video metadata
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    # cap.set(cv2.CAP_PROP_FFMPEG_HWACCEL, cv2.CAP_FFMPEG_HWACCEL_NVDEC)  # try to use

    if not cap.isOpened():
        errprint(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Only used to resolve the window arguments and as a last-resort frame period;
    # the analyzed span is measured off the frames themselves further down. A
    # container that reports no frame rate leaves both to the timestamps.
    nominal_period = 1000 / fps if fps > 0 else None
    nominal_duration = (n_frames - 1) * nominal_period if nominal_period else 0

    # Whether the analyzed timeline covers the whole file, which decides if the
    # reported span and dropouts describe the file or just the requested windows
    timeline_windowed = any(
        w is not None for w in (window1_start, window1_end, window2_start, window2_end)
    )

    # An unspecified end means the end of the file. It must not be derived from
    # the frame count and average frame rate: on a file with dropped frames that
    # duration is far shorter than the real span, and would cut the read short.
    # Offsets from the end have no better reference available, so they resolve
    # against the nominal duration and are only as good as it is.
    if window1_start is None:
        window1_start = 0
    elif window1_start < 0:
        window1_start = max(0, (nominal_duration / 1000) + window1_start)

    if window1_end is None:
        window1_end = math.inf
    elif window1_end < 0:
        window1_end = max(0, (nominal_duration / 1000) + window1_end)

    if window2_start is None:
        window2_start = 0
    elif window2_start < 0:
        window2_start = max(0, (nominal_duration / 1000) + window2_start)

    if window2_end is None:
        window2_end = math.inf
    elif window2_end < 0:
        window2_end = max(0, (nominal_duration / 1000) + window2_end)

    windows = [(window1_start, window1_end)]
    if (
        window2_start > window1_end or window2_end < window1_start
    ):  # check if window2 is not overlapping with window1
        # TODO: better window checking
        windows.append((window2_start, window2_end))

    # Analyze frames
    timestamps = {}
    frame_times = {}
    window_frame_times = []
    for window_start, window_end in windows:
        window_timestamps, window_times = process_video_window(
            video_path,
            camera_type,
            window_start,
            window_end,
            stride,
            debug_dir,
            brightness_boost,
            board,
        )
        timestamps.update(window_timestamps)
        frame_times.update(window_times)
        window_frame_times.append(window_times)

    if len(timestamps) < 2:
        errprint("Error: Insufficient number of timestamped frames.")
        return
    if not frame_times:
        errprint("Error: No frames could be read.")
        return

    # Fit a robust linear model of board time against the frames' own
    # presentation timestamps. Both axes are in milliseconds, so the clock_rate is
    # the clock rate and the clock_offset_ms is the clock offset.
    period = median_frame_period(frame_times.values(), fallback=nominal_period)
    try:
        fit = fit_timeline(frame_times, timestamps, fallback_period=nominal_period)
    except ValueError as e:
        errprint(f"Error: Unable to fit the frame timeline: {e}")
        return

    if len(fit.order) < len(timestamps):
        warnprint(
            f"WARNING: {len(timestamps) - len(fit.order)} timestamped frames have no "
            f"presentation timestamp and were excluded from the fit."
        )

    # Assert that we have at least 80% inliers
    if np.sum(fit.inlier_mask) < 0.8 * len(fit.order):
        warnprint(
            f"WARNING: Estimated model has fewer than 80% inliers ({np.sum(fit.inlier_mask) / len(fit.order):.2%})."
        )
    if abs(fit.clock_rate - 1) > 0.05:
        warnprint(
            f"WARNING: Container clock runs at {fit.clock_rate:.4f}x board time; "
            f"expected approximately 1x."
        )

    # Dropouts are counted per window, so the untouched span between two
    # disjoint windows is not mistaken for missing frames
    n_gaps = n_dropped_frames = 0
    largest_gap_ms = 0.0
    gaps = []
    for window_times in window_frame_times:
        window_gaps, window_dropped, window_largest, found = detect_dropouts(
            window_times.values(), period
        )
        n_gaps += window_gaps
        n_dropped_frames += window_dropped
        largest_gap_ms = max(largest_gap_ms, window_largest)
        gaps.extend(found)
    if n_dropped_frames:
        warnprint(
            f"WARNING: {n_dropped_frames} frames missing from the container in "
            f"{n_gaps} gap(s), largest {largest_gap_ms / 1000:.3f} s."
        )

    # Add error to timestamps, following the order the fit used
    x = np.array([frame_times[k] for k in fit.order])
    y = np.array([timestamps[k][0] for k in fit.order])
    errors = fit.predict(x) - y
    annotated_timestamps = {
        frame_number: (*timestamps[frame_number], error)
        for frame_number, error in zip(fit.order, errors)
    }

    # Remove outliers
    filtered_timestamps = {
        k: annotated_timestamps[k]
        for k, is_inlier in zip(fit.order, fit.inlier_mask)
        if is_inlier
    }
    rejected_timestamps = {
        k: annotated_timestamps[k]
        for k, is_inlier in zip(fit.order, fit.inlier_mask)
        if not is_inlier
    }

    # Calculate statistics. The span is anchored on the first and last frame
    # actually present, so nothing is extrapolated to a frame that may not exist.
    fit_stats = fit.to_dict()
    pts_min, pts_max = min(frame_times.values()), max(frame_times.values())
    exposure_times = [end - start for start, end, _ in filtered_timestamps.values()]
    statistics = VideoStatistics(
        n_frames=n_frames,
        expected_duration=pts_max - pts_min,
        measured_duration=fit_stats["last_frame"] - fit_stats["first_frame"],
        expected_fps=fps,
        measured_fps=1000 / period,
        median_frame_period=period,
        n_gaps=n_gaps,
        n_dropped_frames=n_dropped_frames,
        largest_gap_ms=largest_gap_ms,
        timeline_windowed=timeline_windowed,
        mean_exposure_time=np.mean(exposure_times),
        min_exposure_time=np.min(exposure_times),
        max_exposure_time=np.max(exposure_times),
        std_exposure_time=np.std(exposure_times),
        considered_timestamps=filtered_timestamps,
        rejected_timestamps=rejected_timestamps,
        **fit_stats,
    )

    print_statistics(statistics)

    if debug_dir:
        plot_timechart(
            fit,
            filtered_timestamps,
            rejected_timestamps,
            frame_times,
            exposure_times,
            gaps,
            debug_dir,
        )
        plot_exposure_histogram(exposure_times, debug_dir)

    if export_dir:
        export_frames(video_path, export_dir, fit, n_frames)

    return statistics


def print_statistics(statistics: VideoStatistics):
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
        f"{statistics.n_dropped_frames} in {statistics.n_gaps} gap(s), max {statistics.largest_gap_ms/1000:.3f} s",
        statistics.n_dropped_frames == 0,
    )
    print(format_str.format("First frame:", f"{statistics.first_frame/1000:.3f} s"))
    print(format_str.format("Last frame:", f"{statistics.last_frame/1000:.3f} s"))
    print(
        format_str.format(
            "Framerate (nominal/measured):",
            f"{statistics.expected_fps:.3f}/{statistics.measured_fps:.3f} fps",
        )
    )
    print(
        format_str.format(
            "Clock rate (board/container):",
            f"{statistics.clock_rate:.6f}x",
        )
    )
    scope = "analyzed window" if statistics.timeline_windowed else "container"
    print(
        format_str.format(
            f"Duration ({scope}/board):",
            f"{statistics.expected_duration/1000:.3f}/{statistics.measured_duration/1000:.3f} s (Δ={statistics.measured_duration-statistics.expected_duration:.2f} ms)",
        )
    )
    print(
        format_str.format(
            "Exposure time (mean/min/max/std):",
            f"{statistics.mean_exposure_time:.2f}/{statistics.min_exposure_time:.2f}/{statistics.max_exposure_time:.2f}/{statistics.std_exposure_time:.2f} ms",
        )
    )
    print(71 * "-")


def plot_timechart(
    fit,
    filtered_timestamps,
    rejected_timestamps,
    frame_times,
    exposure_times,
    gaps,
    debug_dir,
):
    pts_min, pts_max = min(frame_times.values()), max(frame_times.values())
    span = np.array([pts_min, pts_max])
    x = np.array([frame_times[k] for k in filtered_timestamps]) / 1000
    y = np.array([start for start, _, _ in filtered_timestamps.values()])

    plt.figure()
    plt.scatter(x, y, color="blue", label="Measurements")
    plt.plot(span / 1000, fit.predict(span), color="blue", label="Fitted frametime")

    # A perfectly matched clock would run parallel to this, so drift shows up as
    # divergence from it rather than as an overall clock_rate
    plt.plot(
        span / 1000,
        fit.predict(pts_min) + (span - pts_min),
        color="red",
        label="Unscaled container clock",
    )

    if rejected_timestamps:
        plt.scatter(
            np.array([frame_times[k] for k in rejected_timestamps]) / 1000,
            [start for start, _, _ in rejected_timestamps.values()],
            color="red",
            marker="x",
            label="Rejected outliers",
        )
    for before, after, _ in gaps:
        plt.axvspan(before / 1000, after / 1000, color="grey", alpha=0.3)
    if gaps:
        plt.axvspan(np.nan, np.nan, color="grey", alpha=0.3, label="Dropped frames")

    plt.xlabel("Presentation timestamp [s]")
    plt.ylabel("Time relative to RocSync [ms]")
    plt.title("Frame timing")
    plt.gca().ticklabel_format(style="plain", useOffset=False)
    plt.legend()
    plt.grid(True)
    ax2 = plt.gca().twinx()
    ax2.scatter(x, exposure_times, color="green", label="Exposure time [ms]")
    ax2.set_ylabel("Exposure time [ms]")
    ax2.ticklabel_format(style="plain", useOffset=False)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(loc="upper right")
    plt.savefig(f"{debug_dir}/timestamps.png")


def plot_exposure_histogram(exposure_times, debug_dir):
    plt.figure()
    unique_values, counts = np.unique(exposure_times, return_counts=True)
    bar = plt.bar(unique_values, counts)
    plt.bar_label(bar, counts)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Exposure time [ms]")
    plt.ylabel("Number of measured frames")
    plt.title("Exposure time histogram")
    plt.savefig(f"{debug_dir}/exposure.png")
