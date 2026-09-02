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

from rocsync.clips import MAX_FRAMES_IN_FLIGHT
from rocsync.printer import errprint, print, printresult, warnprint
from rocsync.timeline import summarize_timeline
from rocsync.video_statistics import VideoStatistics
from rocsync.vision import CameraType, process_frame


def read_frames_async(
    cap, frame_queue, skip_before_pts_ms=None, stop_after_pts_ms=None, stop_event=None
):
    """Push (frame, frame number, pts) onto the queue until EOF or the window ends.

    Seeking is deliberately not used to reach the window: OpenCV maps a requested
    time onto a frame index through the container's average frame rate, which is
    exactly the quantity a dropped span invalidates -- on a file with a 1.5 s hole,
    seeking to 2 s lands 8 frames past it. The file is scanned from the start
    instead, and each frame's own presentation timestamp decides where it belongs.
    """

    def put(item):
        # Wake up regularly so a consumer that went away cannot wedge this thread
        while stop_event is None or not stop_event.is_set():
            try:
                frame_queue.put(item, timeout=0.1)
                return True
            except queue.Full:  # noqa: PERF203 - retrying is the point of the loop
                continue
        return False

    # Frames outside the window are grabbed but never retrieved into an image
    while stop_event is None or not stop_event.is_set():
        if not cap.grab():
            put((None, None, None))
            break

        # Straight after grabbing, this is the grabbed frame's own timestamp
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


def probe_last_pts_ms(video_path):
    """Presentation timestamp of the last frame, or None if no frame could be read.

    Seeking near the end is only a starting point: whatever frame the seek lands on,
    grabbing forward to EOF finds the true last frame, so a frame count the container
    reports wrongly cannot skew the result.
    """
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        # An over-reported frame count seeks past the end, so retry further back
        for first_frame in (max(0, n_frames - 1), max(0, n_frames // 2), 0):
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
            last_pts_ms = None
            while cap.grab():
                last_pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if last_pts_ms is not None:
                return last_pts_ms
    finally:
        cap.release()
    return None


def resolve_windows(windows, video_path):
    """Turns requested search windows into absolute [start, end] spans in seconds.

    A negative bound is an offset from the last frame's presentation timestamp. The
    result is sorted, and overlapping spans are merged so that no frame is scanned --
    and no gap between frames counted -- twice.
    """
    if not windows:
        return [(0.0, math.inf)]

    last_pts_s = None
    if any(bound < 0 for window in windows for bound in window):
        last_pts_ms = probe_last_pts_ms(video_path)
        if last_pts_ms is None:
            raise ValueError("no frame could be read to resolve a window bound given from the end")
        last_pts_s = last_pts_ms / 1000.0

    resolved = []
    for start, end in windows:
        if start < 0:
            start = max(0.0, last_pts_s + start)
        if end < 0:
            end = max(0.0, last_pts_s + end)
        if start >= end:
            raise ValueError(f"window [{start:.3f}s, {end:.3f}s] starts at or after it ends")
        resolved.append((start, end))

    resolved.sort()
    merged = [resolved[0]]
    for start, end in resolved[1:]:
        merged_start, merged_end = merged[-1]
        if start <= merged_end:  # overlapping or touching
            merged[-1] = (merged_start, max(merged_end, end))
        else:
            merged.append((start, end))
    return merged


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
    frame_queue = queue.Queue(maxsize=MAX_FRAMES_IN_FLIGHT)
    thread = threading.Thread(target=read_frames_async, args=(cap, frame_queue))
    thread.daemon = True
    thread.start()

    # Export frames concurrently, but only as fast as they can be encoded
    in_flight = threading.Semaphore(MAX_FRAMES_IN_FLIGHT)
    with ThreadPoolExecutor() as executor:
        futures = []
        pbar = tqdm(total=n_frames, desc="Exporting frames", position=1)
        while True:
            frame, frame_number, pts_ms = frame_queue.get()  # blocking wait
            if frame is None:
                break
            timestamp = fit.clock_rate * pts_ms + fit.clock_offset_ms
            in_flight.acquire()
            future = executor.submit(
                cv2.imwrite,
                f"{output_path}/f{frame_number}_s{timestamp:.0f}.png",
                frame,
            )
            future.add_done_callback(lambda _: in_flight.release())
            futures.append(future)
            pbar.update(1)
        pbar.close()
        for future in futures:
            future.result()
    cap.release()


def process_video_window(
    video_path: str,
    camera_type: CameraType,
    window_start: float,
    window_end: float,
    stride=None,
    debug_dir: str | None = None,
    board=None,
):
    cap = cv2.VideoCapture(video_path)

    # Extract video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # The window is a time span; presentation timestamps are in milliseconds
    window_start_ms = window_start * 1000.0
    window_end_ms = window_end * 1000.0

    # Read frames in separate thread
    frame_queue = queue.Queue(maxsize=MAX_FRAMES_IN_FLIGHT)
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
        # One analyzed frame per second, or every frame without a usable frame rate
        stride = int(fps) if fps >= 1 else 1

    expected_frames = n_frames
    if fps > 0 and math.isfinite(window_end - window_start):
        expected_frames = min(n_frames, math.ceil((window_end - window_start) * fps) + 1)
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

            # Every frame read, analyzed or not: period and dropouts come from this
            frame_times[frame_number] = pts_ms

            if not window_start_ms <= pts_ms <= window_end_ms:
                continue

            if scan_window > 0 or frame_number % stride == 0:
                rocsync_detected, timestamp = process_frame(
                    frame, camera_type, frame_number, board, debug_dir
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
    windows=None,
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

    # Whether the reported span and dropouts describe the file or just the windows
    timeline_windowed = bool(windows)

    try:
        windows = resolve_windows(windows, video_path)
    except ValueError as e:
        errprint(f"Error: Unable to resolve the search windows: {e}")
        return

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
            board,
        )
        timestamps.update(window_timestamps)
        frame_times.update(window_times)
        window_frame_times.append(window_times)

    # Fit board time against the frames' own presentation timestamps, both in ms
    try:
        statistics, fit, filtered_timestamps, rejected_timestamps, gaps = summarize_timeline(
            timestamps,
            frame_times,
            n_frames,
            fps,
            window_frame_times=window_frame_times,
            timeline_windowed=timeline_windowed,
        )
    except ValueError as e:
        errprint(f"Error: {e}")
        return

    if len(fit.order) < len(timestamps):
        warnprint(
            f"WARNING: {len(timestamps) - len(fit.order)} timestamped frames have no "
            f"presentation timestamp and were excluded from the fit."
        )

    # Warn below 80% inliers
    if np.sum(fit.inlier_mask) < 0.8 * len(fit.order):
        warnprint(
            f"WARNING: Estimated model has fewer than 80% inliers ({np.sum(fit.inlier_mask) / len(fit.order):.2%})."
        )
    if abs(fit.clock_rate - 1) > 0.05:
        warnprint(
            f"WARNING: Container clock runs at {fit.clock_rate:.4f}x board time; "
            f"expected approximately 1x."
        )

    if statistics.n_dropped_frames:
        warnprint(
            f"WARNING: {statistics.n_dropped_frames} frames missing from the container in "
            f"{statistics.n_gaps} gap(s), largest {statistics.largest_gap_ms / 1000:.3f} s."
        )

    print_statistics(statistics)

    if debug_dir:
        exposure_times = [end - start for start, end, _ in filtered_timestamps.values()]
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
        f"{statistics.n_dropped_frames} in {statistics.n_gaps} gap(s), max {statistics.largest_gap_ms / 1000:.3f} s",
        statistics.n_dropped_frames == 0,
    )
    print(format_str.format("First frame:", f"{statistics.first_frame / 1000:.3f} s"))
    print(format_str.format("Last frame:", f"{statistics.last_frame / 1000:.3f} s"))
    print(
        format_str.format(
            "Framerate (nominal/measured):",
            f"{statistics.nominal_fps:.3f}/{statistics.measured_fps:.3f} fps",
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
            f"{statistics.container_duration / 1000:.3f}/{statistics.board_duration / 1000:.3f} s (Δ={statistics.board_duration - statistics.container_duration:.2f} ms)",
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

    fig, ax = plt.subplots()
    ax.scatter(x, y, color="blue", label="Measurements")
    ax.plot(span / 1000, fit.predict(span), color="blue", label="Fitted frametime")

    # A matched clock runs parallel to this, so drift shows as divergence from it
    ax.plot(
        span / 1000,
        fit.predict(pts_min) + (span - pts_min),
        color="red",
        label="Unscaled container clock",
    )

    if rejected_timestamps:
        ax.scatter(
            np.array([frame_times[k] for k in rejected_timestamps]) / 1000,
            [start for start, _, _ in rejected_timestamps.values()],
            color="red",
            marker="x",
            label="Rejected outliers",
        )
    for before, after, _ in gaps:
        ax.axvspan(before / 1000, after / 1000, color="grey", alpha=0.3)
    if gaps:
        ax.axvspan(np.nan, np.nan, color="grey", alpha=0.3, label="Dropped frames")

    ax.set_xlabel("Presentation timestamp [s]")
    ax.set_ylabel("Time relative to RocSync [ms]")
    ax.set_title("Frame timing")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.legend()
    ax.grid(True)
    ax2 = ax.twinx()
    ax2.scatter(x, exposure_times, color="green", label="Exposure time [ms]")
    ax2.set_ylabel("Exposure time [ms]")
    ax2.ticklabel_format(style="plain", useOffset=False)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(loc="upper right")
    fig.savefig(f"{debug_dir}/timestamps.png")
    plt.close(fig)


def plot_exposure_histogram(exposure_times, debug_dir):
    fig, ax = plt.subplots()
    unique_values, counts = np.unique(exposure_times, return_counts=True)
    bar = ax.bar(unique_values, counts)
    ax.bar_label(bar, counts)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Exposure time [ms]")
    ax.set_ylabel("Number of measured frames")
    ax.set_title("Exposure time histogram")
    fig.savefig(f"{debug_dir}/exposure.png")
    plt.close(fig)
