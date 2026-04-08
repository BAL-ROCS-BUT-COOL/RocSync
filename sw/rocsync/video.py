import math
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm

from rocsync.printer import *
from rocsync.video_statistics import VideoStatistics
from rocsync.vision import CameraType, aruco_detector, aruco_marker_id, process_frame


def read_frames_async(cap, frame_queue, start_frame=0, end_frame=None):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put((None, None))
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
        if end_frame is not None and frame_number >= end_frame:
            frame_queue.put((None, None))
            break

        frame_queue.put((frame, frame_number))


def export_frame_async(frame_queue, y_pred, path):
    frame, frame_number = frame_queue.get()  # blocking wait
    if frame is None:
        errprint("Error: Input stream ended unexpectedly.")
        return
    cv2.imwrite(f"{path}/f{frame_number}_s{y_pred[frame_number]:.0f}.png", frame)


def export_frames(video_path, output_path, y_pred):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    # cap.set(cv2.CAP_PROP_FFMPEG_HWACCEL, cv2.CAP_FFMPEG_HWACCEL_NVDEC)  # try to use
    if not cap.isOpened():
        errprint(f"Error: Could not open video: {video_path}")
        return
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
        for _ in range(n_frames):
            futures.append(executor.submit(export_frame_async, frame_queue, y_pred, output_path))
        for future in tqdm(
            as_completed(futures), total=n_frames, desc="Exporting frames", position=1
        ):
            future.result()
    cap.release()


def probe_frame(cap, frame_number):
    """Seek to a frame and run ArUco detection only. Returns True if the board is detected."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        return False
    clahe = cv2.createCLAHE(tileGridSize=(8, 8))
    gray = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    _, marker_ids, _ = aruco_detector.detectMarkers(gray)
    if marker_ids is None:
        return False
    return aruco_marker_id in marker_ids.flatten()


def expand_window(cap, hit_frame, fps, n_frames, cooldown_s=0.5):
    """Expand from a hit frame forward and backward until the board disappears for cooldown_s."""
    cooldown_frames = int(cooldown_s * fps)

    # Expand forward
    last_hit = hit_frame
    frame_num = hit_frame + 1
    while frame_num < n_frames and (frame_num - last_hit) <= cooldown_frames:
        if probe_frame(cap, frame_num):
            last_hit = frame_num
        frame_num += 1
    end_frame = last_hit

    # Expand backward
    last_hit = hit_frame
    frame_num = hit_frame - 1
    while frame_num >= 0 and (last_hit - frame_num) <= cooldown_frames:
        if probe_frame(cap, frame_num):
            last_hit = frame_num
        frame_num -= 1
    start_frame = last_hit

    return start_frame, end_frame


def find_windows(cap, fps, n_frames, expected_windows=2, granularity_s=1.0):
    """Binary temporal search for windows where the RocSync board is visible."""
    granularity_frames = max(1, int(granularity_s * fps))
    windows = []

    def search(start, end):
        if len(windows) >= expected_windows:
            return
        if end - start < granularity_frames:
            return
        mid = (start + end) // 2
        if probe_frame(cap, mid):
            # Found the board — expand to find full window
            w_start, w_end = expand_window(cap, mid, fps, n_frames)
            windows.append((w_start, w_end))
            # Recurse into regions outside the found window
            search(start, w_start - 1)
            search(w_end + 1, end)
        else:
            # Not found — split and search both halves
            search(start, mid)
            search(mid + 1, end)

    search(0, n_frames - 1)
    windows.sort()
    return windows


def process_video_window(video_path: str, camera_type: CameraType, window_start: int, window_end: int, stride=None, debug_dir: str = None, brightness_boost: int = None, try_hard: bool = False):
    cap = cv2.VideoCapture(video_path)

    # Extract video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(max(0, math.floor(window_start * fps)))
    end_frame = int(min(math.ceil(window_end * fps) + 1, cap.get(cv2.CAP_PROP_FRAME_COUNT))) # end_frame is exclusive

    # Read frames in separate thread
    frame_queue = queue.Queue(maxsize=100)
    thread = threading.Thread(target=read_frames_async, args=(cap, frame_queue, start_frame, end_frame))
    thread.daemon = False
    thread.start()

    timestamps = {}
    scan_window = 0
    if stride is None:
        stride = int(fps)
    
    pbar = tqdm(range(start_frame, end_frame), desc=f"Analyzing frames in time window [{window_start:.3f}s, {window_end:.3f}s] --> Found {len(timestamps)} timestamps", position=1)
    for _ in pbar:
        frame, frame_number = frame_queue.get()  # blocking wait
        if frame is None:
            errprint("Error: Input stream ended unexpectedly. Could be a sign of skipped frames.")
            break
        if scan_window > 0 or frame_number % stride == 0:
            rocsync_detected, timestamp = process_frame(frame, camera_type, frame_number, debug_dir, brightness_boost, try_hard)
            scan_window -= 1
            if timestamp is not None:
                timestamps[frame_number] = timestamp
            if rocsync_detected:
                scan_window = 5
                pbar.set_description(f"Analyzing frames in time window [{window_start:.3f}s, {window_end:.3f}s] --> Found {len(timestamps)} timestamps")


    thread.join()
    cap.release()
            
    return timestamps


def process_video(video_path, camera_type, export_dir=None, stride=None, debug_dir=None, window1_start=None, window1_end=None, window2_start=None, window2_end=None, brightness_boost=None, try_hard=False, search_windows=False, expected_windows=2, search_granularity=1.0):
    # Get video metadata
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    # cap.set(cv2.CAP_PROP_FFMPEG_HWACCEL, cv2.CAP_FFMPEG_HWACCEL_NVDEC)  # try to use

    if not cap.isOpened():
        errprint(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_duration = (n_frames - 1) / fps * 1000
    cap.release()

    if not search_windows:
        # Use manually specified windows
        if window1_start is None:
            window1_start = 0
        elif window1_start < 0:
            window1_start = max(0, (expected_duration / 1000) + window1_start)

        if window1_end is None:
            window1_end = expected_duration / 1000
        elif window1_end < 0:
            window1_end = max(0, (expected_duration / 1000) + window1_end)

        if window2_start is None:
            window2_start = 0
        elif window2_start < 0:
            window2_start = max(0, (expected_duration / 1000) + window2_start)

        if window2_end is None:
            window2_end = expected_duration / 1000
        elif window2_end < 0:
            window2_end = max(0, (expected_duration / 1000) + window2_end)

        timestamps = process_video_window(video_path, camera_type, window1_start, window1_end, stride, debug_dir, brightness_boost, try_hard)

        if window2_start > window1_end or window2_end < window1_start:
            timestamps2 = process_video_window(video_path, camera_type, window2_start, window2_end, stride, debug_dir, brightness_boost, try_hard)
            timestamps = {**timestamps, **timestamps2}
    else:
        # Auto-discover windows via binary temporal search
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            errprint(f"Error: Could not open video: {video_path}")
            return
        print(f"Searching for temporal windows (expected: {expected_windows}, granularity: {search_granularity}s)...")
        windows = find_windows(cap, fps, n_frames, expected_windows, search_granularity)
        cap.release()

        if not windows:
            errprint("Error: Could not find the RocSync board in any frame.")
            return

        for w_start, w_end in windows:
            print(f"  Found window: [{w_start / fps:.3f}s, {w_end / fps:.3f}s]")

        timestamps = {}
        for w_start, w_end in windows:
            window_timestamps = process_video_window(video_path, camera_type, w_start / fps, w_end / fps, stride, debug_dir, brightness_boost, try_hard)
            timestamps = {**timestamps, **window_timestamps}
    

    if len(timestamps) < 2:
        errprint("Error: Insufficient number of timestamped frames.")
        return

    # Assuming constant frame rate, fit robust linear model
    x = np.array(list(timestamps.keys())).reshape(-1, 1)
    y = np.array([start for start, _ in timestamps.values()])
    model = RANSACRegressor(
        residual_threshold=1000 / fps,  # max one frame deviation
        max_trials=1000,  # more trials for more consistent results
        random_state=0,  # deterministic results
    )
    model.fit(x, y)

    # Assert that we have at least 80% inliers
    if np.sum(model.inlier_mask_) < 0.8 * len(timestamps):
        warnprint(f"WARNING: Estimated model has fewer than 80% inliers ({np.sum(model.inlier_mask_) / len(timestamps):.2%}).")

    # Predict timestamps for all frames
    x_range = np.arange(0, n_frames).reshape(-1, 1)
    y_pred = model.predict(x_range)

    # Add error to timestamps
    errors = model.predict(x) - y
    timestamps = {
        frame_number: (start, end, error)
        for (frame_number, (start, end)), error in zip(timestamps.items(), errors)
    }

    # Remove outliers
    filtered_timestamps = {
        k: v for i, (k, v) in enumerate(timestamps.items()) if model.inlier_mask_[i]
    }
    rejected_timestamps = {
        k: v for i, (k, v) in enumerate(timestamps.items()) if not model.inlier_mask_[i]
    }
    filtered_x = np.array(list(filtered_timestamps.keys())).reshape(-1, 1)
    filtered_y = np.array([start for start, _, _ in filtered_timestamps.values()])

    # Calculate statistics
    exposure_times = [end - start for start, end, _ in filtered_timestamps.values()]
    measured_duration = y_pred[-1] - y_pred[0]
    statistics = VideoStatistics(
        n_frames=n_frames,
        n_considered_frames=len(filtered_timestamps),
        n_rejected_frames=len(timestamps) - len(filtered_timestamps),
        r2_before=model.score(x, y),
        rmse_before=root_mean_squared_error(y, model.predict(x)),
        r2_after=model.score(filtered_x, filtered_y),
        rmse_after=root_mean_squared_error(filtered_y, model.predict(filtered_x)),
        expected_duration=expected_duration,
        measured_duration=measured_duration,
        expected_fps=fps,
        measured_fps=(n_frames - 1) / measured_duration * 1000,
        speed_factor=measured_duration / expected_duration,
        first_frame=y_pred[0],
        last_frame=y_pred[-1],
        mean_exposure_time=np.mean(exposure_times),
        min_exposure_time=np.min(exposure_times),
        max_exposure_time=np.max(exposure_times),
        std_exposure_time=np.std(exposure_times),
        considered_timestamps=filtered_timestamps,
        rejected_timestamps=rejected_timestamps,
        # interpolated_timestamps=y_pred.tolist(),
    )

    print_statistics(statistics)

    if debug_dir:
        plot_timechart(
            filtered_x, filtered_y, x_range, y_pred, exposure_times, expected_duration, debug_dir
        )
        plot_exposure_histogram(exposure_times, debug_dir)

    if export_dir:
        export_frames(video_path, export_dir, y_pred)

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
    print(format_str.format("First frame:", f"{statistics.first_frame/1000:.3f} s"))
    print(format_str.format("Last frame:", f"{statistics.last_frame/1000:.3f} s"))
    print(
        format_str.format(
            "Framerate (expected/measured):",
            f"{statistics.expected_fps:.3f}/{statistics.measured_fps:.3f} fps ({statistics.speed_factor:.6f}x)",
        )
    )
    print(
        format_str.format(
            "Duration (expected/measured):",
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


def plot_timechart(x, y, x_range, y_pred, exposure_times, expected_duration, debug_dir):
    plt.figure()
    plt.scatter(x, y, color="blue", label="Measurements")
    plt.plot(x_range, y_pred, color="blue", label="Measured frametime")
    plt.plot(
        x_range,
        np.linspace(y_pred[0], y_pred[0] + expected_duration, len(x_range)),
        color="red",
        label="Calculated frametime",
    )
    plt.xlabel("Frame number")
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
