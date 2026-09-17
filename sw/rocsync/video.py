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
from rocsync.printer import errprint, warnprint
from rocsync.recording_statistics import print_statistics, warn_about_statistics
from rocsync.timecode import resolve_windows
from rocsync.timeline import source_frame_period_ms, summarize_timeline
from rocsync.video_reader import VideoReader
from rocsync.vision import CameraType, process_frame


def _produce_frames(reader, frame_queue, start_index, stop_index, stop_event):
    """Push (frame, frame number, pts) onto the queue for `reader.frames(start_index,
    stop_index)`, until exhausted, EOF, or `stop_event` fires.
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

    for index, pts_ms, frame in reader.frames(start_index, stop_index):
        if stop_event is not None and stop_event.is_set():
            return
        if not put((frame, index, pts_ms)):
            return
    put((None, None, None))


def export_frames(video_path, output_path, fit, n_frames=None):
    try:
        reader = VideoReader(video_path)
    except OSError as e:
        errprint(f"Error: {e}")
        return
    if n_frames is None:
        n_frames = reader.reported_frame_count
    os.makedirs(output_path, exist_ok=True)

    # Read frames in separate thread
    frame_queue = queue.Queue(maxsize=MAX_FRAMES_IN_FLIGHT)
    thread = threading.Thread(target=_produce_frames, args=(reader, frame_queue, 0, None, None))
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
    reader.close()


def process_video_window(
    video_path: str,
    camera_type: CameraType,
    window_start: float,
    window_end: float,
    stride=None,
    debug_dir: str | None = None,
    board=None,
    try_hard=False,
    reader=None,
):
    owns_reader = reader is None
    if owns_reader:
        try:
            reader = VideoReader(video_path)
        except OSError as e:
            errprint(f"Error: {e}")
            return {}, {}

    fps = reader.fps

    # The window is a time span; presentation timestamps are in milliseconds
    window_start_ms = window_start * 1000.0
    window_end_ms = window_end * 1000.0

    # Only a window that actually restricts something needs the exact pts index
    exact = window_start > 0 or math.isfinite(window_end)
    if exact:
        start_index = reader.index_at(window_start_ms)
        stop_index = reader.index_at(window_end_ms, side="right") if math.isfinite(window_end) else len(reader)
        expected_frames = max(0, stop_index - start_index)
    else:
        start_index, stop_index = 0, None
        expected_frames = reader.reported_frame_count

    # Read frames in separate thread
    frame_queue = queue.Queue(maxsize=MAX_FRAMES_IN_FLIGHT)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_produce_frames,
        args=(reader, frame_queue, start_index, stop_index, stop_event),
    )
    thread.daemon = True
    thread.start()

    timestamps = {}
    frame_times = {}
    scan_window = 0
    if stride is None:
        # One analyzed frame per second, or every frame without a usable frame rate
        stride = int(fps) if fps >= 1 else 1

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

            if scan_window > 0 or frame_number % stride == 0:
                rocsync_detected, timestamp = process_frame(
                    frame, camera_type, frame_number, board, debug_dir, try_hard=try_hard
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
        if owns_reader:
            reader.close()

    return timestamps, frame_times


def process_video(
    video_path,
    camera_type,
    export_dir=None,
    stride=None,
    debug_dir=None,
    windows=None,
    board=None,
    try_hard=False,
):
    try:
        reader = VideoReader(video_path)
    except OSError as e:
        errprint(f"Error: {e}")
        return

    fps = reader.fps
    n_frames = reader.reported_frame_count

    # The true sensor frame period, immune to a file that is itself a decimated clip --
    # sizes the clock fit's inlier band instead of the possibly-subsampled frame spacing
    frame_period_ms = source_frame_period_ms(video_path)

    # Whether the reported span and dropouts describe the file or just the windows
    timeline_windowed = bool(windows)

    try:
        windows = resolve_windows(windows, lambda: reader.pts[-1] / 1000.0 if reader.pts else None)
    except ValueError as e:
        errprint(f"Error: Unable to resolve the search windows: {e}")
        reader.close()
        return

    # Analyze frames, all windows sharing this file's one pts index
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
            try_hard,
            reader=reader,
        )
        timestamps.update(window_timestamps)
        frame_times.update(window_times)
        window_frame_times.append(window_times)
    reader.close()

    # Fit board time against the frames' own presentation timestamps, both in ms
    try:
        statistics, fit, filtered_timestamps, rejected_timestamps, gaps = summarize_timeline(
            timestamps,
            frame_times,
            n_frames,
            fps,
            window_frame_times=window_frame_times,
            timeline_windowed=timeline_windowed,
            frame_period_ms=frame_period_ms,
        )
    except ValueError as e:
        errprint(f"Error: {e}")
        return

    if len(fit.order) < len(timestamps):
        warnprint(
            f"WARNING: {len(timestamps) - len(fit.order)} timestamped frames have no "
            f"presentation timestamp and were excluded from the fit."
        )

    warn_about_statistics(statistics)
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
