import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from tqdm import tqdm

try:
    from rocsync.dataset import (
        VIDEO_SUFFIXES,
        DatasetConfig,
        add_common_args,
        load_video_time_sync,
        resolve_time_sync_json,
    )
    from rocsync.timecode import ms_to_timecode, timecode_to_ms
    from rocsync.timeline import affine_from_statistics, per_frame_times
except ImportError:
    raise SystemExit(
        "This script needs the rocsync package: pip install -e <path to RocSync>/sw"
    )


@dataclass
class Config(DatasetConfig):
    from_camera: str  # camera that defines the local time
    time_string: str  # time in that camera's local time (HH:MM:SS.mmm)


def get_screen_size():
    """
    Try to detect the screen resolution using tkinter.
    Fallback to 1920x1080 if anything goes wrong.
    """
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Check time synchronization by extracting a single moment from ALL cameras.\n"
            "You specify a local time in one camera (via --from-camera and --time); "
            "the script converts it to global time, checks that it lies inside the common "
            "overlap of all cameras, then shows the corresponding frame of each camera."
        )
    )
    add_common_args(p, __file__)
    p.add_argument(
        "--from-camera",
        required=True,
        help="Camera basename that defines the local time (e.g. 'canon33').",
    )
    p.add_argument(
        "--time",
        dest="time_string",
        required=True,
        help="Time in that camera's local time, format 'HH:MM:SS.mmm' (e.g. '00:05:00.000').",
    )
    return p


def compute_global_time_from_camera(
    time_string: str, camera_time_sync_data: dict
) -> float:
    """
    Convert a time given in camera-local units to GLOBAL ms through the camera's
    fitted clock. Local time is a position in the container's own timeline, which
    is what the fit maps, so the clock_offset_ms is the correct anchor.
    (Same math as a Clip built with from_camera_time, but for a single time.)
    """
    local_ms = timecode_to_ms(time_string)
    clock_rate, clock_offset_ms = affine_from_statistics(camera_time_sync_data)
    return float(clock_offset_ms + clock_rate * local_ms)


def validate_moment_in_overlap(
    global_ms: float, time_sync_data: Dict[str, dict]
) -> None:
    """
    Check that the chosen global moment lies inside the COMMON temporal overlap of all cameras:
        global_ms >= max(first_frame_i)
        global_ms <= min(last_frame_i)
    Raise ValueError if not.
    """
    if not time_sync_data:
        raise ValueError("Time synchronization JSON is empty – no cameras found.")

    first_frames = [float(d["first_frame"]) for d in time_sync_data.values()]
    last_frames = [float(d["last_frame"]) for d in time_sync_data.values()]

    overlap_start = max(first_frames)
    overlap_end = min(last_frames)

    if overlap_start >= overlap_end:
        raise ValueError(
            f"No temporal overlap between cameras:\n"
            f"  max(first_frame)={overlap_start:.2f} ms ({ms_to_timecode(overlap_start)}) >=\n"
            f"  min(last_frame)={overlap_end:.2f} ms ({ms_to_timecode(overlap_end)})."
        )

    if not (overlap_start <= global_ms <= overlap_end):
        raise ValueError(
            "Requested moment is OUTSIDE the common overlap of all cameras.\n"
            f"  Moment: {global_ms:.2f} ms ({ms_to_timecode(global_ms)})\n"
            f"  Overlap (all cameras):\n"
            f"    [{overlap_start:.2f} ms ({ms_to_timecode(overlap_start)}), "
            f"{overlap_end:.2f} ms ({ms_to_timecode(overlap_end)})]\n\n"
            "Choose a different --time (in the from-camera) that maps into this overlap."
        )


def extract_frame_for_moment(
    video_path: str, time_sync_data: dict, global_ms: float
) -> Optional[np.ndarray]:
    """
    For a given camera:
      - build its theoretical per-frame timestamps (in global ms),
      - find the frame whose timestamp is closest to global_ms,
      - grab that frame from the video and return it as an image (BGR).
    """
    if not os.path.exists(video_path):
        print(f"[WARN] Video not found: {video_path}")
        return None

    suffix = Path(video_path).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        print(f"[WARN] Not a recognized video format, skipping: {video_path}")
        return None

    timestamps = np.asarray(
        per_frame_times(video_path, time_sync_data),
        dtype=np.float64,
    )
    # Find closest frame
    idx = int(np.argmin(np.abs(timestamps - global_ms)))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[WARN] Could not read frame {idx} from {video_path}")
        return None

    return frame


def main(cfg: Config) -> None:
    dataset_folder = Path(cfg.dataset_folder)
    print(f"[INFO] Dataset folder: {dataset_folder}")
    print(f"[INFO] Using time sync JSON: {cfg.time_sync_json_path}")
    print(f"[INFO] From-camera: {cfg.from_camera}")
    print(f"[INFO] Local time in that camera: {cfg.time_string}")

    screen_w, screen_h = get_screen_size()
    time_sync_data = load_video_time_sync(cfg.time_sync_json_path)

    # Find the time-defining camera entry by basename match
    matching_key = next(
        (
            k
            for k in time_sync_data
            if os.path.splitext(os.path.basename(k))[0] == cfg.from_camera
        ),
        None,
    )
    if not matching_key:
        raise ValueError(
            f"No match found for camera basename '{cfg.from_camera}' in time sync JSON.\n"
            f"Available basenames: "
            f"{sorted({os.path.splitext(os.path.basename(k))[0] for k in time_sync_data.keys()})}"
        )

    time_defining_camera_data = time_sync_data[matching_key]

    # Convert local camera time -> global ms
    global_ms = compute_global_time_from_camera(
        cfg.time_string, time_defining_camera_data
    )
    print(
        f"[INFO] Local time '{cfg.time_string}' in '{cfg.from_camera}' "
        f"maps to global time: {global_ms:.2f} ms ({ms_to_timecode(global_ms)})"
    )

    # Validate that this moment lies in the common overlap of all cameras
    validate_moment_in_overlap(global_ms, time_sync_data)
    print(
        "[INFO] Moment lies inside the common overlap of all cameras. Extracting frames..."
    )

    # For each camera, extract and show the frame corresponding to this global time
    # Pre-extract frames for each camera for this global time
    frames_info = []  # list of (camera_basename, frame)
    for camera_rel_path, cam_sync_data in tqdm(
        time_sync_data.items(), desc="Extracting frames for each camera"
    ):
        video_path = dataset_folder / camera_rel_path
        camera_basename = os.path.splitext(os.path.basename(camera_rel_path))[0]

        frame = extract_frame_for_moment(str(video_path), cam_sync_data, global_ms)
        if frame is None:
            continue

        frames_info.append((camera_basename, frame))

    if not frames_info:
        print("[WARN] No frames could be extracted for any camera.")
        return

    print(
        "[INFO] Use RIGHT arrow to go forward, LEFT arrow to go back. "
        "Press 'q' or ESC to quit."
    )

    idx = 0
    window_name = "Time Sync Check"

    while True:
        camera_basename, frame = frames_info[idx]

        # --- Resize so width <= 500 px while keeping aspect ratio ---
        max_width = 1000
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame_display = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            h, w = new_h, new_w
        else:
            frame_display = frame.copy()
        # ------------------------------------------------------------

        title = (
            f"{camera_basename}  @  {cfg.time_string}  "
            f"(global: {ms_to_timecode(global_ms)})  "
            f"[{idx + 1}/{len(frames_info)}]"
        )

        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setWindowTitle(window_name, title)
        cv2.imshow(window_name, frame_display)

        # Try to get the actual window size (OpenCV >= 4.5)
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
        except Exception:
            win_w, win_h = w, h

        # Center window on screen
        x = max(0, int((screen_w - win_w) / 2))
        y = max(0, int((screen_h - win_h) / 2))
        cv2.moveWindow(window_name, x, y)

        key = cv2.waitKeyEx(0)

        # Handle quit: q or ESC
        if key in (27, ord("q"), ord("Q")):
            break

        # Handle LEFT / RIGHT arrows
        # OpenCV arrow key codes:
        #   left  = 81 or 2424832
        #   right = 83 or 2555904
        if key in (81, 2424832):  # LEFT
            idx = (idx - 1) % len(frames_info)
        elif key in (83, 2555904):  # RIGHT
            idx = (idx + 1) % len(frames_info)
        else:
            # Any other key: move forward
            idx = (idx + 1) % len(frames_info)

    cv2.destroyAllWindows()
    print("[INFO] Done. You’ve inspected this moment across all cameras.")


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    dataset_folder = Path(args.dataset_folder)
    time_sync_path = resolve_time_sync_json(dataset_folder, args.time_sync_json_path)

    cfg = Config(
        dataset_folder=str(dataset_folder),
        time_sync_json_path=str(time_sync_path),
        from_camera=args.from_camera,
        time_string=args.time_string,
    )
    main(cfg)
