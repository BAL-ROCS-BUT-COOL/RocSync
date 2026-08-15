import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2

try:
    from rocsync.clips import (
        MAX_FRAMES_IN_FLIGHT,
        parse_clips_json,
        read_frames_at_indices,
        select_frame_indices,
    )
    from rocsync.dataset import (
        ClipExtractionConfig,
        add_clip_args,
        add_common_args,
        camera_name,
        load_video_time_sync,
        resolve_clips_json,
        resolve_time_sync_json,
        select_camera_key,
    )
    from rocsync.timeline import affine_from_statistics, per_frame_times
except ImportError:
    raise SystemExit(
        "This script needs the rocsync package: pip install -e <path to RocSync>/sw"
    )

# -------------------------
# Config & CLI
# -------------------------


@dataclass
class Config(ClipExtractionConfig):
    only_specified: bool = False  # if True, only process from_raw_camera_time_of_camera


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract synchronized clips as PNG sequences at a fixed target FPS."
    )
    add_common_args(p, __file__)
    add_clip_args(p)
    p.add_argument(
        "--only-specified",
        action="store_true",
        help="If set, only produce PNGs for the camera given by --from-camera.",
    )
    return p


# -------------------------
# Main logic
# -------------------------


def main(config: Config) -> None:
    print(f"Processing dataset folder: {config.dataset_folder}")

    if config.only_specified and not config.from_raw_camera_time_of_camera:
        raise ValueError(
            "--only-specified was set but no --from-camera was provided. "
            "Please specify which camera to process."
        )

    output_folder_name = "synced_clips_pngs"

    time_sync_data = load_video_time_sync(config.time_sync_json_path)

    # Parse clips, optionally using camera time to derive global times
    matching_key = None
    if config.from_raw_camera_time_of_camera is not None:
        matching_key = select_camera_key(
            time_sync_data, config.from_raw_camera_time_of_camera, "--from-camera"
        )
        time_defining_camera_time_sync_data = time_sync_data[matching_key]
        defining_clock_rate, defining_clock_offset_ms = affine_from_statistics(
            time_defining_camera_time_sync_data
        )
        clips = parse_clips_json(
            config.clips_to_extract_json,
            from_camera_time=True,
            clock_offset_ms=defining_clock_offset_ms,
            clock_rate=defining_clock_rate,
        )
    else:
        clips = parse_clips_json(config.clips_to_extract_json)

    # For each clip and each camera, compute frames and save PNGs
    for clip in clips:
        for camera, camera_time_sync_data in time_sync_data.items():
            # If --only-specified is set, skip all other cameras
            if config.only_specified and camera != matching_key:
                continue

            raw_video_path = os.path.join(config.dataset_folder, camera)
            print(f"Processing {raw_video_path}.")
            if not os.path.exists(raw_video_path):
                print(f"No video found at {raw_video_path}. Will skip this camera.")
                continue

            # Build timestamps for each frame of this camera from the frames'
            # own presentation timestamps, so dropped frames stay where they are
            actual_timestamps = per_frame_times(raw_video_path, camera_time_sync_data)

            # One source frame per output frame at target_fps
            frames_to_extract = select_frame_indices(
                config.target_fps, actual_timestamps, clip.start, clip.end
            )

            output_folder = os.path.join(
                config.dataset_folder,
                output_folder_name,
                f"{clip.start_string_formatted}-{clip.end_string_formatted}",
                camera_name(camera),
            )

            _save_frames_as_png(raw_video_path, frames_to_extract, output_folder)


# -------------------------
# Frame extraction & PNG writing
# -------------------------


def _save_frames_as_png(video_path: str, frames: List[int], output_folder: str) -> None:
    """
    Assumes frames is sorted ascendingly.

    Each frame is encoded as soon as it is decoded, so memory stays flat over a
    long clip. PNG compression is the slow part and cv2.imwrite drops the GIL,
    so the writes run on a pool -- bounded, because an unbounded one would just
    queue the whole clip in RAM again.
    """
    if not frames:
        return

    os.makedirs(output_folder, exist_ok=True)

    def write_one_frame(i: int, frame) -> None:
        cv2.imwrite(os.path.join(output_folder, f"{i:03d}.png"), frame)

    in_flight = threading.Semaphore(MAX_FRAMES_IN_FLIGHT)
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, frame in read_frames_at_indices(
            video_path, frames, desc="Extracting frames"
        ):
            in_flight.acquire()
            # The pool outlives the iteration, so it needs its own frame.
            future = executor.submit(write_one_frame, i, frame.copy())
            future.add_done_callback(lambda _: in_flight.release())
            futures.append(future)

        for future in futures:
            future.result()


# -------------------------
# Entry point
# -------------------------

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_folder = Path(args.dataset_folder)
    time_sync_path = resolve_time_sync_json(dataset_folder, args.time_sync_json_path)
    clips_path = resolve_clips_json(dataset_folder, args.clips_to_extract_json)

    cfg = Config(
        dataset_folder=str(dataset_folder),
        time_sync_json_path=str(time_sync_path),
        clips_to_extract_json=str(clips_path),
        target_fps=float(args.target_fps),
        from_raw_camera_time_of_camera=args.from_raw_camera_time_of_camera,
        only_specified=args.only_specified,
    )

    main(cfg)
