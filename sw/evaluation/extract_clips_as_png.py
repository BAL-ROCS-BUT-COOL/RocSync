import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

try:
    from rocsync.clips import MAX_FRAMES_IN_FLIGHT, read_frames_at_indices
    from rocsync.timeline import affine_from_statistics, per_frame_times
except ImportError:
    raise SystemExit(
        "This script needs the rocsync package: pip install -e <path to RocSync>/sw"
    )

# -------------------------
# Config & CLI
# -------------------------


@dataclass
class Config:
    dataset_folder: str  # root folder containing subfolders with raw and internal calibration videos
    time_sync_json_path: str  # path to time sync json
    clips_to_extract_json: str  # path to clips json
    target_fps: float
    from_raw_camera_time_of_camera: Optional[str] = (
        None  # camera basename used to define clip timecodes (optional)
    )
    only_specified: bool = False  # if True, only process from_raw_camera_time_of_camera


def _auto_find_time_sync_json(dataset_folder: Path) -> Path:
    """
    Look for 'time sync/time_synchronization_*.json' inside dataset_folder.
    Prefer a single match; if multiple, pick the first in sorted order.
    (Same behavior as in extract_synced_videos.py)
    """
    base = dataset_folder / "time sync"
    candidates = sorted(base.glob("time_synchronization_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No time_synchronization_*.json found under: {base}")
    return candidates[0]


def _auto_find_clips_json(dataset_folder: Path) -> Path:
    """
    Prefer 'time sync/clips_config_all.json'; otherwise any *clips*.json in 'time sync/'.
    (Same behavior as in extract_synced_videos.py)
    """
    base = dataset_folder / "time sync"
    preferred = base / "clips_config_all.json"
    if preferred.exists():
        return preferred
    candidates = sorted(base.glob("*clips*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No clips config JSON found under: {base}. "
            f"Expected 'clips_config_all.json' or a file matching '*clips*.json'."
        )
    return candidates[0]


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Same style as extract_synced_videos.py:
    - dataset-folder defaults to the parent of the script's directory
    - auto-detect time sync and clips JSONs if not explicitly provided
    """
    try:
        script_parent_dir = Path(__file__).resolve().parent.parent
        default_dataset_folder = str(script_parent_dir)
    except NameError:
        default_dataset_folder = os.getcwd()
    except Exception:
        default_dataset_folder = os.getcwd()

    p = argparse.ArgumentParser(
        description="Extract synchronized clips as PNG sequences at a fixed target FPS."
    )
    p.add_argument(
        "--dataset-folder",
        default=default_dataset_folder,
        help=(
            "Root folder containing camera videos and a 'time sync' subfolder. "
            "Defaults to the parent folder of the script's location."
        ),
    )
    p.add_argument(
        "--target-fps",
        type=float,
        default=30,
        help="Output FPS for the sampled PNG sequences (e.g., 30).",
    )
    p.add_argument(
        "--from-camera",
        dest="from_raw_camera_time_of_camera",
        help="Camera basename that defines clip timecodes (optional).",
    )
    p.add_argument(
        "--only-specified",
        action="store_true",
        help="If set, only produce PNGs for the camera given by --from-camera.",
    )
    # Optional overrides (otherwise auto-detected relative to dataset_folder)
    p.add_argument(
        "--time-sync-json",
        dest="time_sync_json_path",
        help="Path to time_synchronization_*.json (optional).",
    )
    p.add_argument(
        "--clips-json",
        dest="clips_to_extract_json",
        help="Path to clips config JSON (optional).",
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

    # Use provided time sync JSON path
    time_sync_json_path = config.time_sync_json_path

    with open(time_sync_json_path, "r", encoding="utf-8") as f:
        time_sync_data = json.load(f)

    # Adjust keys to reflect relative raw video paths (same logic as original,
    # extended for Windows-style backslashes). Image and FTK device entries land
    # in the same JSON tagged with their own "type"; only video entries have the
    # per-frame timeline the overlap checks below need, so anything else is
    # dropped here.
    time_sync_data = {
        os.path.join(*camera.replace("\\", "/").split("/")[-2:]): data
        for camera, data in time_sync_data.items()
        if data.get("type") == "video"
    }

    # Parse clips, optionally using camera time to derive global times
    if config.from_raw_camera_time_of_camera is not None:
        matching_key = next(
            (
                k
                for k in time_sync_data
                if os.path.splitext(os.path.basename(k))[0]
                == config.from_raw_camera_time_of_camera
            ),
            None,
        )

        if not matching_key:
            raise ValueError(
                f"No match found for camera name: {config.from_raw_camera_time_of_camera}"
            )

        time_defining_camera_time_sync_data = time_sync_data[matching_key]
        defining_clock_rate, defining_clock_offset_ms = affine_from_statistics(
            time_defining_camera_time_sync_data
        )
        clips = _parse_clips_to_extract_json(
            config.clips_to_extract_json,
            from_camera_time=True,
            clock_offset_ms=defining_clock_offset_ms,
            clock_rate=defining_clock_rate,
        )
    else:
        clips = _parse_clips_to_extract_json(config.clips_to_extract_json)

    # For each clip and each camera, compute frames and save PNGs
    for clip in clips:
        for camera, camera_time_sync_data in time_sync_data.items():
            camera_basename = os.path.splitext(os.path.basename(camera))[0]

            # If --only-specified is set, skip all other cameras
            if (
                config.only_specified
                and config.from_raw_camera_time_of_camera is not None
            ):
                if camera_basename != config.from_raw_camera_time_of_camera:
                    continue

            raw_video_path = os.path.join(config.dataset_folder, camera)
            print(f"Processing {raw_video_path}.")
            if not os.path.exists(raw_video_path):
                print(f"No video found at {raw_video_path}. Will skip this camera.")
                continue

            # Build timestamps for each frame of this camera from the frames'
            # own presentation timestamps, so dropped frames stay where they are
            actual_timestamps = per_frame_times(raw_video_path, camera_time_sync_data)

            # Compute which frames to extract at target_fps
            frames_to_extract = _get_frames_to_extract(
                config.target_fps, actual_timestamps, clip.start, clip.end
            )

            # Output folder:
            camera_name = camera_basename
            # strip trailing "_raw" from camera folder name if present
            if camera_name.endswith("_raw"):
                camera_name = camera_name[:-4]

            output_folder = os.path.join(
                config.dataset_folder,
                output_folder_name,
                f"{clip.start_string_formatted}-{clip.end_string_formatted}",
                camera_name,
            )

            _save_frames_as_png(raw_video_path, frames_to_extract, output_folder)


# -------------------------
# Clip + helpers
# -------------------------


class Clip:
    def __init__(
        self,
        start_string: str,
        end_string: str,
        from_camera_time: bool = False,
        clock_offset_ms: float = None,
        clock_rate: float = None,
    ) -> None:
        if from_camera_time:
            start_string, end_string = self._convert_to_real_time(
                start_string, end_string, clock_offset_ms, clock_rate
            )
        self.start = self._parse_timecode_to_milliseconds(start_string)
        self.end = self._parse_timecode_to_milliseconds(end_string)
        self.start_string_formatted = self._parse_timecode_to_string_formatted(
            start_string
        )
        self.end_string_formatted = self._parse_timecode_to_string_formatted(end_string)

    def _convert_to_real_time(
        self,
        start_string: str,
        end_string: str,
        clock_offset_ms: float,
        clock_rate: float,
    ) -> tuple[str, str]:
        start = self._parse_timecode_to_milliseconds(start_string)
        end = self._parse_timecode_to_milliseconds(end_string)

        start_real = int(round(clock_offset_ms + clock_rate * start))
        end_real = int(round(clock_offset_ms + clock_rate * end))

        start_real_string = self._parse_timecode_to_milliseconds_inverse(start_real)
        end_real_string = self._parse_timecode_to_milliseconds_inverse(end_real)

        return start_real_string, end_real_string

    @staticmethod
    def _parse_timecode_to_milliseconds(timecode: str) -> int:
        dt = datetime.strptime(timecode, "%H:%M:%S.%f")
        delta = (
            dt.hour * 3600 + dt.minute * 60 + dt.second
        ) * 1000 + dt.microsecond // 1000
        return delta

    @staticmethod
    def _parse_timecode_to_milliseconds_inverse(ms: int) -> str:
        seconds, milliseconds = divmod(ms, 1000)
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"

    @staticmethod
    def _parse_timecode_to_string_formatted(timecode: str) -> str:
        dt = datetime.strptime(timecode, "%H:%M:%S.%f")
        timecode_formatted = dt.strftime("%H_%M_%S_%f")
        return timecode_formatted


def _parse_clips_to_extract_json(
    path: str,
    from_camera_time: bool = False,
    clock_offset_ms: float = None,
    clock_rate: float = None,
) -> List[Clip]:
    with open(path, "r", encoding="utf-8") as f:
        clips_raw = json.load(f)

    clips: List[Clip] = []
    for clip in clips_raw:
        c = Clip(
            clip["start"],
            clip["end"],
            from_camera_time=from_camera_time,
            clock_offset_ms=clock_offset_ms,
            clock_rate=clock_rate,
        )
        clips.append(c)

    return clips


def _get_frames_to_extract(
    target_fps: float,
    frame_to_timestamp_map: List[float],
    start_time: int,
    end_time: int,
) -> List[int]:
    """
    Compute which source frames to sample so the PNG sequence corresponds
    to [start_time, end_time] at target_fps.

    This is very close in spirit to the version used in extract_synced_videos.py,
    but keeps your original 'closest timestamp' logic based on a uniform
    time grid.
    """
    if target_fps <= 0:
        return []

    # Same idea as original: build a uniform grid at target_fps
    dt = 1000.0 / target_fps
    target_timestamps = np.arange(start_time, end_time, dt)
    actual_timestamps = np.array(frame_to_timestamp_map, dtype=np.float64)

    frames_to_extract: List[int] = []

    i = 0
    for t in tqdm(target_timestamps, desc="Matching timestamps"):
        # Move i forward while the next actual timestamp is closer to t
        while i + 1 < len(actual_timestamps) and abs(
            actual_timestamps[i + 1] - t
        ) < abs(actual_timestamps[i] - t):
            i += 1
        frames_to_extract.append(i)

    return frames_to_extract


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

    # Auto-detect relative config files unless explicitly provided
    if args.time_sync_json_path:
        time_sync_path = Path(args.time_sync_json_path)
    else:
        time_sync_path = _auto_find_time_sync_json(dataset_folder)

    if args.clips_to_extract_json:
        clips_path = Path(args.clips_to_extract_json)
    else:
        clips_path = _auto_find_clips_json(dataset_folder)

    cfg = Config(
        dataset_folder=str(dataset_folder),
        time_sync_json_path=str(time_sync_path),
        clips_to_extract_json=str(clips_path),
        target_fps=float(args.target_fps),
        from_raw_camera_time_of_camera=args.from_raw_camera_time_of_camera,
        only_specified=args.only_specified,
    )

    main(cfg)
