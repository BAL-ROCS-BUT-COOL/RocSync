"""Dataset layout and command-line plumbing shared by the evaluation scripts.

A dataset folder holds one subfolder per camera plus a 'time sync' folder with
the synchronization JSON rocsync produced and the list of clips to extract. The
scripts differ in what they emit, but they all have to locate those two files,
read the JSON the same way and accept the same handful of flags.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# Use lowercase, dotted suffixes; we compare with .lower()
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv"})


@dataclass
class DatasetConfig:
    dataset_folder: str  # root folder containing the camera videos and 'time sync/'
    time_sync_json_path: str  # path to time_synchronization_*.json


@dataclass
class ClipExtractionConfig(DatasetConfig):
    clips_to_extract_json: str  # path to clips json
    target_fps: float
    # camera basename whose local time defines the clip timecodes (optional)
    from_raw_camera_time_of_camera: Optional[str] = None


def default_dataset_folder(script_file: Optional[str]) -> str:
    """The dataset root for a script that lives in a subfolder of it."""
    if not script_file:
        return os.getcwd()
    return str(Path(script_file).resolve().parent.parent)


def add_common_args(
    parser: argparse.ArgumentParser, script_file: Optional[str]
) -> None:
    """Add the dataset-folder and time-sync-JSON flags every evaluation script takes."""
    parser.add_argument(
        "--dataset-folder",
        default=default_dataset_folder(script_file),
        help=(
            "Root folder containing camera videos and a 'time sync' subfolder. "
            "Defaults to the parent folder of the script's location."
        ),
    )
    parser.add_argument(
        "--time-sync-json",
        dest="time_sync_json_path",
        help="Path to time_synchronization_*.json (optional; auto-detected if omitted).",
    )


def add_clip_args(parser: argparse.ArgumentParser) -> None:
    """Add the flags the clip-extraction scripts share on top of `add_common_args`."""
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30,
        help="Output FPS for the sampled clips (e.g., 30).",
    )
    parser.add_argument(
        "--from-camera",
        dest="from_raw_camera_time_of_camera",
        help="Camera basename that defines clip timecodes (optional).",
    )
    parser.add_argument(
        "--clips-json",
        dest="clips_to_extract_json",
        help="Path to clips config JSON (optional; auto-detected if omitted).",
    )


def resolve_time_sync_json(
    dataset_folder: Path, override: Optional[str] = None
) -> Path:
    """The time-sync JSON to use: `override`, else 'time sync/time_synchronization_*.json'.

    Takes the first match in sorted order if there are several.
    """
    if override:
        return Path(override)

    base = dataset_folder / "time sync"
    candidates = sorted(base.glob("time_synchronization_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No time_synchronization_*.json found under: {base}")
    return candidates[0]


def resolve_clips_json(dataset_folder: Path, override: Optional[str] = None) -> Path:
    """The clips config to use: `override`, else 'time sync/clips_config_all.json'.

    Falls back to any '*clips*.json' in the same folder.
    """
    if override:
        return Path(override)

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


def load_video_time_sync(path: str) -> Dict[str, dict]:
    """Read a time-synchronization JSON, keyed by '<parent>/<file>'.

    The keys are normalized to the last two path segments so they can be joined
    with the dataset folder, and Windows-style backslashes are accepted. Image
    and FTK device entries land in the same JSON tagged with their own "type";
    only video entries have the per-frame timeline the callers need, so anything
    else is dropped here.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, dict] = json.load(f)

    return {
        os.path.join(*camera.replace("\\", "/").split("/")[-2:]): data
        for camera, data in raw.items()
        if data.get("type") == "video"
    }
