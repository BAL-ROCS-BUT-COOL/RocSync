import argparse
import json
import math
import os
import pathlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from rocsync.board_profiles import PROFILES_BY_NAME
from rocsync.dataset import VIDEO_SUFFIXES
from rocsync.ftk import process_ftk_recording
from rocsync.printer import errprint, succprint, warnprint
from rocsync.timecode import parse_hms
from rocsync.video import process_video
from rocsync.vision import CameraType, process_frame


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def process_image(path, camera_type, debug_dir=None, board=None):
    image = cv2.imread(path)
    _, timestamp = process_frame(image, camera_type, 0, board, debug_dir)
    if timestamp is not None:
        succprint(f"start: {timestamp[0]} ms, end {timestamp[1]} ms")
        return {"start": timestamp[0], "end": timestamp[1]}
    else:
        errprint("Error: Unable to decode timestamp.")


def mkdir_unique(name, parent_dir):
    parent_path = Path(parent_dir)
    debug_dir = parent_path / name
    if not debug_dir.exists():
        debug_dir.mkdir(parents=True)
    else:
        counter = 2
        debug_dir = parent_path / f"{name} ({counter})"
        while debug_dir.exists():
            counter += 1
            debug_dir = parent_path / f"{name} ({counter})"
        debug_dir.mkdir(parents=True)
    return str(debug_dir)


WINDOW_TIME_FORMATS = "hh:mm:ss, 'end' or 'end-hh:mm:ss'"


def parse_time(time_str: str) -> float:
    """Parses a time in hh:mm:ss format, "end" for the end of the file, or
    "end-hh:mm:ss" for an offset back from it.

    The seconds may be fractional, so a window can be given to the millisecond.
    An offset from the end is returned as a negative time.
    """
    text = time_str.strip().lower()
    if text == "end":
        return math.inf
    # A bare leading "-" means the same, but needs quoting on a command line
    for prefix in ("end-", "-"):
        if text.startswith(prefix):
            return -parse_hms(text[len(prefix) :], time_str, WINDOW_TIME_FORMATS)

    return parse_hms(text, time_str, WINDOW_TIME_FORMATS)


def main():
    parser = argparse.ArgumentParser(
        description="Extract timestamps from images and videos showing the RocSync device."
    )
    parser.add_argument(
        "path",
        type=str,
        metavar="PATH",
        nargs="+",
        help="path to a video, image, or directory containing videos and/or images",
    )
    parser.add_argument(
        "-c",
        "--camera_type",
        choices=[e.value for e in CameraType],
        default=CameraType.RGB.value,
        help="specify the type of camera (default: rgb)",
    )
    parser.add_argument(
        "-s",
        "--stride",
        type=int,
        metavar="N",
        help="scan every N-th frame only (default: same as framerate, only applies to videos)",
    )
    parser.add_argument(
        "-e",
        "--export_frames",
        type=str,
        metavar="DIRECTORY",
        help="directory to store all raw frames as PNGs with timestamp (only applies to videos)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        metavar="FILE",
        help="JSON file to store results",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="do not ask for confirmation when processing multiple files",
    )
    parser.add_argument(
        "--debug",
        type=str,
        metavar="DIRECTORY",
        help="directory to store debug images (very slow)",
    )
    parser.add_argument(
        "--board-version",
        choices=["auto"] + list(PROFILES_BY_NAME.keys()),
        default="auto",
        help="board hardware revision (default: auto-detect from ArUco marker ID)",
    )

    # Specify time windows to search for ROCsync
    parser.add_argument(
        "--window",
        nargs=2,
        action="append",
        metavar=("START", "END"),
        help="time span to search, in hh:mm:ss format with optionally fractional "
        "seconds; 'end' is the end of the file and 'end-hh:mm:ss' counts back from it, "
        "e.g. --window end-0:00:30 end. Repeat for several spans; overlapping ones are "
        "merged (default: whole file)",
    )
    parser.add_argument(
        "--recurse_in_dir",
        action="store_true",
        help="recursively search for videos and images in directories",
    )

    args = parser.parse_args()

    # Auto-detection identifies the board from its ArUco marker, which is invisible in IR
    if args.camera_type == CameraType.INFRARED.value and args.board_version == "auto":
        parser.error(
            f"-c {CameraType.INFRARED.value} requires an explicit --board-version "
            f"(choices: {', '.join(PROFILES_BY_NAME)}); auto-detection needs the "
            "ArUco marker, which is not visible in IR"
        )

    board = (
        PROFILES_BY_NAME.get(args.board_version)
        if args.board_version != "auto"
        else None
    )

    # Parse the search windows; they are resolved against the video and merged later
    windows = []
    for start_str, end_str in args.window or []:
        try:
            start, end = parse_time(start_str), parse_time(end_str)
        except ValueError as e:
            parser.error(f"argument --window: {e}")
        if start >= 0 and start >= end:
            parser.error(
                f"argument --window: start {start_str} is not before end {end_str}"
            )
        windows.append((start, end))

    files = set()
    for path in args.path:
        path_obj = Path(path)
        if path_obj.is_dir():
            # walk dir recursively
            for file in (
                path_obj.rglob("*") if args.recurse_in_dir else path_obj.glob("*")
            ):
                if file.is_file():
                    files.add(file.resolve())
        elif path_obj.is_file():
            files.add(path_obj.resolve())
        else:
            errprint(f"Invalid path: {path}")
            return

    videos = sorted([f for f in files if f.suffix.lower() in VIDEO_SUFFIXES])
    images = sorted([f for f in files if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    ftk_recordings = sorted([f for f in files if f.suffix.lower() == ".csv"])

    if len(videos) + len(images) + len(ftk_recordings) > 1:
        print(
            f"Found {len(videos)} videos, {len(images)} images, and {len(ftk_recordings)} ftk recordings:"
        )
        for file in videos + images + ftk_recordings:
            print(f"    {file}")
        while True and not args.yes:
            response = input("Do you want to continue (Y/n): ").strip().lower()
            if response in ["y", "yes", ""]:
                break
            elif response in ["n", "no"]:
                return
            else:
                print("Please enter 'y' or 'n'.")

    if args.debug:
        os.makedirs(args.debug, exist_ok=True)
        warnprint(f"Debug images will be stored in {args.debug}")

    if args.export_frames:
        os.makedirs(args.export_frames, exist_ok=True)
        warnprint(f"Exported frames will be stored in {args.export_frames}")

    result = {}
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with output_path.open("r") as file:
            result = json.load(file)
        print(f"Loaded previous results from {args.output}")

    for file in tqdm(
        videos + images + ftk_recordings, desc="Processing files", position=0
    ):
        if str(file) in result:
            print(f"Skipping {file}, already processed.")
            continue
        print(f"Working on {file}")

        debug_dir = None
        if args.debug:
            name, _ = os.path.splitext(os.path.basename(file))
            debug_dir = mkdir_unique(name, args.debug)

        export_dir = None
        if args.export_frames:
            name, _ = os.path.splitext(os.path.basename(file))
            export_dir = mkdir_unique(name, args.export_frames)

        ret = None
        entry_type = None
        if file in videos:
            entry_type = "video"
            statistics = process_video(
                file,
                CameraType(args.camera_type),
                export_dir=export_dir,
                stride=args.stride,
                debug_dir=debug_dir,
                windows=windows,
                board=board,
            )
            if statistics is not None:
                ret = statistics.to_dict()

        elif file in images:
            entry_type = "image"
            ret = process_image(file, CameraType(args.camera_type), debug_dir, board)
        elif file in ftk_recordings:
            entry_type = "ftk"
            ret = process_ftk_recording(file, debug_dir)

        if ret is not None:
            result[str(file)] = {"type": entry_type, **ret}
        else:
            errprint(f"Error: Unable to time-sync {file}.")

        # Save result to file after every video to avoid data loss
        with output_path.open("w") as f:
            json.dump(result, f, indent=4, cls=NpEncoder)
        print(f"Result written to {args.output}")


if __name__ == "__main__":
    main()
