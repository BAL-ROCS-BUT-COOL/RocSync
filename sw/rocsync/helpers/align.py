import argparse
import json
import os
import shlex
import subprocess

import cv2
from rocsync.printer import errprint, succprint, warnprint


def hevc_nvenc_available() -> bool:
    """Whether this ffmpeg can encode with hevc_nvenc, which is far faster than
    libx265. Probed once up front rather than per encode: it shells out, and the
    answer cannot change while we run."""
    try:
        encoders = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"], text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return "hevc_nvenc" in encoders


def reap(
    running: list[tuple[str, subprocess.Popen]], failed: list[str], block: bool
) -> None:
    """Remove finished encodes from `running`, appending their input path to
    `failed` if ffmpeg exited non-zero. With `block`, waits for the oldest
    encode to finish first, so the caller makes progress instead of spinning."""
    if block and running:
        running[0][1].wait()
    for entry in list(running):
        path, process = entry
        returncode = process.poll()
        if returncode is None:
            continue
        running.remove(entry)
        if returncode != 0:
            failed.append(path)
            errprint(f"ffmpeg exited with {returncode} for {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create aligned video files based on previously computed synchronization metadata."
    )
    parser.add_argument(
        "sync_file",
        type=str,
        metavar="sync.json",
        help="Path to JSON file containing video synchronization metadata with timestamps and frame offsets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="synced",
        help="Output directory where synchronized videos will be saved (default: synced)",
    )
    parser.add_argument(
        "--compensate-drift",
        action="store_true",
        help="Enable video drift compensation via re-encoding (significantly slower but more accurate)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Target frame rate for synchronized videos. If not specified, uses the source frame rate of the first video",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Maximum number of ffmpeg processes to run concurrently, or 0 for no limit (default: 4)",
    )

    args = parser.parse_args()

    if args.jobs < 0:
        parser.error("--jobs must be 0 (unlimited) or a positive number")

    with open(args.sync_file, "r") as file:
        stats = json.load(file)

    # Filter recordings, keep only videos (image and ftk cannot be processed).
    videos = {
        path: data
        for path, data in stats.items()
        if data and data.get("type") == "video"
    }
    if not videos:
        errprint(
            f"No video entries found in {args.sync_file}; nothing to align. "
            "Run rocsync on the video files first."
        )
        return 1

    skipped = len(stats) - len(videos)
    if skipped:
        print(f"Ignoring {skipped} non-video entries in {args.sync_file}")

    expected_fps = (
        int(round(next(iter(videos.values()))["expected_fps"]))
        if args.fps is None
        else args.fps
    )
    print(f"Syncing {len(videos)} videos to {expected_fps} FPS")

    use_nvenc = args.compensate_drift and hevc_nvenc_available()
    if args.compensate_drift and not use_nvenc:
        warnprint(
            "hevc_nvenc not available, encoding will be very slow. Install NVIDIA drivers and ffmpeg with nvenc support or disable drift compensation."
        )

    running: list[tuple[str, subprocess.Popen]] = []
    failed: list[str] = []
    started = 0
    for file, statistics in videos.items():
        # Check if the output file already exists
        video_name, _ = os.path.splitext(os.path.basename(file))
        video_folder = os.path.dirname(file)
        print(f"video_folder: {video_folder}, video_name: {video_name}")
        output_folder = os.path.join(video_folder, args.output_dir)
        output_file = os.path.join(output_folder, f"{video_name}.mp4")
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output file will be saved to {output_file}. Input file: {file}")

        if os.path.exists(output_file):
            try:
                vid = cv2.VideoCapture(output_file)
                if not vid.isOpened():
                    raise ValueError("Could not open video file")
            except Exception as e:
                pass
            else:
                print(f"Skipping {file}, already synced.")
                continue

        # ffmpeg runs in the background, so throttle before starting another one
        while args.jobs and len(running) >= args.jobs:
            reap(running, failed, block=True)

        running.append(
            (
                file,
                sync_video(
                    file,
                    statistics,
                    output_file=output_file,
                    frame_rate=expected_fps,
                    compensate_drift=args.compensate_drift,
                    use_nvenc=use_nvenc,
                ),
            )
        )
        started += 1

    while running:
        reap(running, failed, block=True)

    if failed:
        errprint(f"{len(failed)} of {started} encoded videos failed.")
        return 1

    succprint(f"Aligned {started} videos into {args.output_dir}")


def sync_video(
    video_path: str,
    stats: dict,
    offset: float = 0,
    output_file: str = "synced.mp4",
    frame_rate: int = 30,
    compensate_drift: bool = True,
    use_nvenc: bool = False,
) -> subprocess.Popen:
    cut_time = stats["first_frame"] * (-1 / 1000) + offset  # in seconds

    # Board ms per container ms, as fitted against the frames' own presentation timestamps. Warn if not close to 1.0
    clock_rate = stats["clock_rate"]
    if abs(clock_rate - 1) > 0.05:
        warnprint(
            f"Video clock runs at {clock_rate:.4f}x board time; "
            f"drift compensation will rescale it substantially."
        )

    ffmpeg_command = [
        "ffmpeg",
        "-ss",
        str(cut_time),
        "-i",
        video_path,
    ]

    if compensate_drift:
        ffmpeg_command += [
            "-c:v",
            "hevc_nvenc" if use_nvenc else "libx265",
            "-crf",
            "0",
            "-filter_complex",
            f"setpts=PTS*{clock_rate}",
            "-r",
            str(frame_rate),
        ]
    else:
        ffmpeg_command += [
            "-c:v",
            "copy",
        ]
    ffmpeg_command += [
        "-y",
        output_file,
    ]

    # No shell: the arguments go to ffmpeg verbatim, so paths containing spaces
    # survive. shlex.join only builds the human-readable echo of the command.
    print(shlex.join(ffmpeg_command))

    return subprocess.Popen(ffmpeg_command)


if __name__ == "__main__":
    main()
