import argparse
import json
import os
import shlex
import subprocess

import cv2

from rocsync.printer import errprint, succprint, warnprint
from rocsync.timeline import affine_from_statistics, per_frame_times


def hevc_nvenc_available() -> bool:
    """Whether this ffmpeg can encode with hevc_nvenc, which is far faster than libx265."""
    try:
        encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return "hevc_nvenc" in encoders


def reap(running: list[tuple[str, subprocess.Popen]], failed: list[str], block: bool) -> None:
    """Drop finished encodes from `running`, recording non-zero exits in `failed`.

    With `block`, waits for the oldest encode first instead of spinning.
    """
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

    with open(args.sync_file) as file:
        stats = json.load(file)

    # Filter recordings, keep only videos (image and ftk cannot be processed).
    videos = {path: data for path, data in stats.items() if data and data.get("type") == "video"}
    if not videos:
        errprint(
            f"No video entries found in {args.sync_file}; nothing to align. "
            "Run rocsync on the video files first."
        )
        return 1

    skipped = len(stats) - len(videos)
    if skipped:
        print(f"Ignoring {skipped} non-video entries in {args.sync_file}")

    nominal_fps = (
        round(next(iter(videos.values()))["nominal_fps"]) if args.fps is None else args.fps
    )
    print(f"Syncing {len(videos)} videos to {nominal_fps} FPS")

    use_nvenc = args.compensate_drift and hevc_nvenc_available()
    if args.compensate_drift and not use_nvenc:
        warnprint(
            "hevc_nvenc not available, encoding will be very slow. Install NVIDIA drivers and ffmpeg with nvenc support or disable drift compensation."
        )
    if not args.compensate_drift:
        warnprint(
            "Stream copy can only cut at a keyframe; the rest is left to a container "
            "edit list. Use --compensate-drift for a frame-exact start."
        )

    # first_frame/last_frame only cover the analyzed frames, so measure the files
    spans = {}
    for file, statistics in videos.items():
        try:
            board_times = per_frame_times(file, statistics)
        except (OSError, KeyError) as e:
            errprint(f"Cannot determine the board-time span of {file}: {e}")
            return 1
        if not board_times:
            errprint(f"No frames found in {file}; cannot align it.")
            return 1
        spans[file] = (board_times[0], board_times[-1])

    # Window covered by every video, in board time
    origin_ms = max(start for start, _ in spans.values())
    end_ms = min(end for _, end in spans.values())
    if origin_ms >= end_ms:
        errprint(
            f"The videos have no common time span: the latest start "
            f"({origin_ms:.1f} ms board time) is at or after the earliest end "
            f"({end_ms:.1f} ms). They do not overlap and cannot be aligned."
        )
        return 1
    print(
        f"Aligning to board time {origin_ms:.1f} ms, keeping "
        f"{(end_ms - origin_ms) / 1000:.3f} s up to {end_ms:.1f} ms"
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
            # A readable output file means this video was already synced
            vid = cv2.VideoCapture(output_file)
            already_synced = vid.isOpened()
            vid.release()
            if already_synced:
                print(f"Skipping {file}, already synced.")
                continue

        # -ss and -t are container time, so map the window through this video's fit
        clock_rate, clock_offset_ms = affine_from_statistics(statistics)
        cut_time = (origin_ms - clock_offset_ms) / clock_rate / 1000
        duration = (end_ms - origin_ms) / clock_rate / 1000

        # ffmpeg runs in the background, so throttle before starting another one
        while args.jobs and len(running) >= args.jobs:
            reap(running, failed, block=True)

        running.append(
            (
                file,
                sync_video(
                    file,
                    cut_time,
                    duration,
                    clock_rate,
                    output_file=output_file,
                    frame_rate=nominal_fps,
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
    cut_time: float,
    duration: float,
    clock_rate: float,
    output_file: str = "synced.mp4",
    frame_rate: int = 30,
    compensate_drift: bool = True,
    use_nvenc: bool = False,
) -> subprocess.Popen:
    """Cut `duration` seconds starting `cut_time` seconds into the video, both in
    container time, rescaling by `clock_rate` if drift is compensated."""
    if abs(clock_rate - 1) > 0.05:
        warnprint(
            f"Video clock runs at {clock_rate:.4f}x board time; "
            f"drift compensation will rescale it substantially."
        )

    ffmpeg_command = [
        "ffmpeg",
        "-ss",
        f"{cut_time:.6f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.6f}",
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
