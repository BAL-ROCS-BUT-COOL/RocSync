import argparse
import json
import os
import shlex
import subprocess
import sys

import cv2
import numpy as np

from rocsync.printer import errprint, succprint, warnprint
from rocsync.recording_statistics import (
    CLOCK_DRIFT_BAD_PPM,
    CLOCK_DRIFT_WARN_PPM,
    drift_ppm,
    rate_uncertainty_limit_ms,
)
from rocsync.timeline import (
    affine_from_statistics,
    frame_pts,
    median_frame_period,
    parse_ratio,
    run_ffprobe,
)

CUT_LEAD_S = 1e-4  # a cut this far ahead of a frame's pts keeps that frame


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


def seek_base_s(video_path) -> float:
    """Seconds from where ffmpeg's -ss counts, the file's start, to the video stream's start."""

    def start_time(section):
        output = run_ffprobe(video_path, "-show_entries", f"{section}=start_time", "-of", "csv=p=0")
        return parse_ratio((output or "").strip().rstrip(","))

    stream_start, file_start = start_time("stream"), start_time("format")
    if stream_start is None or file_start is None:
        return 0.0
    return stream_start - file_start


def warn_about_clock(file, statistics, compensate_drift):
    """Warn about a clock fit that may misplace this video; aligning proceeds regardless."""
    if "extrapolation_stderr_ms" not in statistics:
        warnprint(f"{file}: sync entry has no clock-rate uncertainty; re-run rocsync to assess it.")
    elif statistics["extrapolation_stderr_ms"] is None:
        warnprint(f"{file}: clock-rate uncertainty could not be measured (too few inliers).")
    else:
        uncertainty_ms = statistics["extrapolation_stderr_ms"]
        limit_ms = rate_uncertainty_limit_ms(statistics["median_frame_period"])
        if uncertainty_ms > limit_ms:
            warnprint(
                f"{file}: clock rate uncertain by ±{uncertainty_ms:.1f} ms at the first/last "
                f"frame (limit ±{limit_ms:.1f} ms, half a frame); frames far from the "
                "RocSync windows may be misplaced."
            )

    clock_rate, _ = affine_from_statistics(statistics)
    ppm = drift_ppm(clock_rate)
    if abs(ppm) > CLOCK_DRIFT_BAD_PPM:
        rescale = " Drift compensation will rescale it substantially." if compensate_drift else ""
        warnprint(
            f"{file}: video clock runs {ppm:+.0f} ppm off board time; that is not drift, "
            f"something is wrong with the fit.{rescale}"
        )
    elif abs(ppm) > CLOCK_DRIFT_WARN_PPM:
        warnprint(f"{file}: video clock runs {ppm:+.0f} ppm off board time.")


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
        help="Output directory for the synchronized videos; a relative path is taken inside "
        "each video's own folder, an absolute one is used as is (default: synced)",
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
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be a positive number")

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
            "edit list, which some players ignore. Use --compensate-drift to re-encode instead."
        )

    # first_frame/last_frame only cover the analyzed frames, so measure the files
    timings = {}
    for file, statistics in videos.items():
        try:
            pts = np.asarray(frame_pts(file))
            clock_rate, clock_offset_ms = affine_from_statistics(statistics)
        except (OSError, KeyError) as e:
            errprint(f"Cannot determine the board-time span of {file}: {e}")
            return 1
        if not pts.size:
            errprint(f"No frames found in {file}; cannot align it.")
            return 1
        timings[file] = (pts, clock_rate * pts + clock_offset_ms)
        warn_about_clock(file, statistics, args.compensate_drift)

    # Window covered by every video, in board time
    origin_ms = max(board[0] for _, board in timings.values())
    end_ms = min(board[-1] for _, board in timings.values())
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

        pts, board = timings[file]
        clock_rate, clock_offset_ms = affine_from_statistics(statistics)
        base_s = seek_base_s(file)
        if args.compensate_drift:
            ffmpeg_command = compensate_command(
                file,
                output_file,
                cut_s=base_s + (origin_ms - clock_offset_ms) / clock_rate / 1000,
                span_ms=end_ms - origin_ms,
                clock_rate=clock_rate,
                board_period_ms=clock_rate * (median_frame_period(pts) or 0.0),
                frame_rate=nominal_fps,
                use_nvenc=use_nvenc,
            )
        else:
            # Each video starts and ends on its frames nearest the common span's ends
            first = int(np.argmin(np.abs(board - origin_ms)))
            last = int(np.argmin(np.abs(board - end_ms)))
            ffmpeg_command = stream_copy_command(
                file,
                output_file,
                start_s=base_s + pts[first] / 1000,
                duration_s=(pts[last] - pts[first]) / 1000,
            )

        # ffmpeg runs in the background, so throttle before starting another one
        while args.jobs and len(running) >= args.jobs:
            reap(running, failed, block=True)

        # No shell: the arguments go to ffmpeg verbatim, so paths containing spaces
        # survive. shlex.join only builds the human-readable echo of the command.
        print(shlex.join(ffmpeg_command))
        running.append((file, subprocess.Popen(ffmpeg_command)))
        started += 1

    while running:
        reap(running, failed, block=True)

    if failed:
        errprint(f"{len(failed)} of {started} encoded videos failed.")
        return 1

    succprint(f"Aligned {started} videos into {args.output_dir}")


def stream_copy_command(video_path, output_file, start_s, duration_s):
    """ffmpeg command keeping the frames from `start_s` to `start_s + duration_s` (both
    ffmpeg input seconds, ends included) bit for bit."""
    return [
        "ffmpeg",
        "-ss",
        f"{start_s - CUT_LEAD_S:.6f}",
        "-i",
        video_path,
        "-c:v",
        "copy",
        "-t",
        f"{duration_s + 2 * CUT_LEAD_S:.6f}",
        "-y",
        output_file,
    ]


def compensate_command(
    video_path,
    output_file,
    cut_s,
    span_ms,
    clock_rate,
    board_period_ms,
    frame_rate,
    use_nvenc,
):
    """ffmpeg command re-encoding the `span_ms` of board time from `cut_s` (ffmpeg input
    seconds) at `frame_rate`, each output frame the source frame nearest its board time."""
    # Decode from a frame early, so the frame nearest the start is there to pick
    seek_s = max(0.0, cut_s - board_period_ms / clock_rate / 1000)
    lead_s = cut_s - seek_s
    # Board time since the start, less half a source frame: rounding up then picks the nearest
    timing = (
        f"setpts=(PTS-{lead_s:.6f}/TB)*{clock_rate}-{board_period_ms / 2000:.6f}/TB,"
        f"fps={frame_rate}:start_time=0:round=up"
    )
    return [
        "ffmpeg",
        "-ss",
        f"{seek_s:.6f}",
        "-i",
        video_path,
        "-c:v",
        "hevc_nvenc" if use_nvenc else "libx265",
        "-crf",
        "0",
        "-filter_complex",
        timing,
        # Every output frame within the span, counted rather than timed with -t
        "-frames:v",
        str(int(span_ms * frame_rate / 1000) + 1),
        "-y",
        output_file,
    ]


if __name__ == "__main__":
    sys.exit(main())
