"""Shared plumbing for the end-to-end tiers, which run the installed command-line tools.

What a user sees is judged in the *player view*: frames and timestamps as ffmpeg decodes
them, edit lists honoured. It is read through the ffmpeg and ffprobe binaries rather than
`rocsync.timeline` or `VideoReader`, which are part of what is under test.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rocsync.recording_statistics import (
    CLOCK_DRIFT_WARN_PPM,
    drift_ppm,
    rate_uncertainty_limit_ms,
)
from rocsync.timeline import frame_pts, measured_residual_threshold_ms

CACHE_DIR_VAR = "ROCSYNC_E2E_CACHE_DIR"
DOWNSCALE_HEIGHT = 540  # a quarter of FullHD's pixels, which detection handles as well

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe",
)


def cli(name, *args, timeout=None):
    """Runs one of the installed console scripts, so the entry point itself is under test."""
    executable = Path(sys.executable).parent / name
    assert executable.exists(), f"{name} is not installed next to {sys.executable}"
    return subprocess.run(
        [str(executable), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )


def output_of(result):
    """Everything a run printed, for asserting on messages and for failure reports."""
    return f"{result.stdout}\n{result.stderr}"


def assert_succeeded(result):
    assert result.returncode == 0, (
        f"{result.args[0]} exited with {result.returncode}:\n{output_of(result)}"
    )


def env_dir(var):
    """The directory an environment variable names, or a skip when there is none."""
    value = os.environ.get(var)
    if not value:
        pytest.skip(f"set {var} to run this tier")
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"{var}={value} is not a directory")
    return path


def plausibility_problems(entry):
    """Why a `rocsync` clock fit is doubtful, by the bounds the tools themselves warn at."""
    problems = []
    period = entry["median_frame_period"]
    ppm = drift_ppm(entry["clock_rate"], entry.get("source_tick_ms") or 1.0)
    if abs(ppm) > CLOCK_DRIFT_WARN_PPM:
        problems.append(f"drift {ppm:+.0f} ppm beyond {CLOCK_DRIFT_WARN_PPM}")
    uncertainty = entry.get("extrapolation_stderr_ms")
    if uncertainty is None:
        problems.append("clock-rate uncertainty not measured")
    elif uncertainty > rate_uncertainty_limit_ms(period):
        problems.append(
            f"rate uncertain by {uncertainty:.1f} ms at the ends, "
            f"beyond {rate_uncertainty_limit_ms(period):.1f}"
        )
    rmse = entry.get("rmse_after")
    if rmse is None or rmse > measured_residual_threshold_ms(period):
        problems.append(f"rmse_after {rmse} beyond {measured_residual_threshold_ms(period):.1f} ms")
    return problems


def hms(seconds):
    """Seconds as the h:mm:ss.fff a `--window` bound takes."""
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}:{minutes:02d}:{secs:06.3f}"


def _probe(path, *args):
    return json.loads(
        subprocess.check_output(["ffprobe", "-v", "error", "-of", "json", *args, str(path)])
    )


def video_size(path):
    """(width, height) of the first video stream."""
    stream = _probe(path, "-select_streams", "v:0", "-show_entries", "stream=width,height")[
        "streams"
    ][0]
    return int(stream["width"]), int(stream["height"])


def displayed_frames(path, read_intervals=None):
    """Player-view timestamp in ms of every displayed frame, relative to the stream start.

    `read_intervals` is ffprobe's own, e.g. "%+#3" for the first three frames, so a long
    file need not be decoded whole.
    """
    intervals = ["-read_intervals", read_intervals] if read_intervals else []
    data = _probe(
        path,
        "-select_streams",
        "v:0",
        *intervals,
        "-show_entries",
        "stream=start_time:frame=best_effort_timestamp_time",
    )
    start = float(data["streams"][0].get("start_time") or 0.0)
    return [
        (float(frame["best_effort_timestamp_time"]) - start) * 1000.0 for frame in data["frames"]
    ]


def decode_gray(path):
    """Every displayed frame as an (n, height, width) uint8 array."""
    width, height = video_size(path)
    command = [
        "ffmpeg", "-v", "error", "-i", str(path),
        # passthrough, or the rawvideo muxer duplicates frames onto a constant rate
        "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]  # fmt: skip
    raw = subprocess.check_output(command)
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width)


def cache_dir():
    """Where derived test inputs are kept between runs, outside any dataset."""
    return Path(os.environ.get(CACHE_DIR_VAR) or Path.home() / ".cache" / "rocsync" / "e2e")


def real_sample_copy(data_dir, path):
    """Where the downscaled copy of a real sample under `data_dir` is kept."""
    return cache_dir() / "real_samples" / Path(path).relative_to(data_dir).with_suffix(".mp4")


def _downscale_commands(source, output):
    """ffmpeg commands that downscale `source` to `output`, to try in turn: GPU, then CPU."""
    # Keep every frame, on the source's own time base and timestamps
    timing = ["-fps_mode", "passthrough", "-enc_time_base", "demux"]
    gpu = [
        "ffmpeg", "-v", "error", "-y", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", str(source), "-map", "0:v:0", "-an", "-vf", f"scale_cuda=-2:{DOWNSCALE_HEIGHT}",
        # nvenc's B-frames shift the second frame's timestamp
        *timing, "-c:v", "h264_nvenc", "-bf", "0", "-rc", "vbr", "-cq", "20", "-b:v", "0",
        str(output),
    ]  # fmt: skip
    cpu = [
        "ffmpeg", "-v", "error", "-y", "-i", str(source), "-map", "0:v:0", "-an",
        "-vf", f"scale=-2:{DOWNSCALE_HEIGHT}",
        *timing, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(output),
    ]  # fmt: skip
    return gpu, cpu


def downscaled(source, target):
    """`target`, a copy of `source` at DOWNSCALE_HEIGHT with every frame on its own pts.

    Made once and reused while the source is unchanged. Only the pixels change, so a clock
    fitted to the copy holds for the source; the copy is refused if any timestamp moved.
    """
    source, target = Path(source), Path(target)
    stat = source.stat()
    key = {
        "source": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "height": DOWNSCALE_HEIGHT,
    }
    stamp = target.with_suffix(".json")
    if target.is_file() and stamp.is_file() and json.loads(stamp.read_text()) == key:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.stem}.partial{target.suffix}")
    for command in _downscale_commands(source, partial):
        done = subprocess.run(command).returncode == 0
        if done and frame_pts(partial) == pytest.approx(frame_pts(source), abs=1e-6):
            break
        partial.unlink(missing_ok=True)
    else:
        raise RuntimeError(f"downscaling {source} failed or moved frame timestamps")
    partial.replace(target)
    stamp.write_text(json.dumps(key))
    return target
