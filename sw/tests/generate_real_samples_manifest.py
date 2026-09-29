"""Writes `e2e_real_samples.json`, the manifest `test_e2e_real.py` runs from.

Every recording is first downscaled into the e2e cache (see `tests.e2e.downscaled`), which
is what the test runs on too. Its first and last minutes, and any `--search` span, are then
searched for the board, and a searched span that ends on a sighting is extended until the
board is out of view. The sightings, padded so frames the pipeline currently misses are
still searched, become the windows the test passes to `rocsync`. With `--try-hard`, the
recordings are searched and fitted with rocsync's `--try-hard`, and the test runs them so.

A windowed `rocsync` pass over those windows decides how the recording is judged: by the
round trip through `rocsync-align` alone when its clock fit is plausible, or also against
the clock frozen here when it is not. Review the table this prints, flip a recording's
`check` by hand if the verdict is wrong, and commit the manifest.

    uv run python -m tests.generate_real_samples_manifest /path/to/real_samples
    uv run python -m tests.generate_real_samples_manifest /path/to/real_samples \
        --only session/camera.mp4 --search 0:37:00 0:40:00 --try-hard
"""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from rocsync.camera import CameraType
from rocsync.dataset import VIDEO_SUFFIXES
from rocsync.main import parse_time
from rocsync.timecode import resolve_windows
from rocsync.timeline import frame_pts
from rocsync.video import process_video, process_video_window
from tests.e2e import (
    DOWNSCALE_HEIGHT,
    cache_dir,
    downscaled,
    plausibility_problems,
    real_sample_copy,
)

MANIFEST = Path(__file__).with_name("e2e_real_samples.json")
DATA_DIR_VAR = "ROCSYNC_REAL_SAMPLES_DIR"
HEAD_S = 300.0  # searched from the start of a recording
TAIL_S = 600.0  # searched back from its end
EDGE_S = 15.0  # a sighting this close to a searched span's edge extends the span
EXTEND_S = 120.0  # by this much
PADDING_S = 15.0  # around each span the board was seen in
GAP_S = 60.0  # a longer pause between sightings starts a new window
DOWNSCALE_JOBS = 3


def merge(spans):
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def board_windows(times_s, duration_s, gap_s=GAP_S, padding_s=PADDING_S):
    """[[start, end]] in seconds covering every sighting, padded and merged."""
    clusters = []
    for t in sorted(times_s):
        if clusters and t - clusters[-1][1] <= gap_s:
            clusters[-1][1] = t
        else:
            clusters.append([t, t])
    padded = [(max(0.0, s - padding_s), min(duration_s, e + padding_s)) for s, e in clusters]
    return [[round(s, 3), round(e, 3)] for s, e in merge(padded)]


def sightings(path, start_s, end_s, try_hard):
    """{frame: pts in s} of every frame between `start_s` and `end_s` the board decoded in."""
    timestamps, frame_times = process_video_window(
        str(path), CameraType.RGB, start_s, end_s, try_hard=try_hard
    )
    return {n: frame_times[n] / 1000 for n in timestamps}


def search(path, duration_s, extra_spans, try_hard):
    """(sightings in s, searched spans): head, tail and extra spans, grown while the board is in view."""
    extra = resolve_windows(extra_spans, lambda: duration_s) if extra_spans else []
    spans = merge(
        [
            (0.0, min(HEAD_S, duration_s)),
            (max(0.0, duration_s - TAIL_S), duration_s),
            *((max(0.0, s), min(e, duration_s)) for s, e in extra),
        ]
    )
    seen = {}
    for start, end in spans:
        seen.update(sightings(path, start, end, try_hard))

    grown = True
    while grown:
        grown = False
        for span in spans:
            start, end = span
            if start > 0 and any(start <= t < start + EDGE_S for t in seen.values()):
                span[0] = max(0.0, start - EXTEND_S)
                seen.update(sightings(path, span[0], start, try_hard))
                grown = True
            if end < duration_s and any(end - EDGE_S < t <= end for t in seen.values()):
                span[1] = min(duration_s, end + EXTEND_S)
                seen.update(sightings(path, end, span[1], try_hard))
                grown = True
        spans = merge(spans)
    return sorted(seen.values()), [[round(s, 3), round(e, 3)] for s, e in spans]


def describe(path, extra_spans=(), try_hard=False):
    """The manifest entry for one (downscaled) recording."""
    duration_s = frame_pts(path)[-1] / 1000
    times, searched = search(path, duration_s, list(extra_spans), try_hard)
    windows = board_windows(times, duration_s)
    described = {"searched": searched, "windows": windows, "n_sightings": len(times)}
    if try_hard:
        described["arguments"] = ["--try-hard"]
    if not windows:
        return {**described, "check": "excluded", "reason": "the board was never decoded"}

    statistics = process_video(str(path), CameraType.RGB, windows=windows, try_hard=try_hard)
    if statistics is None:
        return {**described, "check": "excluded", "reason": "rocsync cannot time-sync it"}
    entry = statistics.to_dict()
    problems = plausibility_problems(entry)
    if len(windows) < 2:
        problems.append("board seen in one window only, so the fit cannot be checked apart")

    described.update(check="frozen" if problems else "round_trip", problems=problems)
    if problems:
        # A frozen clock is only trusted to the larger of half a frame and its own spread
        described["frozen_clock"] = {
            "clock_rate": entry["clock_rate"],
            "clock_offset_ms": entry["clock_offset_ms"],
            "tolerance_ms": max(
                statistics.median_frame_period / 2, entry["extrapolation_stderr_ms"] or 0.0
            ),
        }
    return described


def checkout():
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def main():
    parser = argparse.ArgumentParser(description="Writes the manifest test_e2e_real.py runs from.")
    parser.add_argument("data_dir", nargs="?", default=os.environ.get(DATA_DIR_VAR))
    parser.add_argument(
        "--only", action="append", default=[], help="session or session/file to regenerate"
    )
    parser.add_argument(
        "--search",
        nargs=2,
        action="append",
        default=[],
        metavar=("START", "END"),
        help="also search this span (hh:mm:ss, as rocsync's --window) of every selected recording",
    )
    parser.add_argument(
        "--try-hard", action="store_true", help="search and fit with rocsync's --try-hard"
    )
    parser.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = parser.parse_args()
    if not args.data_dir:
        parser.error(f"give the data directory or set {DATA_DIR_VAR}")
    try:
        extra_spans = [(parse_time(start), parse_time(end)) for start, end in args.search]
    except ValueError as e:
        parser.error(f"argument --search: {e}")
    data_dir = Path(args.data_dir)

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"sessions": {}}
    videos = [
        p
        for p in sorted(data_dir.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in VIDEO_SUFFIXES
        and (
            not args.only
            or p.parent.relative_to(data_dir).as_posix() in args.only
            or p.relative_to(data_dir).as_posix() in args.only
        )
    ]

    print(f"Downscaling {len(videos)} recordings into {cache_dir()}")
    with ThreadPoolExecutor(DOWNSCALE_JOBS) as pool:
        copies = list(pool.map(lambda p: downscaled(p, real_sample_copy(data_dir, p)), videos))

    for path, copy in zip(videos, copies, strict=True):
        session = path.parent.relative_to(data_dir).as_posix()
        print(f"Searching {session}/{path.name}")
        manifest["sessions"].setdefault(session, {})[path.name] = describe(
            copy, extra_spans, args.try_hard
        )

    manifest["generated"] = {
        **checkout(),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "downscale_height": DOWNSCALE_HEIGHT,
        "head_s": HEAD_S,
        "tail_s": TAIL_S,
        "edge_s": EDGE_S,
        "extend_s": EXTEND_S,
        "padding_s": PADDING_S,
        "gap_s": GAP_S,
    }
    manifest["sessions"] = dict(sorted(manifest["sessions"].items()))

    print(f"\n{'recording':<40} {'check':<11} {'windows [s]':<40} problems")
    for session, recordings in manifest["sessions"].items():
        for name, entry in sorted(recordings.items()):
            windows = ", ".join(f"{s:.0f}-{e:.0f}" for s, e in entry.get("windows", []))
            problems = "; ".join(entry.get("problems", [])) or entry.get("reason", "")
            print(f"{session + '/' + name:<40} {entry['check']:<11} {windows:<40} {problems}")

    if not args.dry_run:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nWrote {MANIFEST}")


if __name__ == "__main__":
    main()
