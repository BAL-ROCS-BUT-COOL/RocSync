#!/usr/bin/env python3
"""Benchmark rocsync pipeline on validation data.

Runs process_frame on all images, collects per-image results in a structure
mirroring the ground truth format (aruco, corners, counter, ring), plus
per-step timing. Positions are in original image coordinates, matching the
annotations. Saves results as JSON without aggregate statistics — those are
computed by the evaluate tool.
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from rocsync.benchmark.common import (
    STEP_ORDER,
    FrameSource,
    collect_frames,
    corner_positions_in_image,
    frame_key,
    parse_frame_key,
)
from rocsync.board_profiles import PROFILES_BY_ARUCO
from rocsync.camera import CameraType
from rocsync.timeline import frame_pts, summarize_timeline
from rocsync.vision import process_frame


def extract_pipeline_result(stats):
    """Extract a ground-truth-compatible result dict from pipeline stats.

    Returns a dict with keys: aruco, corners, counter, ring, timestamp.
    Structure mirrors ground_truth.json to simplify comparison.
    """
    steps = stats.get("steps", {})

    # -- ArUco --
    aruco_id = stats.get("aruco_id", None)
    aruco_step = steps.get("aruco_detection", {})
    aruco_visible = aruco_step.get("success", False)
    aruco = {
        "visible": aruco_visible,
        "id": stats.get("aruco_id") if aruco_visible else None,
        "corners": stats.get("aruco_corners") if aruco_visible else None,
    }
    board = PROFILES_BY_ARUCO[aruco_id] if aruco_id is not None else None

    # -- Corners (original image coordinates, as annotated) --
    corners = [
        {"visible": pos is not None, "position": pos} for pos in corner_positions_in_image(stats)
    ]

    # -- Counter --
    counter_step = steps.get("counter_reading", {})
    counter_value = counter_step.get("value")
    counter = {
        "visible": counter_value is not None,
        "value": counter_value,
    }

    # -- Ring --
    ring_step = steps.get("ring_reading", {})
    ring_visible = ring_step.get("success", False)
    timestamp = stats.get("timestamp")

    if ring_visible and timestamp is not None:
        # read_ring returns inclusive end (last ON LED); convert to half-open
        # (first OFF LED) to match ground truth convention.
        ring = {
            "start": timestamp[0] % board.period if board is not None else timestamp[0],
            "end": (timestamp[1] + 1) % board.period if board is not None else timestamp[1] + 1,
        }
    else:
        ring = {"start": 0, "end": 0}

    return {
        "aruco": aruco,
        "corners": corners,
        "counter": counter,
        "ring": ring,
        "timestamp": timestamp,
    }


def extract_pipeline_timing(stats):
    """Extract per-step timing from pipeline stats."""
    steps = stats.get("steps", {})
    timing = {}
    for step in STEP_ORDER:
        if step in steps:
            timing[f"{step}_ms"] = steps[step]["time_ms"]
    timing["total_ms"] = stats.get("total_time_ms")
    return timing


def run_provenance(data_dir, n_images):
    """Identify the checkout that produced a result file, so columns are self-describing."""

    def git(*args):
        try:
            return subprocess.run(
                ["git", *args],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "data_dir": str(data_dir),
        "n_images": n_images,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }


def run_benchmark(frames, debug_dir=None):
    """Run pipeline on all frames, returning results dict keyed by frame key."""
    results = {}
    with FrameSource() as source:
        for i, ref in enumerate(tqdm(frames)):
            image = source.read(ref)
            if image is None:
                print(f"  WARNING: could not read {ref.key}", file=sys.stderr)
                continue

            stats = {}
            success, timestamp = process_frame(
                image,
                CameraType.RGB,
                i,
                debug_dir=debug_dir,
                stats=stats,
            )

            result = extract_pipeline_result(stats)
            timing = extract_pipeline_timing(stats)

            results[ref.key] = {
                **result,
                "success": success and timestamp is not None,
                "timing": timing,
            }

    return results


def fit_videos(frames, results):
    """Fit a clock per video and fold the per-frame outcome back into `results`.

    Runs the production summarization over every frame that decoded, rather than the
    one-per-second sample a real run takes: the benchmark has already paid to decode
    them all, and using all of them keeps fit quality separate from sampling policy.
    """
    by_path = defaultdict(list)
    for ref in frames:
        if ref.index is not None:
            by_path[ref.path].append(ref)

    videos = {}
    for path in sorted(by_path):
        refs = by_path[path]
        rel_path, _ = parse_frame_key(refs[0].key)
        frame_times = dict(enumerate(frame_pts(path)))

        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # A frame's presentation timestamp is its own, whether or not the fit succeeds
        for ref in refs:
            if ref.key in results and ref.index in frame_times:
                results[ref.key]["pts_ms"] = frame_times[ref.index]

        timestamps = {
            ref.index: tuple(results[ref.key]["timestamp"])
            for ref in refs
            if results.get(ref.key, {}).get("timestamp") is not None
        }

        try:
            statistics, _, considered, rejected, _ = summarize_timeline(
                timestamps, frame_times, len(frame_times), fps
            )
        except ValueError as e:
            videos[rel_path] = {
                "n_frames": len(frame_times),
                "n_timestamped_frames": len(timestamps),
                "error": str(e),
            }
            print(f"  WARNING: no clock for {rel_path}: {e}", file=sys.stderr)
            continue

        for index, (*_, residual) in {**considered, **rejected}.items():
            key = frame_key(rel_path, index)
            if key in results:
                results[key]["fit"] = {
                    "residual_ms": float(residual),
                    "inlier": index in considered,
                }

        record = statistics.to_dict()
        # The per-frame residuals now live next to the frames they belong to
        del record["considered_timestamps"], record["rejected_timestamps"]
        videos[rel_path] = {**record, "n_timestamped_frames": len(timestamps), "error": None}

    return videos


def main():
    parser = argparse.ArgumentParser(description="Benchmark rocsync on validation data")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="validation_data",
        help="Path to validation data directory (default: validation_data)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="benchmark_results.json",
        help="Output JSON file (default: benchmark_results.json)",
    )
    parser.add_argument("--debug", default=None, help="Directory for debug images")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    frames = collect_frames(data_dir)
    if not frames:
        print(f"No images or videos found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    if args.debug:
        Path(args.debug).mkdir(parents=True, exist_ok=True)

    n_stills = sum(1 for ref in frames if ref.index is None)
    n_videos = len({ref.path for ref in frames if ref.index is not None})
    print(
        f"Found {len(frames)} frames: {n_stills} images and "
        f"{len(frames) - n_stills} frames from {n_videos} videos"
    )
    results = run_benchmark(frames, debug_dir=args.debug)

    n_success = sum(1 for r in results.values() if r["success"])
    print(f"Detection rate: {n_success}/{len(results)} ({n_success / len(results):.1%})")

    videos = fit_videos(frames, results)
    for rel_path, video in videos.items():
        if video["error"] is not None:
            print(f"{rel_path}: no clock ({video['error']})")
            continue
        print(
            f"{rel_path}: {video['clock_rate']:.6f}x, "
            f"offset {video['clock_offset_ms']:.1f} ms, "
            f"RMSE {video['rmse_after']:.2f} ms, "
            f"{video['n_considered_frames']}/{video['n_timestamped_frames']} inliers"
        )

    output = {
        "config": run_provenance(data_dir, len(results)),
        "videos": videos,
        "images": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
