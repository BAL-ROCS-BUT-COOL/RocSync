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
    annotation_camera,
    collect_frames,
    corner_positions_in_image,
    frame_key,
    parse_frame_key,
    retimed_videos,
    source_key,
)
from rocsync.board_profiles import PROFILES_BY_ARUCO
from rocsync.camera import CameraType
from rocsync.timeline import frame_pts, source_frame_period_ms, summarize_timeline
from rocsync.vision import process_frame


def _matrix(H):
    """A homography as a JSON-safe nested list, or None."""
    return np.asarray(H, dtype=np.float64).tolist() if H is not None else None


def extract_pipeline_result(stats, camera, board):
    """Extract a ground-truth-compatible result dict from pipeline stats.

    `camera` and `board` are the mode and profile the frame was run under — an
    explicit annotation, or RGB with auto-resolution when there is none — so a run
    can be scored against the layout it actually used rather than one re-derived
    from the ArUco marker, which IR never detects.

    Returns a dict with keys: aruco, corners, counter, ring, timestamp,
    homography, rough_homography. Structure mirrors ground_truth.json to
    simplify comparison.
    """
    steps = stats.get("steps", {})

    # -- ArUco --
    aruco_step = steps.get("aruco_detection", {})
    aruco_visible = aruco_step.get("success", False)
    aruco = {
        "visible": aruco_visible,
        "id": stats.get("aruco_id") if aruco_visible else None,
        "corners": stats.get("aruco_corners") if aruco_visible else None,
    }

    # -- Corners (original image coordinates, as annotated) --
    n_leds = len(board.always_on_leds[camera]) if board is not None else 4
    corners = [
        {"visible": pos is not None, "position": pos}
        for pos in corner_positions_in_image(stats, n_leds)
    ]

    # -- Counter --
    counter_step = steps.get("counter_reading", {})
    counter_value = counter_step.get("value")
    counter = {
        "visible": counter_value is not None,
        "value": counter_value,
    }

    # -- Ring --
    # Read off the arc rather than the timestamp, so an arc that wraps the period
    # end is still scored: the pipeline reads it correctly and then refuses to time
    # it, and those are two different things to be right or wrong about.
    ring_window = stats.get("ring_window")
    timestamp = stats.get("timestamp")

    if ring_window is not None and board is not None:
        # read_ring returns an inclusive end (last ON LED); convert to half-open
        # (first OFF LED) to match the ground truth convention.
        start, end = ring_window
        ring = {"start": start % board.period, "end": (end + 1) % board.period}
    else:
        ring = {"start": 0, "end": 0}

    return {
        "aruco": aruco,
        "corners": corners,
        "counter": counter,
        "ring": ring,
        "timestamp": timestamp,
        "homography": _matrix(stats.get("homography")),
        "rough_homography": _matrix(stats.get("rough_homography")),
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


def run_provenance(data_dir, n_images, try_hard):
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
        "try_hard": try_hard,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }


def run_benchmark(frames, ground_truth=None, debug_dir=None, try_hard=False):
    """Run pipeline on all frames, returning results dict keyed by frame key.

    `ground_truth`, when given, decides each frame's camera mode and board: IR has no
    marker to auto-resolve either from, so a frame with no annotation falls back to RGB
    with the board left for the pipeline to detect, exactly as an unannotated run always
    has. The validator walks retimed clips while annotations are keyed to the recordings
    they were cut from, so a retimed frame's key is resolved back to its source first.
    """
    ground_truth = ground_truth or {}
    images = ground_truth.get("images") or {}
    retimed = retimed_videos(ground_truth)

    results = {}
    with FrameSource() as source:
        for i, ref in enumerate(tqdm(frames)):
            image = source.read(ref)
            if image is None:
                print(f"  WARNING: could not read {ref.key}", file=sys.stderr)
                continue

            entry = images.get(source_key(ref.key, retimed))
            camera = annotation_camera(entry) if entry is not None else CameraType.RGB
            aruco_id = entry.get("aruco", {}).get("id") if entry is not None else None
            profile = PROFILES_BY_ARUCO.get(aruco_id)

            stats = {}
            success, timestamp = process_frame(
                image,
                camera,
                i,
                board=profile,
                debug_dir=debug_dir,
                try_hard=try_hard,
                stats=stats,
            )

            result_board = profile or PROFILES_BY_ARUCO.get(stats.get("aruco_id"))
            result = extract_pipeline_result(stats, camera, result_board)
            timing = extract_pipeline_timing(stats)

            results[ref.key] = {
                **result,
                "success": success and timestamp is not None,
                "timing": timing,
            }

    return results


def _container_fps(path):
    """Nominal frame rate the container reports."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def _fit_record(rel_path, timestamps, frame_times, fps, frame_period_ms, timeline_windowed=False):
    """Fit one video's clock, returning its summary plus a per-frame table keyed like `images`.

    A frame's pts is its own whether or not the fit succeeds, so `frames` is built first and
    the fit's own residual/inlier verdict folded in afterward. Kept on the video record rather
    than in `results`, because a retimed clip and the recording it was cut from can each carry
    a different fit over the same frame -- one field per frame per fit, not two fields on one.
    """
    frames = {
        frame_key(rel_path, index): {"pts_ms": pts_ms} for index, pts_ms in frame_times.items()
    }
    try:
        statistics, _, considered, rejected, _ = summarize_timeline(
            timestamps,
            frame_times,
            len(frame_times),
            fps,
            timeline_windowed=timeline_windowed,
            frame_period_ms=frame_period_ms,
        )
    except ValueError as e:
        print(f"  WARNING: no clock for {rel_path}: {e}", file=sys.stderr)
        return {
            "n_frames": len(frame_times),
            "n_timestamped_frames": len(timestamps),
            "error": str(e),
            "frames": frames,
        }

    for index, (*_, residual) in {**considered, **rejected}.items():
        frames[frame_key(rel_path, index)].update(
            residual_ms=float(residual), inlier=index in considered
        )

    record = statistics.to_dict()
    # The per-frame residuals now live in this record's own `frames` table
    del record["considered_timestamps"], record["rejected_timestamps"]
    return {**record, "n_timestamped_frames": len(timestamps), "error": None, "frames": frames}


def fit_videos(frames, results, data_dir, ground_truth=None):
    """Fit a clock per video, plus a derived one for each recording behind a retimed clip.

    Runs the production summarization over every frame that decoded, rather than the
    one-per-second sample a real run takes: the benchmark has already paid to decode
    them all, and using all of them keeps fit quality separate from sampling policy.

    A retimed clip is a stream copy, so its decoded frame j is the recording's frame
    j + offset, pixel for pixel. Re-fitting those same decodes against the recording's own
    timestamps reports the recording's clock too, without decoding it a second time.
    """
    by_path = defaultdict(list)
    for ref in frames:
        if ref.index is not None:
            by_path[ref.path].append(ref)

    videos = {}
    decoded = {}
    clip_frame_times = {}
    for path in sorted(by_path):
        refs = by_path[path]
        rel_path, _ = parse_frame_key(refs[0].key)
        frame_times = dict(enumerate(frame_pts(path)))
        timestamps = {
            ref.index: tuple(results[ref.key]["timestamp"])
            for ref in refs
            if results.get(ref.key, {}).get("timestamp") is not None
        }
        decoded[rel_path] = timestamps
        clip_frame_times[rel_path] = frame_times
        videos[rel_path] = _fit_record(
            rel_path, timestamps, frame_times, _container_fps(path), source_frame_period_ms(path)
        )

    for clip in retimed_videos(ground_truth or {}).values():
        if clip.path not in decoded or clip.source in videos:
            continue  # nothing decoded for the clip, or the recording was benchmarked itself
        source_path = data_dir / clip.source
        if not source_path.is_file():
            continue
        source_pts = dict(enumerate(frame_pts(source_path)))
        # Every clip frame counts, not just the ones that decoded a timestamp, so a
        # misdecode does not masquerade as a dropped frame in the recording's own gaps
        window = {
            index + clip.frame_offset: source_pts[index + clip.frame_offset]
            for index in clip_frame_times[clip.path]
            if index + clip.frame_offset in source_pts
        }
        timestamps = {index + clip.frame_offset: ts for index, ts in decoded[clip.path].items()}
        videos[clip.source] = {
            **_fit_record(
                clip.source,
                timestamps,
                window,
                _container_fps(source_path),
                source_frame_period_ms(source_path),
                timeline_windowed=True,
            ),
            "derived_from": clip.path,
        }

    return dict(sorted(videos.items()))


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
    parser.add_argument(
        "-g",
        "--ground-truth",
        default=None,
        help="Ground truth JSON, for each frame's camera mode and board "
        "(default: <data_dir>/ground_truth.json)",
    )
    parser.add_argument("--debug", default=None, help="Directory for debug images")
    parser.add_argument(
        "--try-hard",
        action="store_true",
        help="relaxed mode: refine the homography from partial corner LED detections "
        "and accept the board at any distance",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    frames = collect_frames(data_dir)
    if not frames:
        print(f"No images or videos found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    if args.debug:
        Path(args.debug).mkdir(parents=True, exist_ok=True)

    gt_path = Path(args.ground_truth) if args.ground_truth else data_dir / "ground_truth.json"
    ground_truth = None
    if gt_path.is_file():
        with open(gt_path) as f:
            ground_truth = json.load(f)
    elif args.ground_truth:
        print(f"No ground truth at {gt_path}", file=sys.stderr)
        sys.exit(1)

    n_stills = sum(1 for ref in frames if ref.index is None)
    n_videos = len({ref.path for ref in frames if ref.index is not None})
    print(
        f"Found {len(frames)} frames: {n_stills} images and "
        f"{len(frames) - n_stills} frames from {n_videos} videos"
    )
    results = run_benchmark(
        frames, ground_truth=ground_truth, debug_dir=args.debug, try_hard=args.try_hard
    )

    n_success = sum(1 for r in results.values() if r["success"])
    print(f"Detection rate: {n_success}/{len(results)} ({n_success / len(results):.1%})")

    videos = fit_videos(frames, results, data_dir, ground_truth)
    for rel_path, video in videos.items():
        derived = f" (derived from {video['derived_from']})" if "derived_from" in video else ""
        if video["error"] is not None:
            print(f"{rel_path}{derived}: no clock ({video['error']})")
            continue
        print(
            f"{rel_path}{derived}: {video['clock_rate']:.6f}x, "
            f"offset {video['clock_offset_ms']:.1f} ms, "
            f"RMSE {video['rmse_after']:.2f} ms, "
            f"{video['n_considered_frames']}/{video['n_timestamped_frames']} inliers"
        )

    output = {
        "config": run_provenance(data_dir, len(results), args.try_hard),
        "videos": videos,
        "images": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
