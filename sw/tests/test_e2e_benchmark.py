"""`rocsync` end to end on the hand-annotated benchmark videos.

Runs the command-line tool the way a user does -- its own frame sampling, board
auto-detection for RGB, `--try-hard` where the annotated board is held too far away for the
default distance check -- and scores each video's clock against the reference frozen in
`ground_truth.json`, by the measure `rocsync-evaluate` uses: the board-time error at the
first and last annotated frame, against the tolerance that reference was checked by.

Set ROCSYNC_BENCHMARK_DIR to the benchmark's data directory to run it.
"""

import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from rocsync.benchmark.common import (
    ReferenceClock,
    annotation_camera,
    ground_truth_path,
    load_ground_truth,
    parse_frame_key,
    residual_threshold_ms,
)
from rocsync.board_profiles import PROFILES_BY_ARUCO
from rocsync.camera import CameraType
from rocsync.recording_statistics import drift_ppm
from rocsync.vision import MIN_ARUCO_AREA_FRACTION
from tests.e2e import cli, env_dir, output_of, video_size

pytestmark = pytest.mark.e2e

DATA_DIR_VAR = "ROCSYNC_BENCHMARK_DIR"


def _reference_videos():
    """Relative paths of every video with a reference clock, read at collection time."""
    value = os.environ.get(DATA_DIR_VAR)
    path = ground_truth_path(value, None) if value else None
    if path is None or not path.is_file():
        return [pytest.param(None, marks=pytest.mark.skip(reason=f"set {DATA_DIR_VAR}"))]
    return sorted(load_ground_truth(path)["videos"])


def marker_area(entry):
    """Image area in px of the annotated ArUco marker, or None if it was not annotated."""
    profile = PROFILES_BY_ARUCO.get(entry.get("aruco", {}).get("id"))
    homography = entry.get("homography")  # image to rectified board
    if profile is None or homography is None or not entry["aruco"].get("visible"):
        return None
    board = np.array([profile.rectify().aruco_corners_coords], np.float64)
    x, y = cv2.perspectiveTransform(board, np.linalg.inv(homography))[0].T
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def run_arguments(ground_truth, data_dir, rel_path):
    """The arguments a user would pass for this video, read off its annotations."""
    reference = ground_truth["videos"][rel_path]
    annotated = reference.get("source", rel_path)  # retimed clips borrow their source's
    entries = [
        entry
        for key, entry in ground_truth["images"].items()
        if parse_frame_key(key)[0] == annotated
    ]
    cameras = Counter(annotation_camera(entry) for entry in entries)
    camera = cameras.most_common(1)[0][0] if cameras else CameraType.RGB
    if camera == CameraType.INFRARED:
        boards = Counter(entry.get("aruco", {}).get("id") for entry in entries)
        profile = PROFILES_BY_ARUCO[boards.most_common(1)[0][0]]
        return ("-c", camera.value, "--board-version", profile.name)

    # RGB identifies the board from its marker, but only reads one held close enough
    areas = [a for a in map(marker_area, entries) if a is not None]
    width, height = video_size(data_dir / rel_path)
    if areas and np.median(areas) < MIN_ARUCO_AREA_FRACTION * width * height:
        return ("--try-hard",)
    return ()


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    """(ground truth, {rel path: output.json entry}, [completed runs], [videos run])."""
    data_dir = env_dir(DATA_DIR_VAR).resolve()
    ground_truth = load_ground_truth(ground_truth_path(data_dir, None))
    output = tmp_path_factory.mktemp("benchmark") / "output.json"

    present = [p for p in sorted(ground_truth["videos"]) if (data_dir / p).is_file()]
    groups = {}
    for rel_path in present:
        groups.setdefault(run_arguments(ground_truth, data_dir, rel_path), []).append(rel_path)

    # One run per set of arguments, all resuming into the same output file
    runs = [
        cli("rocsync", *(data_dir / p for p in paths), "-y", "-o", output, *arguments)
        for arguments, paths in groups.items()
    ]
    written = json.loads(output.read_text()) if output.exists() else {}
    results = {
        Path(path).relative_to(data_dir).as_posix(): entry for path, entry in written.items()
    }
    return ground_truth, results, runs, present


def test_rocsync_exit_code_reports_unsynced_videos(benchmark):
    _, results, runs, present = benchmark
    missing = [p for p in present if p not in results]
    failed = [run for run in runs if run.returncode != 0]

    assert bool(failed) == bool(missing), "\n".join(output_of(run)[-2000:] for run in runs)


@pytest.mark.parametrize("rel_path", _reference_videos())
def test_rocsync_recovers_the_reference_clock(benchmark, rel_path):
    ground_truth, results, _, present = benchmark
    if rel_path not in present:
        pytest.skip(f"{rel_path} is in the ground truth but not on disk")
    reference = ground_truth["videos"][rel_path]
    entry = results.get(rel_path)
    assert entry is not None, "rocsync could not time-sync this video"

    rate, offset = entry["clock_rate"], entry["clock_offset_ms"]
    first, last = ReferenceClock.from_dict(reference).span_errors(rate, offset)
    threshold = residual_threshold_ms(reference)
    assert max(abs(first), abs(last)) <= threshold, (
        f"board time off by {first:+.2f} ms at the first annotated frame and {last:+.2f} ms "
        f"at the last (tolerance {threshold:.2f} ms); rate error "
        f"{drift_ppm(rate) - drift_ppm(reference['clock_rate']):+.0f} ppm, "
        f"rmse_after {entry.get('rmse_after')}"
    )
