"""Check a ground truth file against itself.

Each annotation carries corner LED positions in image space and a homography, and the
evaluation relies on the two describing the same board pose: warping the modelled LED
layout through the homography must land on the annotated positions. Both ways of
producing an annotation make that exact rather than approximate — a four-point
``getPerspectiveTransform`` fit has no residual, and the pipeline's fine transform is
built from the same corners it reports — so this pins an invariant rather than measuring
agreement. It earns its keep by catching a file whose positions were edited without
recomputing the homography, and it becomes a real least-squares check on v2 boards,
whose fifth always-on LED over-determines the fit.

The dataset lives outside the repository, so the test skips when it is absent. Point
ROCSYNC_GROUND_TRUTH at a ground_truth.json to run it elsewhere.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from rocsync.benchmark.annotate import annotated_starts, derive_reference_clock, video_rel_paths
from rocsync.benchmark.common import (
    MIN_REFERENCE_FRAMES,
    REFERENCE_RESIDUAL_THRESHOLD_MS,
    ReferenceClock,
    collect_frames,
)
from rocsync.board_profiles import PROFILES_BY_ARUCO
from rocsync.camera import CameraType
from rocsync.timeline import frame_pts

GROUND_TRUTH = Path(
    os.environ.get(
        "ROCSYNC_GROUND_TRUTH", "/home/jonas/datasets/rocsync_benchmark/ground_truth.json"
    )
)

# Exact for a four-point fit; the slack covers v2's least-squares fit over five LEDs.
TOL_PX = 2.0

pytestmark = pytest.mark.skipif(
    not GROUND_TRUTH.is_file(), reason=f"ground truth not available at {GROUND_TRUTH}"
)


@pytest.fixture(scope="module")
def ground_truth():
    with open(GROUND_TRUTH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def annotations(ground_truth):
    return ground_truth["images"]


def _nominal_in_image(gt):
    """Modelled corner LED layout warped into image space, or None if not derivable."""
    H = gt.get("homography")
    aruco_id = gt.get("aruco", {}).get("id")
    if H is None or aruco_id not in PROFILES_BY_ARUCO:
        return None
    board = PROFILES_BY_ARUCO[aruco_id].rectify()
    inv_H = np.linalg.inv(np.array(H, dtype=np.float64))
    pts = np.array([board.always_on_leds[CameraType.RGB]], dtype=np.float64)
    return cv2.perspectiveTransform(pts, inv_H).reshape(-1, 2)


def test_annotated_corners_match_the_warped_layout(annotations):
    """The homography and the annotated positions describe the same board pose.

    Evaluation compares predictions against the annotated positions; this is what
    licenses that, rather than reprojecting the modelled layout instead.
    """
    offenders = []
    compared = 0

    for key, gt in annotations.items():
        nominal = _nominal_in_image(gt)
        if nominal is None:
            continue
        for i, corner in enumerate(gt.get("corners", [])):
            if not corner.get("visible") or corner.get("position") is None:
                continue
            if i >= len(nominal):
                continue
            err = float(np.linalg.norm(np.array(corner["position"], dtype=np.float64) - nominal[i]))
            compared += 1
            if err > TOL_PX:
                offenders.append(f"{key} corner {i}: {err:.2f} px")

    assert compared > 0, "no annotated corner had a homography to check against"
    assert not offenders, (
        f"{len(offenders)}/{compared} annotated corners disagree with the warped layout "
        f"by more than {TOL_PX} px:\n  " + "\n  ".join(offenders[:20])
    )


def test_annotated_positions_are_image_space(annotations):
    """Positions are image coordinates, not the 640 px rectified grid.

    The two spaces are easy to confuse and a file in the wrong one would still load.
    Real frames are far larger than the board grid, so the spread gives it away.
    """
    xs = [
        c["position"][0]
        for gt in annotations.values()
        for c in gt.get("corners", [])
        if c.get("visible") and c.get("position") is not None
    ]
    assert xs, "no annotated corner positions found"
    assert max(xs) > 640, (
        f"all annotated x coordinates fall within {max(xs):.0f} px; "
        "the file may hold rectified board coordinates instead of image coordinates"
    )


def test_every_reference_clock_still_fits_its_annotations(ground_truth):
    """A stored reference clock reproduces, and no annotation contradicts it.

    The evaluation measures fitted clocks against these numbers, so a residual beyond
    the threshold means the reference is scoring against a frame that is itself wrong.
    Deriving it here rather than trusting the file also catches annotations edited
    without re-running `rocsync-annotate --fit-clocks`.
    """
    references = ground_truth.get("videos", {})
    if not references:
        pytest.skip("no reference clocks in this ground truth")

    for rel_path, stored in references.items():
        pts = dict(enumerate(frame_pts(GROUND_TRUTH.parent / rel_path)))
        clock, outliers = derive_reference_clock(ground_truth["images"], rel_path, pts)

        assert clock is not None, f"{rel_path}: too few annotations to re-derive"
        assert not outliers, (
            f"{rel_path}: {len(outliers)} annotated frame(s) beyond "
            f"{REFERENCE_RESIDUAL_THRESHOLD_MS} ms:\n  "
            + "\n  ".join(f"#{i:06d} {residual:+.2f} ms" for i, residual in outliers[:20])
        )
        assert clock == ReferenceClock.from_dict(stored), (
            f"{rel_path}: stored reference does not match the annotations it claims to "
            f"describe; re-run `rocsync-annotate --fit-clocks`"
        )


def test_every_sufficiently_annotated_video_has_a_reference_clock(ground_truth):
    """A video the evaluation could score must not be silently unscored."""
    frames = collect_frames(GROUND_TRUTH.parent)
    references = ground_truth.get("videos", {})

    missing = [
        rel_path
        for rel_path in video_rel_paths(frames)
        if rel_path not in references
        and len(annotated_starts(ground_truth["images"], rel_path)) >= MIN_REFERENCE_FRAMES
    ]

    assert not missing, (
        "videos with enough annotations but no reference clock: "
        + ", ".join(missing)
        + "; run `rocsync-annotate --fit-clocks`"
    )
