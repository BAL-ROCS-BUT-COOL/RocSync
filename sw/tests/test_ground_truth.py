"""Check a ground truth file against itself.

Each annotation carries corner LED positions in image space and a homography, and the
evaluation relies on the two describing the same board pose: warping the modelled LED
layout through the homography must land on the annotated positions. That holds where the
visible corners determine the homography — a four-point ``getPerspectiveTransform`` fit
has no residual, and a v2 board's fifth always-on LED turns it into a least-squares fit
that lands within a few pixels. It earns its keep by catching a file whose positions were
edited without recomputing the homography.

Below four corners in general position there is nothing to refit, so the annotator leaves
whatever homography was in place while the corners are dragged. Those frames are reported
rather than failed: the annotation tool is where that gap belongs, not a test.

The dataset lives outside the repository, so the test skips when it is absent. Point
ROCSYNC_GROUND_TRUTH at a ground_truth.json to run it elsewhere.
"""

import json
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

from rocsync.benchmark.annotate import (
    annotated_starts,
    derive_reference_clock,
    fit_corner_homography,
    video_rel_paths,
)
from rocsync.benchmark.common import (
    MIN_REFERENCE_FRAMES,
    ReferenceClock,
    annotation_camera,
    collect_frames,
    reference_outliers,
    residual_threshold_ms,
)
from rocsync.board_profiles import PROFILES_BY_ARUCO
from rocsync.timeline import frame_pts

GROUND_TRUTH = Path(
    os.environ.get(
        "ROCSYNC_GROUND_TRUTH", "/home/jonas/datasets/rocsync_benchmark/ground_truth.json"
    )
)

# Exact for a four-point fit; the slack covers v2's least-squares fit over five LEDs,
# whose two leftmost LEDs sit 38 px apart and so trade a few pixels against each other.
TOL_PX = 4.0

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
    pts = np.array([board.always_on_leds[annotation_camera(gt)]], dtype=np.float64)
    return cv2.perspectiveTransform(pts, inv_H).reshape(-1, 2)


def _determines_its_homography(gt):
    """Whether this frame's visible corners are the ones its homography could come from."""
    aruco_id = gt.get("aruco", {}).get("id")
    if aruco_id not in PROFILES_BY_ARUCO:
        return False
    corners = [
        {"visible": bool(c.get("visible")), "position": c.get("position")}
        for c in gt.get("corners", [])
    ]
    board = PROFILES_BY_ARUCO[aruco_id].rectify()
    return fit_corner_homography(corners, board, annotation_camera(gt)) is not None


def test_annotated_corners_match_the_warped_layout(annotations):
    """The homography and the annotated positions describe the same board pose.

    Evaluation compares predictions against the annotated positions; this is what
    licenses that, rather than reprojecting the modelled layout instead.
    """
    offenders = []
    unverifiable = []
    compared = 0

    for key, gt in annotations.items():
        nominal = _nominal_in_image(gt)
        if nominal is None:
            continue
        if not _determines_its_homography(gt):
            unverifiable.append(key)
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

    if unverifiable:
        warnings.warn(
            f"{len(unverifiable)} frame(s) have too few corners to determine a homography, "
            "so their annotated positions and their homography cannot be checked against "
            "each other:\n  " + "\n  ".join(unverifiable[:20]),
            stacklevel=1,
        )

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


def clip_starts(images, rel_path, stored):
    """Annotated board times in one clip's own frame numbering.

    A retimed clip has no annotations of its own -- they stay with the recording it
    was cut from -- so its stored mapping is what puts them back on its frames.
    """
    source = stored.get("source", rel_path)
    offset = int(stored.get("source_frame_offset", 0))
    return {index - offset: start for index, start in annotated_starts(images, source).items()}


def test_every_reference_clock_still_fits_its_annotations(ground_truth):
    """No annotation contradicts the reference clock stored for its video.

    The evaluation measures fitted clocks against these numbers, so a residual beyond
    the video's tolerance means the reference is scoring against a frame that is itself
    wrong. A measured reference is additionally re-derived here rather than trusted,
    which catches annotations edited without re-running `rocsync-annotate --fit-clocks`;
    a synthesized one is drawn rather than fitted, so the annotations are what it has to
    answer to, not a least-squares line through them.
    """
    references = ground_truth.get("videos", {})
    if not references:
        pytest.skip("no reference clocks in this ground truth")

    for rel_path, stored in references.items():
        pts = dict(enumerate(frame_pts(GROUND_TRUTH.parent / rel_path)))
        threshold = residual_threshold_ms(stored)
        starts = clip_starts(ground_truth["images"], rel_path, stored)
        outliers = reference_outliers(ReferenceClock.from_dict(stored), starts, pts, threshold)

        assert len(starts) >= MIN_REFERENCE_FRAMES, f"{rel_path}: too few annotations to check"
        assert not outliers, (
            f"{rel_path}: {len(outliers)} annotated frame(s) beyond {threshold:.2f} ms:\n  "
            + "\n  ".join(f"#{i:06d} {residual:+.2f} ms" for i, residual in outliers[:20])
        )
        if stored.get("timeline") == "synthesized":
            continue

        derived, _ = derive_reference_clock(starts, pts, threshold)
        assert derived == ReferenceClock.from_dict(stored), (
            f"{rel_path}: stored reference does not match the annotations it claims to "
            f"describe; re-run `rocsync-annotate --fit-clocks`"
        )


def test_every_sufficiently_annotated_video_has_a_reference_clock(ground_truth):
    """A video the evaluation could score must not be silently unscored."""
    frames = collect_frames(GROUND_TRUTH.parent, sources_only=True)
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
