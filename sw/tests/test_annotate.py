"""The annotation tool's corner fit.

The rectified board view follows the corner LEDs the user has annotated, and four of
them are enough — a v2 board hides one of its five often enough that waiting for all of
them would leave the counter and ring unannotatable. What four is *not* enough for is a
set whose board coordinates are collinear, which v2's LEDs 0, 1 and 4 are.
"""

import cv2
import numpy as np
import pytest

from rocsync.benchmark.annotate import MIN_HOMOGRAPHY_CORNERS, fit_corner_homography
from rocsync.board_profiles import PROFILES_BY_ARUCO
from rocsync.camera import CameraType

BOARDS = {p.name: p.rectify() for p in PROFILES_BY_ARUCO.values()}

# An arbitrary but firmly perspective board pose, as image ← board.
POSE = np.array(
    [
        [0.9, 0.15, 120.0],
        [-0.1, 0.85, 260.0],
        [2.0e-4, 1.5e-4, 1.0],
    ]
)


def annotation_corners(board, visible):
    """Corner annotations for `board` under POSE, with `visible` the visible indices."""
    coords = np.array(board.always_on_leds[CameraType.RGB], dtype=np.float64)
    image_pts = cv2.perspectiveTransform(coords.reshape(1, -1, 2), POSE).reshape(-1, 2)
    return [
        {"visible": i in visible, "position": [float(x), float(y)]}
        for i, (x, y) in enumerate(image_pts)
    ]


def board_error(H, board, indices):
    """Largest distance, in board pixels, between H's mapping and the modelled layout."""
    corners = annotation_corners(board, set(indices))
    src = np.array([corners[i]["position"] for i in indices], dtype=np.float64)
    mapped = cv2.perspectiveTransform(src.reshape(1, -1, 2), H).reshape(-1, 2)
    want = np.array([board.always_on_leds[CameraType.RGB][i] for i in indices])
    return float(np.abs(mapped - want).max())


@pytest.mark.parametrize("name", sorted(BOARDS))
def test_too_few_corners_have_no_fit(name):
    """Three points leave a projective map undetermined, so no fit is offered."""
    board = BOARDS[name]
    assert fit_corner_homography(annotation_corners(board, {0, 1, 2}), board) is None


@pytest.mark.parametrize("name", sorted(BOARDS))
def test_four_corners_fit_exactly(name):
    """The minimum the tool now accepts, and it reproduces the pose it was built from."""
    board = BOARDS[name]
    indices = [0, 1, 2, 3]
    H = fit_corner_homography(annotation_corners(board, set(indices)), board)
    assert H is not None
    assert board_error(H, board, indices) < 1e-3


def test_a_collinear_corner_set_has_no_fit():
    """v2's LEDs 0, 1 and 4 sit on one line: four points, but only three constraints."""
    board = BOARDS["v2"]
    indices = {0, 1, 2, 4}
    assert len(indices) == MIN_HOMOGRAPHY_CORNERS
    assert fit_corner_homography(annotation_corners(board, indices), board) is None


def test_all_five_corners_fit_by_least_squares():
    """The over-determined case still lands on the modelled layout."""
    board = BOARDS["v2"]
    indices = [0, 1, 2, 3, 4]
    H = fit_corner_homography(annotation_corners(board, set(indices)), board)
    assert H is not None
    assert board_error(H, board, indices) < 1.0


@pytest.mark.parametrize("hidden", [0, 1, 4])
def test_one_hidden_corner_still_fits(hidden):
    """v2 keeps a usable board view through the occlusions that leave a proper quad."""
    board = BOARDS["v2"]
    indices = [i for i in range(5) if i != hidden]
    H = fit_corner_homography(annotation_corners(board, set(indices)), board)
    assert H is not None
    assert board_error(H, board, indices) < 1.0


@pytest.mark.parametrize("hidden", [2, 3])
def test_hiding_a_far_corner_leaves_a_collinear_set(hidden):
    """Losing 2 or 3 leaves LEDs 0, 1 and 4 plus one — one line and a point, no fit."""
    board = BOARDS["v2"]
    indices = {i for i in range(5) if i != hidden}
    assert fit_corner_homography(annotation_corners(board, indices), board) is None
