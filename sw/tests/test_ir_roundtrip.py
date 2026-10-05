"""Render a board the way an IR camera sees it, then locate its corners."""

import cv2
import numpy as np
import pytest

from rocsync.board_detection import find_corners_layout
from rocsync.board_profiles import BOARD_V1, BOARD_V2
from rocsync.camera import CameraType

IMAGE_SIZE = 900
# Large enough to read as lit, small enough that neighbouring ring LEDs stay separate
LED_RADIUS = 7


def render(profile, counter, ring_window, homography, board_size=640):
    """An IR-style binary mask: lit discs at every LED position, everywhere else 0."""
    board = profile.rectify(board_size)
    lit = [board.always_on_leds[CameraType.INFRARED]]

    bits = board.counter_led_coords[CameraType.INFRARED]
    n = board.counter_bits
    lit.append(
        np.array([p for i, p in enumerate(bits) if counter >> (n - 1 - i) & 1]).reshape(-1, 2)
    )

    start, end = ring_window
    ring = board.ring_led_coords(CameraType.INFRARED)
    lit.append(np.array([ring[i % board.period] for i in range(start, end + 1)]))

    points = np.concatenate(lit).astype(np.float32)
    warped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 2)

    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    for x, y in warped:
        cv2.circle(mask, (round(x), round(y)), LED_RADIUS, 255, -1)
    return mask


def view(board_size, corners):
    """Homography taking the rectified board square onto ``corners`` in the image."""
    square = np.array(
        [[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(square, np.array(corners, dtype=np.float32))


def _by_angle(points):
    """Points sorted around their centroid, so two windings of the same quad compare equal."""
    centre = points.mean(axis=0)
    order = np.argsort(np.arctan2(*(points - centre).T[::-1]))
    return points[order]


# Includes a 90-degree rotation, so the search must also settle the orientation
VIEWS = {
    "head-on": [(100, 100), (800, 100), (800, 800), (100, 800)],
    "tilted": [(160, 120), (780, 200), (740, 790), (120, 700)],
    "rotated-90": [(800, 100), (800, 800), (100, 800), (100, 100)],
}


@pytest.mark.parametrize("profile", [BOARD_V1, BOARD_V2], ids=["v1", "v2"])
@pytest.mark.parametrize("view_name", list(VIEWS))
def test_find_corners_layout_locates_the_board(profile, view_name):
    board_size = 640
    counter = 12345 if profile.counter_bits >= 20 else 4321
    ring_window = (10, 30)
    homography = view(board_size, VIEWS[view_name])

    mask = render(profile, counter, ring_window, homography, board_size)
    found = find_corners_layout(mask, profile.rectify(board_size), frame_number=0)

    assert found is not None, f"{profile.name} {view_name}: board not located"
    # Expected corners are the warped anchor LEDs, which sit inset from the board edge
    board = profile.rectify(board_size)
    anchors = board.transform_corners(CameraType.INFRARED)
    expected = cv2.perspectiveTransform(anchors.reshape(-1, 1, 2), homography).reshape(-1, 2)
    np.testing.assert_allclose(
        _by_angle(np.asarray(found, dtype=np.float64)), _by_angle(expected), atol=1.0
    )


def _noise_mask(rng, n_blobs):
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    for x, y in rng.integers(50, IMAGE_SIZE - 50, size=(n_blobs, 2)):
        cv2.circle(mask, (int(x), int(y)), LED_RADIUS, 255, -1)
    # Same morphological close as vision.py's IR preprocessing
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def test_find_corners_layout_rarely_accepts_a_frame_without_a_board():
    """Scattered blobs rarely yield an accepted fit; chance coincidences cannot be excluded."""
    board = BOARD_V2.rectify(640)
    rng = np.random.default_rng(0)
    false_accepts = sum(
        find_corners_layout(_noise_mask(rng, 20), board, frame_number=i) is not None
        for i in range(100)
    )
    assert false_accepts <= 15, f"{false_accepts}/100 false accepts at 20 blobs"
