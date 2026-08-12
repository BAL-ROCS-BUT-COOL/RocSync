"""Render a board the way an IR camera sees it, then decode it back.

The repo has an RGB fixture but no IR one, and the IR path is the half that has no
ArUco marker to lean on: it has to find the board among the blobs, settle its
orientation, and only then decode. Synthesising the frame from the board profile lets
that whole path be exercised against a known answer, at viewing angles a fixture cannot
cover.
"""

import cv2
import numpy as np
import pytest

from rocsync.board_profiles import BOARD_V1, BOARD_V2
from rocsync.camera import CameraType
from rocsync.vision import process_frame

IMAGE_SIZE = 900
# read_led takes the 0.75 quantile of a disc of LED_SAMPLE_RADIUS_MM, so a rendered LED
# has to cover comfortably more than a quarter of that disc to read as lit. It also has
# to stay smaller than the ~19 px ring spacing, or neighbouring LEDs merge into one blob.
LED_RADIUS = 7


def render(profile, counter, ring_window, homography, board_size=640):
    """An IR-style frame: white discs on black, at every LED that would be lit."""
    board = profile.rectify(board_size)
    lit = [board.always_on_leds[CameraType.INFRARED]]

    bits = board.counter_led_coords[CameraType.INFRARED]
    n = board.counter_bits
    lit.append(
        np.array([p for i, p in enumerate(bits) if counter >> (n - 1 - i) & 1]).reshape(
            -1, 2
        )
    )

    start, end = ring_window
    ring = board.ring_led_coords(CameraType.INFRARED)
    lit.append(np.array([ring[i % board.period] for i in range(start, end + 1)]))

    points = np.concatenate(lit).astype(np.float32)
    warped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(
        -1, 2
    )

    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    for x, y in warped:
        cv2.circle(image, (round(x), round(y)), LED_RADIUS, (255, 255, 255), -1)
    return image


def view(board_size, corners):
    """Homography taking the rectified board square onto ``corners`` in the image."""
    square = np.array(
        [[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(square, np.array(corners, dtype=np.float32))


# Head-on, and two oblique views. The board is 4-fold symmetric apart from the 5th LED,
# so these also check that the search settles on the right orientation rather than a
# rotation of it.
VIEWS = {
    "head-on": [(100, 100), (800, 100), (800, 800), (100, 800)],
    "tilted": [(160, 120), (780, 200), (740, 790), (120, 700)],
    "rotated-90": [(800, 100), (800, 800), (100, 800), (100, 100)],
}


@pytest.mark.parametrize("profile", [BOARD_V1, BOARD_V2], ids=["v1", "v2"])
@pytest.mark.parametrize("view_name", list(VIEWS))
def test_ir_roundtrip(profile, view_name):
    board_size = 640
    counter = 12345 if profile.counter_bits >= 20 else 4321
    ring_window = (10, 30)

    image = render(
        profile, counter, ring_window, view(board_size, VIEWS[view_name]), board_size
    )
    found, result = process_frame(
        image, CameraType.INFRARED, frame_number=0, board=profile
    )

    assert found, f"{profile.name} {view_name}: board not located"
    assert result is not None, f"{profile.name} {view_name}: decoded nothing"

    start, end = result
    expected_start = ring_window[0] + counter * profile.period
    expected_end = ring_window[1] + counter * profile.period
    assert (start, end) == (expected_start, expected_end)


@pytest.mark.xfail(
    strict=True,
    reason="MIN_SCORE=4 is at the noise floor: the layout mask covers 9.3% of the "
    "board, so ~36 stray blobs are expected to put ~3.4 points on it by chance, and "
    "the search maximises over 512 candidates x 4 rotations. Measured false-accept "
    "rate on pure noise: 0/100 at 10 blobs, 26/100 at 20, 47/100 at 40, 69/100 at 80. "
    "Pre-existing; fixing it is a tuning decision with a false-negative cost.",
)
def test_ir_rejects_a_frame_without_a_board():
    """Scattered blobs must not yield a confident decode."""
    rng = np.random.default_rng(0)
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    for x, y in rng.integers(50, IMAGE_SIZE - 50, size=(40, 2)):
        cv2.circle(image, (int(x), int(y)), LED_RADIUS, (255, 255, 255), -1)

    found, result = process_frame(
        image, CameraType.INFRARED, frame_number=0, board=BOARD_V2
    )
    assert result is None
