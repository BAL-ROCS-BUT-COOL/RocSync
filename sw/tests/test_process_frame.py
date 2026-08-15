from pathlib import Path

import cv2 as cv
import pytest

from rocsync.board_profiles import BOARD_V2
from rocsync.camera import CameraType
from rocsync.vision import process_frame, read_counter, read_ring, rectify_board

TEST_DIR = Path(__file__).resolve().parent

# What the board in img1.jpg reads: 23.5 s, exposed over ring LEDs 2 to 14.
EXPECTED_RING = (2, 14)
EXPECTED_COUNTER = 235
EXPECTED_BOARD_TIME = (23502, 23514)


@pytest.fixture
def image():
    image = cv.imread(str(TEST_DIR / "img1.jpg"))
    assert image is not None, "test image could not be read"
    return image


def test_piecewise(image, tmp_path):
    """Rectification and each LED reading of an RGB frame, stage by stage."""
    detected, pcb, board = rectify_board(
        image, CameraType.RGB, frame_number=999, board=BOARD_V2, debug_dir=tmp_path
    )
    assert detected, "board was not detected"
    assert pcb is not None, "board was detected but could not be rectified"
    cv.imwrite(str(tmp_path / "pcb.jpg"), pcb)

    leds_overlay = cv.cvtColor(pcb, cv.COLOR_GRAY2BGR)
    ring = read_ring(pcb, CameraType.RGB, board, draw_on=leds_overlay)
    counter = read_counter(pcb, CameraType.RGB, board, draw_on=leds_overlay)
    cv.imwrite(str(tmp_path / "leds.jpg"), leds_overlay)

    assert ring == EXPECTED_RING
    assert counter == EXPECTED_COUNTER


def test_full(image, tmp_path):
    """The whole RGB frame pipeline, from image to board time."""
    detected, board_time = process_frame(
        image, CameraType.RGB, frame_number=999, board=BOARD_V2, debug_dir=tmp_path
    )
    assert detected, "board was not detected"
    assert board_time == EXPECTED_BOARD_TIME
