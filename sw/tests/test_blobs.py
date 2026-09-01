"""The 2D-point route, on points projected from the board's lit LEDs through a homography."""

import cv2
import numpy as np
import pytest

from rocsync.blobs import decode_camera
from rocsync.board_profiles import BOARD_V2
from rocsync.camera import CameraType
from rocsync.decode import NO_BOARD


def lit_board_mm(profile, counter: int, lit: list[int]) -> np.ndarray:
    ir = CameraType.INFRARED
    pts = [profile.always_on_leds[ir]]
    ring = profile.ring_led_coords(ir)
    if lit:
        pts.append(np.asarray([ring[i % profile.period] for i in lit]))
    bits = profile.counter_led_coords[ir]
    n = profile.counter_bits
    on = [bits[i] for i in range(n) if counter & (1 << (n - 1 - i))]
    if on:
        pts.append(np.asarray(on))
    return np.concatenate(pts, axis=0)


def to_pixels(points_mm: np.ndarray, quad=None) -> np.ndarray:
    """Project board millimetres into a camera image through a homography."""
    src = np.array([[0, 0], [250, 0], [250, 250], [0, 250]], dtype=np.float32)
    if quad is None:
        quad = np.array([[420, 300], [1180, 340], [1130, 1010], [380, 960]])
    matrix = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    return cv2.perspectiveTransform(points_mm.reshape(-1, 1, 2).astype(np.float32), matrix).reshape(
        -1, 2
    )


def points_for(counter: int, lit: list[int], extra=()) -> np.ndarray:
    px = to_pixels(lit_board_mm(BOARD_V2, counter, lit))
    if len(extra):
        px = np.vstack([px, np.asarray(extra, dtype=float)])
    return px


CASES = [(7, [40]), (48937, [27]), (1234, [10, 11])]


@pytest.mark.parametrize("counter,lit", CASES)
def test_decode_from_blobs(counter, lit):
    result = decode_camera(points_for(counter, lit), BOARD_V2)
    assert result.reject is None, f"decode failed: {result.reject}"
    assert result.counter == counter
    assert result.board_ms == pytest.approx(counter * BOARD_V2.period + lit[0])
    assert result.exposure_ms == pytest.approx(len(lit) - 1)


def test_other_markers_in_view_do_not_prevent_a_decode():
    """A tracker's centroid list contains everything retroreflective in the volume."""
    extra = [(150, 120), (200, 140), (170, 175), (1600, 900), (1700, 250), (90, 800)]
    result = decode_camera(points_for(4242, [55, 56], extra=extra), BOARD_V2)
    assert result.reject is None, f"confusers broke the decode: {result.reject}"
    assert result.counter == 4242
    assert result.board_ms == pytest.approx(4242 * BOARD_V2.period + 55)


@pytest.mark.parametrize(
    "points",
    [
        np.array([(100, 100), (300, 120), (140, 380), (320, 400), (500, 500)]),
        np.array([(10, 10), (20, 20)]),
        np.empty((0, 2)),
    ],
    ids=["scattered", "too_few", "empty"],
)
def test_no_board_is_refused(points):
    assert decode_camera(points, BOARD_V2).reject == NO_BOARD
