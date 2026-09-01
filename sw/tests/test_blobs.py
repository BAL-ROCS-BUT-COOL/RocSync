"""Synthetic tests for the 2D-blob route, including the failure mode it cannot defend
against.

Points are synthesised by projecting the board's lit LEDs through a known homography,
which is what a tracker's centroid list looks like. Assertions are exact -- board time
is an integer count of milliseconds.

The blob-status filtering (STATUS_MERGED, STATUS_INVALID) that the ROS package's
route_blob.py also carried stays downstream: it depends on
``marker_tracking_msgs/Blob``'s own status-bit convention, which this module never sees
-- only the plain ``(N, 2)`` centroid array it filters down to.

The last test is deliberately a *negative* result: this route has no depth information,
so a confuser lying along a ray through the board plane is indistinguishable from an
LED. That is recorded here rather than left implicit, because it is the reason the
3D-fiducial route is preferred where both are available.
"""

import cv2
import numpy as np
import pytest

from rocsync.blobs import decode_camera
from rocsync.board_profiles import BOARD_V2
from rocsync.camera import CameraType


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


def test_no_board_is_refused():
    points = np.array([(100, 100), (300, 120), (140, 380), (320, 400), (500, 500)])
    assert decode_camera(points, BOARD_V2).reject is not None


def test_too_few_points_is_refused():
    assert decode_camera(np.array([(10, 10), (20, 20)]), BOARD_V2).reject is not None
    assert decode_camera(np.empty((0, 2)), BOARD_V2).reject is not None


def test_a_confuser_adjacent_to_the_arc_is_not_rejectable():
    """This route's structural weakness, recorded rather than glossed over.

    With no depth, a point from an object in front of or behind the board projects
    inside the board outline and is indistinguishable from an LED. Here one is placed
    exactly on the ring LED position immediately after the true arc: the decoder cannot
    know it is not a real detection, and it reads as the arc simply running one LED
    longer, changing the exposure and end time. The 3D-fiducial route rejects the same
    confuser by coplanarity.

    A confuser placed further from the arc, with unlit LEDs between it and the real one,
    is a weaker case: ``board_profiles.decode_ring``'s tolerant best-window search (see
    ``test_fiducial_decode.py::test_tolerates_an_isolated_false_ring_detection``) treats
    an isolated stray point as noise and returns the honest answer regardless -- an
    improvement over the ROS package's stricter original reader, not a regression, but it
    means an isolated confuser no longer demonstrates this weakness. An adjacent one still
    does, which is what is exercised here.

    This is why the 3D-fiducial route is preferred wherever it is available, and why the
    fit's counter-jump guard and RANSAC threshold are the real defences on this one.
    """
    profile = BOARD_V2
    counter, lit = 800, [20]
    ring = profile.ring_led_coords(CameraType.INFRARED)
    fake_mm = ring[21 % profile.period].reshape(1, 2)  # immediately after the true arc

    honest = decode_camera(points_for(counter, lit), profile)
    assert honest.reject is None and honest.board_ms == counter * profile.period + lit[0]

    px = to_pixels(np.vstack([lit_board_mm(profile, counter, lit), fake_mm]))
    spoofed = decode_camera(px, profile)
    # The arc now reads one LED longer -- what it must NOT do is silently return the
    # honest answer, because the decoder genuinely cannot tell the two cases apart.
    assert spoofed.reject is not None or spoofed.exposure_ms != honest.exposure_ms, (
        "an adjacent confuser was silently ignored, which would mean the test is not "
        "exercising the weakness it claims to"
    )
