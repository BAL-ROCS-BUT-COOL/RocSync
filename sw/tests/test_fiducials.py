"""Synthetic tests for the 3D-fiducial route, including cases live data cannot be
relied on to show.

The board is built in its own frame from BOARD_V2, placed at an arbitrary pose, and
handed to the decoder as 3D points. Assertions are exact: board time is an integer
count of milliseconds, so a near miss is a wrong answer.

Two properties matter most here and are hard to check on hardware:

* **Origin-convention independence.** ``fiducial_decode.process_frame`` adds (5, 5) to
  transformed fiducials ("Adjust for PCB origin") while the current .ini already places
  its first fiducial at (5, 5) -- a convention this route sidesteps entirely by fitting
  the in-plane frame from the corner LEDs themselves. The test for it is that the decode
  is unchanged when the pose's translation is deliberately offset.
* **Off-plane rejection**, this route's advantage over the 2D-blob route. Confusers
  placed off the board plane must not participate.
"""

import math

import numpy as np
import pytest

from rocsync.board_profiles import BOARD_V2
from rocsync.camera import CameraType
from rocsync.fiducials import decode_fiducials, plane_from_pose


def board_points_mm(profile, counter: int, lit: list[int]) -> np.ndarray:
    """The lit LEDs of one board state, in board millimetres."""
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


def place(points_2d: np.ndarray, rpy=(0.3, -0.2, 0.7), t=(120.0, -80.0, 1400.0)):
    """Lift board-plane points into an arbitrary 3D pose. Returns (points_3d, rot, t)."""
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    rot = rz @ ry @ rx
    flat3 = np.column_stack([points_2d, np.zeros(len(points_2d))])
    return flat3 @ rot.T + np.asarray(t), rot, np.asarray(t)


def quat_from_matrix(rot):
    """(x, y, z, w) for a rotation matrix -- the inverse of fiducials._quat_to_matrix."""
    w = math.sqrt(max(0.0, 1.0 + rot[0, 0] + rot[1, 1] + rot[2, 2])) / 2.0
    w = w or 1e-9
    return (
        (rot[2, 1] - rot[1, 2]) / (4 * w),
        (rot[0, 2] - rot[2, 0]) / (4 * w),
        (rot[1, 0] - rot[0, 1]) / (4 * w),
        w,
    )


CASES = [(7, [40]), (48937, [27]), (1234, [10, 11])]


@pytest.mark.parametrize("counter,lit", CASES)
def test_decode_with_pose(counter, lit):
    profile = BOARD_V2
    flat = board_points_mm(profile, counter, lit)
    pts3, rot, t = place(flat)
    plane = plane_from_pose(t, quat_from_matrix(rot))
    result = decode_fiducials(pts3, profile, plane)

    assert result.reject is None, f"decode failed: {result.reject}"
    assert result.counter == counter
    assert result.board_ms == pytest.approx(counter * profile.period + lit[0])
    assert result.exposure_ms == pytest.approx(len(lit) - 1)
    assert result.plane_source == "pose"


@pytest.mark.parametrize("counter,lit", CASES)
def test_decode_without_a_registered_geometry(counter, lit):
    """The failover: find the corner square among the fiducials, no pose at all."""
    profile = BOARD_V2
    pts3, _, _ = place(board_points_mm(profile, counter, lit))
    result = decode_fiducials(pts3, profile, None)

    assert result.reject is None, f"failover failed: {result.reject}"
    assert result.counter == counter
    assert result.board_ms == pytest.approx(counter * profile.period + lit[0])
    assert result.plane_source == "constellation"


def test_decode_is_independent_of_the_pose_origin_convention():
    """The whole reason the in-plane frame is fitted rather than taken from the pose.

    fiducial_decode.process_frame adds (5, 5) to transformed fiducials while the
    current .ini already puts its first fiducial at (5, 5). If the decode depended on
    that convention, shifting the pose's translation would change the answer. It must
    not.
    """
    profile = BOARD_V2
    counter, lit = 4242, [55, 56]
    flat = board_points_mm(profile, counter, lit)
    pts3, rot, t = place(flat)

    baseline = decode_fiducials(pts3, profile, plane_from_pose(t, quat_from_matrix(rot)))
    assert baseline.reject is None

    for shift in ((5.0, 5.0, 0.0), (-5.0, -5.0, 0.0), (125.0, 125.0, 0.0)):
        offset_t = t + np.asarray(shift) @ rot.T
        shifted = decode_fiducials(pts3, profile, plane_from_pose(offset_t, quat_from_matrix(rot)))
        assert shifted.reject is None, f"offset {shift} broke the decode"
        assert shifted.board_ms == baseline.board_ms, (
            f"offset {shift} changed the decoded time -- the in-plane origin is leaking "
            "into the result"
        )


def test_off_plane_confusers_are_rejected():
    """This route's advantage over the blob route: a confuser off the plane cannot match."""
    profile = BOARD_V2
    counter, lit = 999, [70]
    flat = board_points_mm(profile, counter, lit)
    pts3, rot, t = place(flat)

    # Markers scattered well off the board plane, including some that would project
    # INSIDE the board outline -- exactly the case a homography cannot exclude.
    normal = np.cross(rot[:, 0], rot[:, 1])
    normal /= np.linalg.norm(normal)
    confusers = np.array(
        [
            pts3.mean(axis=0) + normal * 200.0,
            pts3.mean(axis=0) - normal * 150.0,
            pts3[0] + normal * 80.0,
            pts3[3] - normal * 300.0,
        ]
    )
    result = decode_fiducials(
        np.vstack([pts3, confusers]), profile, plane_from_pose(t, quat_from_matrix(rot))
    )

    assert result.reject is None, f"confusers broke the decode: {result.reject}"
    assert result.counter == counter
    assert result.board_ms == pytest.approx(counter * profile.period + lit[0])


def test_no_board_is_refused():
    """A handful of unrelated markers must not produce a timestamp."""
    pts = np.array(
        [[0.0, 0.0, 1000.0], [50.0, 10.0, 1010.0], [10.0, 60.0, 990.0], [70.0, 70.0, 1005.0]]
    )
    assert decode_fiducials(pts, BOARD_V2, None).reject is not None


def test_empty_input_is_refused():
    assert decode_fiducials(np.empty((0, 3)), BOARD_V2, None).reject is not None
