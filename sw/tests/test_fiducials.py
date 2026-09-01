"""The 3D-fiducial route, on a synthetic board placed at an arbitrary pose.

Board time is an integer count of milliseconds, so assertions are exact.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rocsync.board_profiles import BOARD_V2
from rocsync.camera import CameraType
from rocsync.decode import NO_BOARD
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
    rot = Rotation.from_euler("xyz", rpy).as_matrix()
    flat3 = np.column_stack([points_2d, np.zeros(len(points_2d))])
    return flat3 @ rot.T + np.asarray(t), rot, np.asarray(t)


def quat_from_matrix(rot):
    return Rotation.from_matrix(rot).as_quat()


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


@pytest.mark.parametrize("counter,lit", CASES)
def test_decode_without_a_registered_geometry(counter, lit):
    """Without a pose, the plane comes from the corner square among the fiducials."""
    profile = BOARD_V2
    pts3, _, _ = place(board_points_mm(profile, counter, lit))
    result = decode_fiducials(pts3, profile, None)

    assert result.reject is None, f"failover failed: {result.reject}"
    assert result.counter == counter
    assert result.board_ms == pytest.approx(counter * profile.period + lit[0])


def test_decode_is_independent_of_the_pose_origin_convention():
    """Shifting the pose's in-plane origin must not change the decoded time."""
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
        assert shifted.board_ms == baseline.board_ms, f"offset {shift} changed the decoded time"


def test_off_plane_confusers_are_rejected():
    """Markers off the board plane do not match, even where they project inside it."""
    profile = BOARD_V2
    counter, lit = 999, [70]
    flat = board_points_mm(profile, counter, lit)
    pts3, rot, t = place(flat)

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


@pytest.mark.parametrize(
    "points",
    [
        np.array(
            [[0.0, 0.0, 1000.0], [50.0, 10.0, 1010.0], [10.0, 60.0, 990.0], [70.0, 70.0, 1005.0]]
        ),
        np.empty((0, 3)),
    ],
    ids=["unrelated_markers", "empty"],
)
def test_no_board_is_refused(points):
    assert decode_fiducials(points, BOARD_V2, None).reject == NO_BOARD
