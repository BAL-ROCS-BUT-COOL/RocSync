"""Decode a RocSync board from 3D fiducials, a tracker's reconstructed markers.

The board plane comes from a registered rigid body's pose (``plane_from_pose``) or from
the corner-LED square among the fiducials (``plane_from_constellation``). The on-plane
fiducials are then decoded by the 2D-point route, whose corner search settles which LED
is which; a pose fixes only the plane, not the in-plane origin or orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from rocsync.blobs import decode_camera
from rocsync.board_profiles import BoardProfile
from rocsync.decode import CLUTTERED, NO_BOARD, Decode

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# The board puts ~20-30 fiducials in its plane; the corner search is quadratic in this
MAX_FIDUCIALS = 64


@dataclass
class PlaneFit:
    origin: np.ndarray  # (3,) a point on the board plane
    basis: np.ndarray  # (2, 3) orthonormal in-plane axes
    source: str  # "pose" or "constellation"


def plane_from_rotation(position_xyz, rotation) -> PlaneFit:
    """The board plane from a tracked rigid body's position and 3x3 rotation."""
    rot = np.asarray(rotation, dtype=float).reshape(3, 3)
    origin = np.asarray(position_xyz, dtype=float)
    return PlaneFit(origin=origin, basis=rot[:, :2].T.copy(), source="pose")


def plane_from_pose(position_xyz, quaternion_xyzw) -> PlaneFit:
    """The board plane from a tracked rigid body's position and (x, y, z, w) rotation."""
    return plane_from_rotation(position_xyz, Rotation.from_quat(quaternion_xyzw).as_matrix())


def plane_from_constellation(points: np.ndarray, tolerance_mm: float = 6.0) -> PlaneFit | None:
    """The board plane from four fiducials forming the corner square."""
    corner_side_mm = 240.0
    corner_diagonal_mm = corner_side_mm * np.sqrt(2)

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 4:
        return None

    # Match pairs on the diagonal length first: O(N^2) rather than O(N^4) quads
    idx_i, idx_j = np.triu_indices(len(points), k=1)
    lengths = np.linalg.norm(points[idx_i] - points[idx_j], axis=1)
    diagonals = np.where(np.abs(lengths - corner_diagonal_mm) <= tolerance_mm)[0]
    if len(diagonals) < 2:
        return None

    midpoints = (points[idx_i] + points[idx_j]) / 2.0
    for a in diagonals:
        for b in diagonals:
            if b <= a:
                continue
            quad = {idx_i[a], idx_j[a], idx_i[b], idx_j[b]}
            if len(quad) != 4:
                continue
            # A square's diagonals bisect each other
            if np.linalg.norm(midpoints[a] - midpoints[b]) > tolerance_mm:
                continue
            corners = points[sorted(quad)]
            centred = corners - corners.mean(axis=0)
            # Plane basis: the two dominant singular directions of the corners
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            basis = vt[:2]
            flat = centred @ basis.T
            sides = np.sort(np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=2).ravel())
            # 4 zeros, then 4 sides, then 2 diagonals; the sides rule out a rectangle
            if not np.all(np.abs(sides[4:8] - corner_side_mm) <= tolerance_mm * 2):
                continue
            return PlaneFit(origin=corners.mean(axis=0), basis=basis, source="constellation")
    return None


def project(points: np.ndarray, plane: PlaneFit, plane_tolerance_mm: float) -> np.ndarray:
    """In-plane 2D coordinates of the fiducials within ``plane_tolerance_mm`` of the plane."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points):
        return np.empty((0, 2))
    relative = points - plane.origin
    normal = np.cross(plane.basis[0], plane.basis[1])
    normal /= np.linalg.norm(normal) or 1.0
    on_plane = np.abs(relative @ normal) <= plane_tolerance_mm
    return relative[on_plane] @ plane.basis.T


def decode_fiducials(
    points_3d: np.ndarray,
    profile: BoardProfile,
    plane: PlaneFit | None,
    plane_tolerance_mm: float = 5.0,
    tolerance_mm: float = 6.0,
    max_fiducials: int = MAX_FIDUCIALS,
    ax: Axes | None = None,
) -> Decode:
    """Decode from 3D fiducials on ``plane``, or on the corner constellation if None.

    More than ``max_fiducials`` points are rejected as ``CLUTTERED`` before any search.
    """
    points_3d = np.asarray(points_3d, dtype=float).reshape(-1, 3)
    if len(points_3d) > max_fiducials:
        return Decode(reject=CLUTTERED)
    if plane is None:
        plane = plane_from_constellation(points_3d, tolerance_mm)
    if plane is None:
        return Decode(reject=NO_BOARD)

    flat = project(points_3d, plane, plane_tolerance_mm)
    return decode_camera(flat * profile.rectify().px_per_mm, profile, ax=ax)
