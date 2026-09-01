"""Decode a RocSync board from 3D fiducials -- a tracker's own reconstructed markers,
with no image to rectify.

Two ways to find the board plane, tried cheapest first:

1. **A registered rigid geometry.** With ``misc/ftk_geometry_rev{1,2}.ini`` registered,
   the tracker returns the board's own 6-DoF pose directly.
2. **A corner constellation search.** The corner LEDs form a square of side 240 mm and
   diagonal ~339.4 mm, a distinctive signature among stray fiducials, needing no
   registration at all.

Measured live, (2) is the workhorse and (1) is the exception: a registered geometry
matched in only ~4% of frames against 100% for an ordinary tracked tool in the same
volume, most likely because the board's own ring and counter LEDs put ~20 further
coplanar fiducials in the same plane, which is ill-conditioned for constellation
matching and offers a marker-matcher plenty of false candidates to prefer instead.

So calling (2) a fallback undersells it -- it carries the large majority of live
decodes, and the pose-derived ones pool into the same fit with zero outliers, which is
also the evidence the two paths agree.

Either way the result is a plane. The fiducials are projected into it and handed to
``board_detection.find_board``, the same corner search every route uses -- which is why
even the pose path still runs it: a pose only fixes the plane's orientation and origin,
not which LED is which, and ``find_board`` settles that from what is actually lit,
insensitive to the plane's convention.

This is a different, more conservative choice than ``fiducial_decode.process_frame``
(also 3D fiducials, but transforms them by a registered rigid body's own reported
rotation and rotates in 90-degree steps until the counter reads non-zero). That method
trusts the registration outright; measured on the FusionTrack, trusting it barely ever
matches, and even when it does, a genuinely 4-fold-symmetric board (counter at zero)
would settle its rotation by guessing rather than by evidence. ``process_frame`` remains
what it always was -- the reader for ``ftk.process_ftk_recording``'s offline CSV
analysis, a different consumer with a registered marker it can simply trust -- and nothing
here calls it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from rocsync.board_detection import find_board
from rocsync.board_profiles import NO_CORNERS, BoardProfile
from rocsync.camera import CameraType
from rocsync.fiducial_decode import Decode, decode_board_points


@dataclass
class PlaneFit:
    origin: np.ndarray  # (3,) a point on the board plane
    basis: np.ndarray  # (2, 3) orthonormal in-plane axes
    source: str  # "pose" or "constellation"


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def plane_from_pose(position_xyz, quaternion_xyzw) -> PlaneFit:
    """The board plane from a tracked rigid body's own pose.

    Only the orientation's first two axes are used, as the plane's basis; the in-plane
    origin and handedness are settled later by ``find_board``. So this is insensitive to
    both the geometry-origin convention and to the planar mirror ambiguity that
    ``fiducial_decode.process_frame`` patches by swapping rotation columns when r22 < 0.

    Takes plain arrays rather than a ``geometry_msgs/Pose`` -- the boundary this module
    crosses is 3D points and a rotation, not a ROS message shape.
    """
    x, y, z, w = quaternion_xyzw
    rot = _quat_to_matrix(x, y, z, w)
    origin = np.asarray(position_xyz, dtype=float)
    return PlaneFit(origin=origin, basis=rot[:, :2].T.copy(), source="pose")


def plane_from_constellation(points: np.ndarray, tolerance_mm: float = 6.0) -> PlaneFit | None:
    """The board plane from four fiducials forming the corner square.

    Pairs are matched on the side and diagonal lengths rather than tried exhaustively:
    with N fiducials that is O(N^2) instead of O(N^4), and a volume can easily hold 40.
    """
    corner_side_mm = 240.0
    corner_diagonal_mm = corner_side_mm * np.sqrt(2)  # ~339.41

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 4:
        return None

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
            # A square's diagonals bisect each other; that is what distinguishes the real
            # constellation from two unrelated pairs that happen to be 339 mm apart.
            if np.linalg.norm(midpoints[a] - midpoints[b]) > tolerance_mm:
                continue
            corners = points[sorted(quad)]
            # Sides must be the expected length too, or it is a rectangle, not our square.
            centred = corners - corners.mean(axis=0)
            # Plane basis from the corner spread: the two dominant singular directions.
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            basis = vt[:2]
            flat = centred @ basis.T
            sides = np.sort(np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=2).ravel())
            # 4 zeros, then 4 sides, then 2 diagonals.
            if not np.all(np.abs(sides[4:8] - corner_side_mm) <= tolerance_mm * 2):
                continue
            return PlaneFit(origin=corners.mean(axis=0), basis=basis, source="constellation")
    return None


def project(points: np.ndarray, plane: PlaneFit, plane_tolerance_mm: float) -> np.ndarray:
    """In-plane 2D coordinates of the fiducials that actually lie on the board plane.

    The off-plane rejection is this route's advantage over the 2D-blob route: a
    homography has no depth, so any blob along a ray through the board maps inside it
    and can false-match. Here a confuser is simply not coplanar.
    """
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
) -> Decode:
    """Decode from 3D fiducials, given a board plane (or find one)."""
    points_3d = np.asarray(points_3d, dtype=float).reshape(-1, 3)
    if plane is None:
        plane = plane_from_constellation(points_3d, tolerance_mm)
    if plane is None:
        return Decode(reject=NO_CORNERS)

    flat = project(points_3d, plane, plane_tolerance_mm)
    if len(flat) < 4:
        return Decode(reject=NO_CORNERS)

    # The in-plane coordinates are metric millimetres but their origin and orientation
    # are arbitrary -- find_board settles both from the corner LEDs, working in the
    # rectified board's pixel scale, so all three routes share one corner search.
    board = profile.rectify()
    corners = find_board(flat * board.px_per_mm, board)
    if corners is None:
        return Decode(reject=NO_CORNERS)

    homography = cv2.getPerspectiveTransform(
        corners.astype(np.float32), board.transform_corners(CameraType.INFRARED)
    )
    mapped_px = cv2.perspectiveTransform(
        (flat * board.px_per_mm).reshape(-1, 1, 2).astype(np.float32), homography
    ).reshape(-1, 2)

    result = decode_board_points(mapped_px / board.px_per_mm, profile)
    result.plane_source = plane.source
    return result
