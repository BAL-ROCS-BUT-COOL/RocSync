"""Decode a RocSync board from 2D points, e.g. a tracker's per-camera blob centroids.

A homography has no depth, so a point anywhere along a ray through the board can match an
LED position.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from rocsync.board_detection import find_board
from rocsync.board_profiles import BoardProfile
from rocsync.camera import CameraType
from rocsync.decode import NO_BOARD, Decode
from rocsync.fiducial_decode import decode_board_points

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def decode_camera(points: np.ndarray, profile: BoardProfile, ax: Axes | None = None) -> Decode:
    """Locate and decode the board among 2D points of arbitrary scale and orientation."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 4:
        return Decode(reject=NO_BOARD)

    board = profile.rectify()
    corners = find_board(points, board)
    if corners is None:
        return Decode(reject=NO_BOARD)

    homography = cv2.getPerspectiveTransform(
        corners.astype(np.float32), board.transform_corners(CameraType.INFRARED)
    )
    mapped_px = cv2.perspectiveTransform(
        points.reshape(-1, 1, 2).astype(np.float32), homography
    ).reshape(-1, 2)

    return decode_board_points(mapped_px / board.px_per_mm, profile, ax=ax)
