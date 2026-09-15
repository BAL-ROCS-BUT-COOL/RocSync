"""Read a board's counter and ring off 2D points in its millimetre frame.

``read_leds`` marks an LED lit when a point lies within ``fiducial_tol_mm`` of it, since
tracker centroids carry no intensity to threshold. ``decode_board_points`` decodes points
whose orientation the caller has already resolved, e.g. via ``board_detection.find_board``.

matplotlib is imported only when a caller passes an ``ax``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from rocsync.board_profiles import BoardProfile
from rocsync.camera import CameraType
from rocsync.decode import Decode, decode_reading

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def decode_board_points(
    points_mm,
    board: BoardProfile,
    ax: Axes | None = None,
    camera_type: CameraType = CameraType.INFRARED,
) -> Decode:
    """Decode a board's counter and ring from points already in its millimetre frame."""
    points = [np.asarray(p, dtype=float) for p in points_mm]
    counter = read_counter(points, board, ax, camera_type)
    ring = read_ring(points, board, ax, camera_type) if counter else None
    return decode_reading(board, counter, ring)


def read_leds(
    fiducials: Sequence[np.ndarray | tuple[float, float]],
    led_coords: np.ndarray,
    tol_mm: float,
    ax: Axes | None = None,
) -> np.ndarray:
    """LED states for the given centres: lit where a fiducial sits within tol_mm."""
    leds = np.zeros(len(led_coords), dtype=bool)
    for i, led in enumerate(led_coords):
        leds[i] = any(np.linalg.norm(fiducial - led) < tol_mm for fiducial in fiducials)

        if ax is not None:
            from matplotlib.patches import Circle

            color = "red" if leds[i] else "blue"
            ax.add_patch(Circle(led, tol_mm, color=color, fill=False))

    return leds


def read_ring(
    fiducials: Sequence[np.ndarray | tuple[float, float]],
    board: BoardProfile,
    ax: Axes | None = None,
    camera_type: CameraType = CameraType.INFRARED,
) -> tuple[int, int] | None:
    """Ring reading of a board seen by the tracker: first and last lit LED, or None."""
    tol_mm = board.fiducial_tol_mm(camera_type)
    leds = read_leds(fiducials, board.ring_led_coords(camera_type), tol_mm, ax)
    return board.decode_ring(leds)


def read_counter(
    fiducials: Sequence[np.ndarray | tuple[float, float]],
    board: BoardProfile,
    ax: Axes | None = None,
    camera_type: CameraType = CameraType.INFRARED,
) -> int:
    """Counter reading of a board seen by the tracker."""
    tol_mm = board.fiducial_tol_mm(camera_type)
    leds = read_leds(fiducials, board.counter_led_coords[camera_type], tol_mm, ax)
    return board.decode_counter(leds)
