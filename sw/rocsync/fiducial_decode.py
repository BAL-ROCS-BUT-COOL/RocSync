"""Read a board's counter and ring off 2D points, and decode 3D fiducials against one.

Split out of ``ftk.py``, which mixed this with FusionTrack CSV recording IO. That mixing
had a real cost: ``ftk.py`` imports matplotlib and tqdm at module scope, and pulls in
scikit-learn transitively through ``rocsync.timeline`` -- so anything that wanted these
readers, including a plotting-free caller such as a ROS node, paid for a plotting stack
it never uses. This module keeps only numpy and the board definitions at module scope;
the ``ax`` debug-plotting parameter still works, but matplotlib is imported lazily, only
when a caller actually passes one.

``read_leds`` decides lit/unlit by nearest-point distance against a tolerance derived
from the board's own ring pitch (``fiducial_tol_mm``), which is the natural precision a
tracker's own centroids carry -- unlike an image, there is no intensity to threshold.
``process_frame`` is the 3D-fiducial entry point: it transforms fiducials into the
board's local frame from a registered rigid body's own position and rotation, rotates
in 90-degree steps until the counter reads non-zero (the four corners are otherwise
interchangeable), and reads the counter and ring once oriented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rocsync.board_profiles import BoardProfile
from rocsync.camera import CameraType

if TYPE_CHECKING:
    from matplotlib.axes import Axes

FTK_PLANE_TOL_MM = 5.0  # out-of-plane slack for a fiducial to count as on the board


def read_leds(
    fiducials: list[tuple[float, float]],
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
    fiducials: list[tuple[float, float]],
    board: BoardProfile,
    ax: Axes | None = None,
) -> tuple[int, int] | None:
    """Ring reading of a board seen by the tracker: first and last lit LED, or None."""
    tol_mm = board.fiducial_tol_mm(CameraType.INFRARED)
    leds = read_leds(fiducials, board.ring_led_coords(CameraType.INFRARED), tol_mm, ax)
    return board.decode_ring(leds)


def read_counter(
    fiducials: list[tuple[float, float]],
    board: BoardProfile,
    ax: Axes | None = None,
) -> int:
    """Counter reading of a board seen by the tracker."""
    tol_mm = board.fiducial_tol_mm(CameraType.INFRARED)
    leds = read_leds(fiducials, board.counter_led_coords[CameraType.INFRARED], tol_mm, ax)
    return board.decode_counter(leds)


def process_frame(
    position: np.ndarray,
    rotation_matrix: np.ndarray,
    fiducials: list[dict],
    board: BoardProfile,
    ax: Axes | None = None,
) -> tuple[int, int] | None:
    # Transform fiducials into local coordinate system
    transformed_fiducials = []
    inv_rotation = np.linalg.inv(rotation_matrix)
    for fiducial in fiducials:
        fid_pos_world = np.array(
            [
                float(fiducial["x_position"]),
                float(fiducial["y_position"]),
                float(fiducial["z_position"]),
                1.0,
            ]
        )
        fid_pos_marker = inv_rotation @ (fid_pos_world - position)
        fid_pos_marker[:2] += np.array([5, 5])  # Adjust for PCB origin

        # Filter fiducials within the PCB area
        if (
            abs(fid_pos_marker[2]) < FTK_PLANE_TOL_MM
            and 0 < fid_pos_marker[0] < board.size_mm
            and 0 < fid_pos_marker[1] < board.size_mm
        ):
            transformed_fiducials.append(fid_pos_marker[:2])

    # Rotate until counter is readable
    # TODO: not required for Rev2
    counter = 0
    for _ in range(4):
        counter = read_counter(transformed_fiducials, board, ax)
        if counter > 0:
            break

        # Rotate 90 degrees arround center
        rot90 = np.array([[0, -1], [1, 0]])
        rotated_fiducials = []
        center = np.array([board.centre_mm, board.centre_mm])
        for f in transformed_fiducials:
            v = f - center
            rotated = rot90 @ v + center
            rotated_fiducials.append(rotated)
        transformed_fiducials = rotated_fiducials

    if ax is not None:
        for fid in transformed_fiducials:
            ax.scatter(fid[0], fid[1], color="green")

    if counter == 0:
        return None

    ring = read_ring(transformed_fiducials, board, ax)
    if ring is None:
        return None
    return board.board_time_from_ring(counter, ring)
