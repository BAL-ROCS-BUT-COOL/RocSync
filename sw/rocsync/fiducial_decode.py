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

``decode_board_points`` is the entry point for callers that have already resolved the
board's orientation themselves -- typically via ``board_detection.find_board``, which
settles it from the LEDs it explains rather than by trial rotation, the way the 3D and
2D-blob routes both do. It takes points already in the board's own millimetre frame and
needs no rotation search, only a decode.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rocsync.board_profiles import COUNTER_ZERO, RING, BoardProfile
from rocsync.camera import CameraType

if TYPE_CHECKING:
    from matplotlib.axes import Axes

FTK_PLANE_TOL_MM = 5.0  # out-of-plane slack for a fiducial to count as on the board


@dataclass
class Decode:
    """One decode attempt from points already in the board's own millimetre frame.

    ``reject`` is None exactly when ``board_ms`` is a real reading. It is one of
    ``board_profiles.COUNTER_ZERO`` or ``.RING`` -- never ``NO_CORNERS``, which is the
    caller's concern: this function only runs once the board itself has been found.
    """

    reject: str | None = None
    counter: int = 0
    ring_start: int = 0
    ring_end: int = 0
    board_ms: float = 0.0
    # The lit arc's length -- a measured output, never an assumption. -1 on reject,
    # since 0 is itself a valid (if implausible) exposure.
    exposure_ms: float = -1.0
    # Set by the 3D-fiducial route only ("pose" or "constellation"); empty otherwise.
    plane_source: str = ""


def decode_board_points(points_mm, board: BoardProfile) -> Decode:
    """Decode a board's counter and ring from points already in its millimetre frame.

    Rejects a zero counter outright, matching ``process_frame``'s rule: it means either
    the count has not started or -- when the caller's orientation search left a
    4-fold-symmetric board's rotation ambiguous -- that the ring index would be
    meaningless anyway. ``board.board_time_from_ring`` covers the remaining case, an
    arc that sits across the counter's wrap: the counter incremented mid-exposure, so
    the reading is correct but no single board time follows from it.
    """
    points = [np.asarray(p, dtype=float) for p in points_mm]
    counter = read_counter(points, board)
    if counter == 0:
        return Decode(reject=COUNTER_ZERO)

    ring = read_ring(points, board)
    if ring is None:
        return Decode(reject=RING)

    result = board.board_time_from_ring(counter, ring)
    if result is None:
        return Decode(reject=RING)

    start, end = result
    return Decode(
        counter=counter,
        ring_start=start,
        ring_end=end,
        board_ms=float(start),
        exposure_ms=float(end - start),
    )


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
) -> tuple[int, int] | None:
    """Ring reading of a board seen by the tracker: first and last lit LED, or None."""
    tol_mm = board.fiducial_tol_mm(CameraType.INFRARED)
    leds = read_leds(fiducials, board.ring_led_coords(CameraType.INFRARED), tol_mm, ax)
    return board.decode_ring(leds)


def read_counter(
    fiducials: Sequence[np.ndarray | tuple[float, float]],
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
