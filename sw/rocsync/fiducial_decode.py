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

``decode_board_points`` is the entry point for callers that have already resolved the
board's orientation themselves -- typically via ``board_detection.find_board``, which
settles it from the LEDs it explains rather than by trial rotation. It takes points
already in the board's own millimetre frame and needs no rotation search, only a decode.
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


def decode_board_points(
    points_mm,
    board: BoardProfile,
    camera_type: CameraType = CameraType.INFRARED,
    ax: Axes | None = None,
) -> Decode:
    """Decode a board's counter and ring from points already in its millimetre frame.

    ``camera_type`` selects which of the board's two physically distinct LED layouts
    ``points_mm`` were resolved against (see ``board_profiles``); defaults to IR since
    every current caller (tracker centroids, 3D fiducials) is IR-only.

    Rejects a zero counter outright: it means either the count has not started or --
    when the caller's orientation search left a 4-fold-symmetric board's rotation
    ambiguous -- that the ring index would be meaningless anyway.
    ``board.board_time_from_ring`` covers the remaining case, an arc that sits across
    the counter's wrap: the counter incremented mid-exposure, so the reading is correct
    but no single board time follows from it.
    """
    points = [np.asarray(p, dtype=float) for p in points_mm]
    counter = read_counter(points, board, ax, camera_type=camera_type)
    if counter == 0:
        return Decode(reject=COUNTER_ZERO)

    ring = read_ring(points, board, ax, camera_type=camera_type)
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
