"""Decode a RocSync board from a tracker's own 2D centroids -- the least work of the
three routes, since the tracker has already done detection.

``board_detection.find_board`` was built to locate a board among arbitrary 2D points,
which is exactly what a tracker's per-camera blob list already is: no threshold, no
mask, just centroids. So this module is the thinnest of the three -- a call into the
shared corner search, then the shared point decoder.

Two things make this route weaker than the 3D-fiducial one, and neither is fixable
here, only worth knowing:

* **No depth test.** A homography has no third dimension, so any point lying anywhere
  along a ray through the board plane maps *inside* it and can false-match a modelled
  LED position. The fiducial route rejects those by coplanarity; here the only defence
  is the corner search's own scoring (an unrelated point still has to land on enough of
  the right modelled positions, and unexplained ones inside the outline count against
  the candidate).
* **No board identification.** The fiducial route can read a geometry id off a tracked
  rigid body; from bare centroids there is nothing to read an id from, so the board
  revision has to be supplied, exactly as ``rocsync.vision`` requires ``--board-version``
  for its own infrared path.

Filtering *which* centroids are worth feeding this at all -- dropping ones a tracker
marked invalid or merged -- depends on that tracker's own status-bit convention (a
`marker_tracking_msgs/Blob`'s ``status`` field, downstream of this package) and so stays
with the caller. What crosses this boundary is a plain ``(N, 2)`` array.
"""

from __future__ import annotations

import cv2
import numpy as np

from rocsync.board_detection import find_board
from rocsync.board_profiles import NO_CORNERS, BoardProfile
from rocsync.camera import CameraType
from rocsync.fiducial_decode import Decode, decode_board_points


def decode_camera(points: np.ndarray, profile: BoardProfile) -> Decode:
    """Locate and decode the board in one camera's 2D centroids.

    No per-call tolerance knob: ``find_board``'s layout tolerance is a property of the
    ``RectifiedBoard`` model it builds once and caches (keyed on the profile and pixel
    size alone), the same one the image and fiducial routes use unmodified. A call-time
    override would silently either miss that cache or corrupt it for later callers with
    the same profile, so this route accepts the board's own default (``LAYOUT_TOL_MM`` in
    ``board_profiles.py``) rather than reintroduce a knob the shared model has no room for.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 4:
        return Decode(reject=NO_CORNERS)

    board = profile.rectify()
    corners = find_board(points, board)
    if corners is None:
        return Decode(reject=NO_CORNERS)

    homography = cv2.getPerspectiveTransform(
        corners.astype(np.float32), board.transform_corners(CameraType.INFRARED)
    )
    mapped_px = cv2.perspectiveTransform(
        points.reshape(-1, 1, 2).astype(np.float32), homography
    ).reshape(-1, 2)

    return decode_board_points(mapped_px / board.px_per_mm, profile)
