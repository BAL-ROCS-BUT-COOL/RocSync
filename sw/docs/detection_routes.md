# Board detection

## Routes

Every route maps one frame to a `rocsync.decode.Decode`. Its `reject` is `None` when the
board time was read; otherwise it names the reason.

| Route | Entry point | Input |
|---|---|---|
| Image | `vision.process_frame` | RGB or IR image |
| 2D points | `blobs.decode_camera` | 2D points, e.g. a tracker's blob centroids |
| 3D fiducials | `fiducials.decode_fiducials(..., plane=None)` | 3D points |
| 3D fiducials + pose | `fiducials.decode_fiducials(..., plane_from_pose(...))` | 3D points and the board's 6DoF pose |

The RGB and IR image routes differ only in how the board corners are found: via the ArUco
marker and corner LEDs in RGB, and via `board_detection.find_corners_layout` in IR, where
the marker is not visible. Both then warp the board upright and read each LED from the
image intensity at its expected position.

The 2D-point route has no marker to identify the board revision, so the caller supplies
it, as `--board-version` does for the IR image route. Filtering points by a tracker's
status flags is also left to the caller.

The 3D routes project the fiducials near the board plane into it and pass them to the
2D-point route. Points off the plane are dropped first, which a homography alone cannot
do. The plane comes from the pose, or else from the square of corner LEDs among the
fiducials. Its in-plane origin and orientation are left to the corner search.

Properties observed on FusionTrack recordings:

- The board's registered rigid geometry matched in about 4% of frames, against 100% for
  an ordinary tracked tool in the same volume. The board's roughly 20 further coplanar
  LEDs are the likely cause. The constellation search carries most decodes, and
  pose-derived decodes fall on the same timeline fit.

## Locating the board among 2D points

`board_detection.find_board` locates the four IR corner LEDs among arbitrary 2D points:
image blobs, a tracker's 2D centroids, or 3D fiducials projected onto a plane.

Properties observed on FusionTrack recordings:

- A tracking volume holds other retroreflective markers next to the board. A convex hull
  over all blobs spans both and still yields four "corners", so the failure is silent
  and only shows up as a counter that reads zero.
- The board's IR LEDs saturate but are small: 22-42 px in area, against 152-219 px for
  the marker spheres. OpenCV's `SimpleBlobDetector` defaults to `minArea=25` and drops
  most LEDs, so masks are split by connected components instead.
- The v2 board's 5th always-on LED lies on the convex hull, so hull-based corner
  detection drops a real corner even in clean frames.
- On synthetic masks with 20 scattered blobs and no board, about 6 in 100 are
  accepted as a board.
