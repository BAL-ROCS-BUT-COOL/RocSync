# Board detection

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
