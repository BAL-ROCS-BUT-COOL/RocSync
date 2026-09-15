"""Board-drawing primitives shared by the annotator and the results viewer.

Both tools warp an image into rectified board space and draw the same LED layout on
top of it; this module holds the geometry and drawing calls neither tool owns more than
the other, so `annotate.py` (the GUI editor) and `inspect.py` (the results viewer) both
sit above it instead of one importing from the other.
"""

import cv2
import numpy as np

from rocsync.board_profiles import BOARD_V1

# Every board rectifies to the same pixel grid, so one radius serves both tools.
LED_RADIUS_PX = BOARD_V1.rectify().led_sample_radius

# Slack warped in around the board, so LEDs that a coarse fit pushes off the edge stay visible.
BOARD_MARGIN_PX = 10

COLOR_ON = (0, 0, 255)  # red  — LED is ON
COLOR_OFF = (255, 0, 0)  # blue — LED is OFF
COLOR_NOT_VIS = (128, 128, 128)  # gray — component hidden/undecodable

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.75


def draw_text(img, text, org, color, scale=FONT_SCALE, thickness=1):
    """Draw a label in the one global font. Hershey strokes look ragged without LINE_AA."""
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def fit_scale(text, max_w, scale=FONT_SCALE):
    """Largest font scale up to `scale` at which `text` still fits `max_w` pixels."""
    w = cv2.getTextSize(text, FONT, scale, 1)[0][0]
    return scale if w <= max_w else scale * max_w / w


def ring_led_positions(board, camera):
    """Return list of (x, y) tuples for ring LEDs in rectified board coords."""
    return [(int(x), int(y)) for x, y in board.ring_led_coords(camera)]


def counter_led_positions(board, camera):
    """Return list of (x, y) tuples for counter LEDs in rectified board coords."""
    coords = board.counter_led_coords[camera]
    return [(int(p[0]), int(p[1])) for p in coords]


def corner_led_positions(board, camera):
    """Return list of (x, y) tuples for corner LEDs in rectified board coords."""
    return [(int(p[0]), int(p[1])) for p in board.always_on_leds[camera]]


def counter_bbox(counter_pos):
    """Compute counter bounding box (x1, y1, x2, y2) with margin around LEDs."""
    cx = [p[0] for p in counter_pos]
    cy = [p[1] for p in counter_pos]
    return (
        min(cx) - LED_RADIUS_PX - 10,
        min(cy) - LED_RADIUS_PX - 10,
        max(cx) + LED_RADIUS_PX + 10,
        max(cy) + LED_RADIUS_PX + 10,
    )


def warp_board_view(image, H, board, margin=BOARD_MARGIN_PX):
    """Rectify `image` into a board grid padded by `margin` px on every side."""
    T = np.array([[1, 0, margin], [0, 1, margin], [0, 0, 1]], dtype=np.float64)
    side = board.board_size + 2 * margin
    return cv2.warpPerspective(image[:, :, 2], T @ np.asarray(H, dtype=np.float64), (side, side))


def board_from_view(view, board, margin=BOARD_MARGIN_PX):
    """The board itself, cropped back out of a padded view."""
    bs = board.board_size
    return view[margin : margin + bs, margin : margin + bs].copy()
