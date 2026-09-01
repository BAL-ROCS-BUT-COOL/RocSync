"""Render a board the way an IR camera sees it, then locate its corners.

The repo has an RGB fixture but no IR one, and the IR path is the half that has no
ArUco marker to lean on: it has to find the board among the blobs and settle its
orientation before anything downstream can decode a counter or a ring. Synthesising
the frame from the board profile exercises exactly that -- ``find_corners_layout`` --
against known corners, at viewing angles a fixture cannot cover.

Decoding the counter and ring off the located corners is not tested here. That is
``decode_points``'s job, ported separately once ``vision.py`` is wired to this module;
this file only has to answer "where is the board", not "what does it say".
"""

import cv2
import numpy as np
import pytest

from rocsync.board_detection import find_corners_layout
from rocsync.board_profiles import BOARD_V1, BOARD_V2
from rocsync.camera import CameraType

IMAGE_SIZE = 900
# read_led takes the 0.75 quantile of a disc of LED_SAMPLE_RADIUS_MM, so a rendered LED
# has to cover comfortably more than a quarter of that disc to read as lit. It also has
# to stay smaller than the ~19 px ring spacing, or neighbouring LEDs merge into one blob.
LED_RADIUS = 7


def render(profile, counter, ring_window, homography, board_size=640):
    """An IR-style binary mask: lit discs at every LED position, everywhere else 0."""
    board = profile.rectify(board_size)
    lit = [board.always_on_leds[CameraType.INFRARED]]

    bits = board.counter_led_coords[CameraType.INFRARED]
    n = board.counter_bits
    lit.append(
        np.array([p for i, p in enumerate(bits) if counter >> (n - 1 - i) & 1]).reshape(-1, 2)
    )

    start, end = ring_window
    ring = board.ring_led_coords(CameraType.INFRARED)
    lit.append(np.array([ring[i % board.period] for i in range(start, end + 1)]))

    points = np.concatenate(lit).astype(np.float32)
    warped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 2)

    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    for x, y in warped:
        cv2.circle(mask, (round(x), round(y)), LED_RADIUS, 255, -1)
    return mask


def view(board_size, corners):
    """Homography taking the rectified board square onto ``corners`` in the image."""
    square = np.array(
        [[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(square, np.array(corners, dtype=np.float32))


def _by_angle(points):
    """Points sorted around their centroid, so two windings of the same quad compare equal."""
    centre = points.mean(axis=0)
    order = np.argsort(np.arctan2(*(points - centre).T[::-1]))
    return points[order]


# Head-on, and two oblique views. The board is 4-fold symmetric apart from the 5th LED,
# so these also check that the search settles on the right orientation rather than a
# rotation of it.
VIEWS = {
    "head-on": [(100, 100), (800, 100), (800, 800), (100, 800)],
    "tilted": [(160, 120), (780, 200), (740, 790), (120, 700)],
    "rotated-90": [(800, 100), (800, 800), (100, 800), (100, 100)],
}


@pytest.mark.parametrize("profile", [BOARD_V1, BOARD_V2], ids=["v1", "v2"])
@pytest.mark.parametrize("view_name", list(VIEWS))
def test_find_corners_layout_locates_the_board(profile, view_name):
    board_size = 640
    counter = 12345 if profile.counter_bits >= 20 else 4321
    ring_window = (10, 30)
    homography = view(board_size, VIEWS[view_name])

    mask = render(profile, counter, ring_window, homography, board_size)
    found = find_corners_layout(mask, profile.rectify(board_size), frame_number=0)

    assert found is not None, f"{profile.name} {view_name}: board not located"
    # The corner LEDs sit 5 mm inset from the board edge, not on it -- so "true" is
    # where the anchor LEDs actually land once warped, not the frame corners in VIEWS.
    board = profile.rectify(board_size)
    anchors = board.transform_corners(CameraType.INFRARED)
    expected = cv2.perspectiveTransform(anchors.reshape(-1, 1, 2), homography).reshape(-1, 2)
    np.testing.assert_allclose(
        _by_angle(np.asarray(found, dtype=np.float64)), _by_angle(expected), atol=1.0
    )


def _noise_mask(rng, n_blobs):
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    for x, y in rng.integers(50, IMAGE_SIZE - 50, size=(n_blobs, 2)):
        cv2.circle(mask, (int(x), int(y)), LED_RADIUS, 255, -1)
    # Match vision.py's IR preprocessing: it closes the mask before detection, which
    # can merge nearby blobs.
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def test_find_corners_layout_rarely_accepts_a_frame_without_a_board():
    """Scattered blobs should rarely yield a confident corner fit.

    Not never: with acceptance keyed on which LEDs a candidate quad explains rather
    than a distance threshold, a false accept needs the scattered blobs to coincide
    with enough modelled positions by chance, which is possible but should stay rare
    even at a blob count well above what a real frame carries. The board's own
    profile has been rewritten (mm-based coordinates, different LED layout) since this
    module's false-accept rate was last measured on the WIP branch it came from, so
    that old figure (26/100 at 20 blobs) no longer applies -- this re-measures it
    against the current ``BOARD_V2`` rather than trust a stale number.
    """
    board = BOARD_V2.rectify(640)
    rng = np.random.default_rng(0)
    false_accepts = sum(
        find_corners_layout(_noise_mask(rng, 20), board, frame_number=i) is not None
        for i in range(100)
    )
    # Measured against the current BOARD_V2: 6/100. Bounded well above that rather than
    # pinned to it, so an unrelated, small scoring change does not make this test flaky.
    assert false_accepts <= 15, f"{false_accepts}/100 false accepts at 20 blobs"
