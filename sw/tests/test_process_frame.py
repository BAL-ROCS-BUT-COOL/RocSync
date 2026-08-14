import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.normcase(Path(__file__).resolve().parents[1]))

import cv2 as cv
import numpy as np
from rocsync.board_profiles import BOARD_V2
from rocsync.vision import (
    CameraType,
    find_corners_aruco,
    find_corners_dots,
    process_frame,
    read_counter,
    read_ring,
)


def test_piecewise():
    """test the individual elements of the process_frame function with the new led layout"""

    board = BOARD_V2.rectify()
    board_size = board.board_size
    TEST_DIR = Path(__file__).resolve().parents[0]
    TEST_OUT_DIR = TEST_DIR / "output" / "piecewise"
    TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # prepare the pcb ----------------------------------------------------------
    image = cv.imread(TEST_DIR / "img1.jpg")

    markers = find_corners_aruco(
        image, frame_number=999, debug_dir=TEST_OUT_DIR
    )
    assert board.aruco_marker_id in markers, "ArUco marker was not detected"
    aruco_corners = markers[board.aruco_marker_id]
    red_channel = image[:, :, 2]
    rough_transformation_matrix = cv.getPerspectiveTransform(
        aruco_corners, board.aruco_corners_coords
    )
    rough_pcb = cv.warpPerspective(
        red_channel, rough_transformation_matrix, (board_size, board_size)
    )
    cv.imwrite(TEST_OUT_DIR / "rough_pcb.jpg", rough_pcb)

    # Matches every always-on dot, including the 5th one, as a sanity check
    corners = find_corners_dots(
        rough_pcb, 999, board, debug_dir=TEST_OUT_DIR
    )
    assert corners is not None, "always-on corner dots were not detected"

    # however, this only works with exactly 4 points
    transformation_matrix = np.dot(
        cv.getPerspectiveTransform(
            corners[:4], board.transform_corners(CameraType.RGB)
        ),
        rough_transformation_matrix,
    )
    pcb = cv.warpPerspective(
        red_channel, transformation_matrix, (board_size, board_size)
    )

    cv.imwrite(TEST_OUT_DIR / "pcb.jpg", pcb)

    leds_overlay = cv.cvtColor(pcb, cv.COLOR_GRAY2BGR)

    # decode the ring (should remain the same) ---------------------------------
    ring = read_ring(pcb, camera_type=CameraType.RGB, board=board, draw_on=leds_overlay)
    print(f"ring decoded: {ring}")

    # decode clock (must be adjusted to new layout) ----------------------------
    counter = read_counter(
        pcb, camera_type=CameraType.RGB, board=board, draw_on=leds_overlay
    )
    print(f"decoded clock: {counter} [0.1s]")

    cv.imwrite(TEST_OUT_DIR / "leds.jpg", leds_overlay)


def test_full():
    """test the full process_frame function for validation."""

    TEST_DIR = Path(__file__).resolve().parents[0]
    TEST_OUT_DIR = TEST_DIR / "output" / "full"
    TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # prepare the pcb ----------------------------------------------------------
    image = cv.imread(TEST_DIR / "img1.jpg")

    out = process_frame(
        image,
        camera_type=CameraType.RGB,
        frame_number=999,
        board=BOARD_V2,
        debug_dir=TEST_OUT_DIR,
    )
    print(f"output of process frame was: {out}")


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    test_piecewise()
    test_full()
