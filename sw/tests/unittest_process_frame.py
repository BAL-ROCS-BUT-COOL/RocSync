import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.normcase(Path(__file__).resolve().parents[1]))

import cv2 as cv
import matplotlib.pyplot as plt
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

    # prepare the pcb ----------------------------------------------------------
    image = cv.imread(TEST_DIR / "img1.jpg")

    markers = find_corners_aruco(
        image, frame_number=999, debug_dir=TEST_DIR / "output_piecewise"
    )
    aruco_corners = markers[board.aruco_marker_id]
    red_channel = image[:, :, 2]
    rough_transformation_matrix = cv.getPerspectiveTransform(
        aruco_corners, board.aruco_corners_coords
    )
    rough_pcb = cv.warpPerspective(
        red_channel, rough_transformation_matrix, (board_size, board_size)
    )
    cv.imwrite(TEST_DIR / "output_piecewise" / "rough_pcb.jpg", rough_pcb)

    # Matches every always-on dot, including the 5th one, as a sanity check
    corners = find_corners_dots(
        rough_pcb, 999, board, debug_dir=TEST_DIR / "output_piecewise"
    )

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

    cv.imwrite(TEST_DIR / "output_piecewise" / "pcb.jpg", pcb)

    # decode the ring (should remain the same) ---------------------------------
    ring = read_ring(pcb, camera_type=CameraType.RGB, board=board, draw_result=True)
    print(f"ring decoded: {ring}")

    # decode clock (must be adjusted to new layout) ----------------------------
    counter = read_counter(
        pcb, camera_type=CameraType.RGB, board=board, draw_result=True
    )
    print(f"decoded clock: {counter} [0.1s]")

    # show the debug image output
    plt.imshow(pcb)
    plt.show()


def test_full():
    """test the full process_frame function for validation."""

    TEST_DIR = Path(__file__).resolve().parents[0]

    # prepare the pcb ----------------------------------------------------------
    image = cv.imread(TEST_DIR / "img1.jpg")

    out = process_frame(
        image,
        camera_type=CameraType.RGB,
        frame_number=999,
        board=BOARD_V2,
        debug_dir=TEST_DIR / "output_full",
    )
    print(f"output of process frame was: {out}")


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    test_piecewise()
    test_full()
