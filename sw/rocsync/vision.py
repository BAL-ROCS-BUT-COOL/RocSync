import cv2
import numpy as np

from rocsync.board_profiles import (
    DEFAULT_BOARD_SIZE,
    PROFILES_BY_ARUCO,
    RING_BG_OFFSET_MM,
)
from rocsync.camera import CameraType
from rocsync.printer import print

MIN_ARUCO_AREA_FRACTION = 0.002  # smallest marker area, as a fraction of the frame

# Blob detector params
params = cv2.SimpleBlobDetector.Params()

# Detect white blobs
params.filterByColor = True
params.blobColor = 255

# Exclude elongated blobs caused by motion blur
params.filterByInertia = True
params.minInertiaRatio = 0.5

blob_detector = cv2.SimpleBlobDetector.create(params)

# ArUco detector params
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
aruco_detector = cv2.aruco.ArucoDetector(dictionary, parameters)


def draw_polygon(points, image, color):
    for i in range(len(points)):
        cv2.line(
            image,
            tuple(map(int, points[i])),
            tuple(map(int, points[(i + 1) % len(points)])),
            color,
            2,
        )


def read_led(img, x, y, radius):
    led_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(led_mask, (x, y), radius, (255), -1)
    led_intensity = np.quantile(img[led_mask > 0], 0.75)
    return led_intensity


def read_ring(extracted_board, camera_type, board, draw_on=None):
    """Ring reading of a rectified board: first and last lit LED, or None."""
    radius = board.led_sample_radius
    led_coords = board.ring_led_coords(camera_type).astype(int)
    bg_coords = board.ring_led_coords(camera_type, RING_BG_OFFSET_MM).astype(int)

    # Collect LED intensities relative to local background
    led_intensities = np.zeros(board.period, dtype=np.uint8)
    for i, ((x, y), (x_bg, y_bg)) in enumerate(zip(led_coords, bg_coords, strict=True)):
        led_intensity = read_led(extracted_board, x, y, radius)
        bg_intensity = read_led(extracted_board, x_bg, y_bg, radius)
        led_intensities[i] = np.clip(led_intensity - bg_intensity, 0, 255)

    # Apply Otsu's thresholding to led_intensities
    _, otsu_thresh = cv2.threshold(led_intensities, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    leds = otsu_thresh.astype(bool).flatten()

    if draw_on is not None:
        for state, (x, y) in zip(leds, led_coords, strict=True):
            color = (0, 0, 255) if state else (255, 0, 0)
            cv2.circle(draw_on, (x, y), radius, color, 1)

    return board.decode_ring(leds)


def read_counter(extracted_board, camera_type, board, draw_on=None):
    """Counter reading of a rectified board."""
    led_coords = board.counter_led_coords[camera_type].astype(int)
    bg_y = int(board.counter_bg_y[camera_type])
    radius = board.led_sample_radius

    # Collect LED intensities relative to local background
    led_intensities = np.zeros(led_coords.shape[0], dtype=np.uint8)
    for i, (x, y) in enumerate(led_coords):
        led_intensity = read_led(extracted_board, x, y, radius)
        bg_intensity = read_led(extracted_board, x, bg_y, radius)
        led_intensities[i] = np.clip(led_intensity - bg_intensity, 0, 255)

    # Apply Otsu's thresholding to led_intensities
    _, otsu_thresh = cv2.threshold(led_intensities, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    leds = otsu_thresh.astype(bool).squeeze()

    # draw optional debug output
    if draw_on is not None:
        for state, (x, y) in zip(leds, led_coords, strict=True):
            cv2.circle(
                draw_on,
                (x, y),
                radius,
                (0, 0, 255) if state else (255, 0, 0),
                1,
            )

    return board.decode_counter(leds)


def find_corners_convexhull(mask, frame_number, debug_dir=None):
    points = blob_detector.detect(mask)

    # Draw detected blobs as red circles
    debug_image = (
        cv2.drawKeypoints(
            mask,
            points,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        if debug_dir
        else None
    )

    points = [kp.pt for kp in points]

    # Find the convex hull and identify the corners
    corners = None
    if len(points) >= 4:
        hull = cv2.convexHull(np.array(points, dtype=np.float32))
        corners = hull.reshape(-1, 2)
        if debug_image is not None:
            draw_polygon(corners, debug_image, (0, 255, 0))

        if len(hull) > 4:
            # Approximate to 4 points
            epsilon_factor = 0.02
            n_points = len(hull)
            approx_hull = hull
            while n_points > 4:
                epsilon = epsilon_factor * cv2.arcLength(hull, True)
                approx_hull = cv2.approxPolyDP(hull, epsilon, True)
                n_points = len(approx_hull)
                epsilon_factor += 0.02

            # Draw the approximated convex hull
            corners = approx_hull.reshape(-1, 2)
            if debug_image is not None:
                draw_polygon(corners, debug_image, (255, 0, 0))

    if debug_image is not None:
        cv2.imwrite(f"{debug_dir}/convexhull_{frame_number}.png", debug_image)
    if corners is not None and len(corners) == 4:
        return corners


def find_corners_dots(mask, frame_number, board, debug_dir=None):
    corner_dots = board.always_on_leds[CameraType.RGB]
    points = blob_detector.detect(mask)
    if not points:
        return
    if debug_dir:
        debug_image = cv2.drawKeypoints(
            mask,
            points,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv2.imwrite(f"{debug_dir}/corner_{frame_number}.png", debug_image)

    closest_points = [
        min(points, key=lambda p: np.linalg.norm(p.pt - target)).pt for target in corner_dots
    ]
    max_distance = max(
        [np.linalg.norm(act - exp) for act, exp in zip(closest_points, corner_dots, strict=True)]
    )
    if max_distance > board.rough_corner_tol:
        return  # Some corner is too far away from where it should be

    return np.array(closest_points, dtype=np.float32)


def find_corners_aruco(mask, frame_number, debug_dir=None):
    markers, marker_ids, _ = aruco_detector.detectMarkers(mask)
    if debug_dir:
        debug_image = mask.copy()
        cv2.aruco.drawDetectedMarkers(debug_image, markers, marker_ids)
        cv2.imwrite(f"{debug_dir}/aruco_{frame_number}.png", debug_image)

    if marker_ids is None:
        return {}
    return {id.item(): marker for id, marker in zip(marker_ids, markers, strict=True)}


def rectify_board(
    image,
    camera_type,
    frame_number,
    board=None,
    debug_dir=None,
    board_size=DEFAULT_BOARD_SIZE,
):
    """Locate the board in a frame and warp it onto a square pixel grid.

    Returns (detected, pcb, board): whether the board was seen at all, the
    rectified single-channel image or None if it could not be squared up, and the
    RectifiedBoard the reading should be decoded against.
    """
    match camera_type:
        case CameraType.RGB:
            # Detect ArUco markers
            markers = find_corners_aruco(image, frame_number, debug_dir)
            if not markers:
                return False, None, None

            # Resolve board profile
            if board is None:
                for marker_id, corners in markers.items():
                    if marker_id in PROFILES_BY_ARUCO:
                        board = PROFILES_BY_ARUCO[marker_id].rectify(board_size)
                        aruco_corners = corners
                        break
                else:
                    return False, None, None
            else:
                board = board.rectify(board_size)
                if board.aruco_marker_id not in markers:
                    return False, None, board
                aruco_corners = markers[board.aruco_marker_id]

            # Check if aruco marker fills x % of the image to make sure the PCB was held close enough
            area = 0
            for i in range(4):
                x1, y1 = aruco_corners[0][i]
                x2, y2 = aruco_corners[0][(i + 1) % 4]  # Wrap around to the first point
                area += (x1 * y2) - (y1 * x2)
            area = abs(area) / 2
            height, width = image.shape[:2]
            image_area = width * height
            area_percentage = area / image_area
            if area_percentage < MIN_ARUCO_AREA_FRACTION:
                print(
                    f"Rejected {frame_number}: aruco marker only fills {area_percentage:.2%} of the image"
                )
                return False, None, board

            red_channel = image[:, :, 2]
            mask = red_channel

            # Use coarse PCB to accurately extract corners
            rough_transformation_matrix = cv2.getPerspectiveTransform(
                aruco_corners, board.aruco_corners_coords
            )
            rough_pcb = cv2.warpPerspective(
                mask, rough_transformation_matrix, (board_size, board_size)
            )
            corners = find_corners_dots(rough_pcb, frame_number, board, debug_dir)
            if corners is None:
                return True, None, board

            # Only the four anchors define the transform; any extra always-on dots were
            # matched purely as a sanity check.
            transformation_matrix = np.dot(
                cv2.getPerspectiveTransform(corners[:4], board.transform_corners(CameraType.RGB)),
                rough_transformation_matrix,
            )
            pcb = cv2.warpPerspective(mask, transformation_matrix, (board_size, board_size))

        case CameraType.INFRARED:
            if board is None:
                raise ValueError("IR mode requires an explicit board version (--board-version)")
            board = board.rectify(board_size)

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            corners = find_corners_convexhull(mask, frame_number, debug_dir)
            if corners is None:
                return False, None, board
            transformation_matrix = cv2.getPerspectiveTransform(
                corners, board.transform_corners(CameraType.INFRARED)
            )
            pcb = cv2.warpPerspective(mask, transformation_matrix, (board_size, board_size))

            # Find correct rotation
            for _ in range(4):
                if read_counter(pcb, CameraType.INFRARED, board) == 0:
                    pcb = cv2.rotate(pcb, cv2.ROTATE_90_CLOCKWISE)
            if read_counter(pcb, CameraType.INFRARED, board) == 0:
                return True, None, board  # Counter was 0, orientation undeterminable

        case _:
            raise ValueError(f"Unsupported camera type: {camera_type!r}")

    return True, pcb, board


def process_frame(
    image,
    camera_type,
    frame_number,
    board=None,
    debug_dir=None,
    board_size=DEFAULT_BOARD_SIZE,
):
    """Board time (start_ms, end_ms) read off one frame, and whether a board was seen."""
    detected, pcb, board = rectify_board(
        image, camera_type, frame_number, board, debug_dir, board_size
    )
    if pcb is None or board is None:
        return detected, None

    # Sample the pristine board; overlays go onto a separate canvas
    debug_canvas = cv2.cvtColor(pcb, cv2.COLOR_GRAY2BGR) if debug_dir else None

    counter = read_counter(pcb, camera_type, board, draw_on=debug_canvas)
    ring = read_ring(pcb, camera_type, board, draw_on=debug_canvas)

    if debug_canvas is not None:
        cv2.imwrite(f"{debug_dir}/leds_{frame_number}.png", debug_canvas)

    if ring is None:
        return True, None
    return True, board.board_time_from_ring(counter, ring)
