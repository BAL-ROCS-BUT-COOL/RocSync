import math
import time
from enum import Enum

import cv2
import numpy as np


def _record_step(stats, name, t0, **kwargs):
    """Record a pipeline step's timing and metadata into the stats dict."""
    if stats is not None:
        stats["steps"][name] = {"time_ms": (time.perf_counter() - t0) * 1000, **kwargs}


def _finalize_stats(stats, total_start, success, timestamp):
    """Finalize the stats dict with total timing and result."""
    if stats is not None:
        stats["total_time_ms"] = (time.perf_counter() - total_start) * 1000
        stats["success"] = success
        stats["timestamp"] = list(timestamp) if timestamp else None


class CameraType(Enum):
    RGB = "rgb"
    INFRARED = "ir"


# Blob detector params
params = cv2.SimpleBlobDetector_Params()

# Detect white blobs
params.filterByColor = True
params.blobColor = 255

# Exclude elongated blobs caused by motion blur
params.filterByInertia = False
params.minInertiaRatio = 0.8

blob_detector = cv2.SimpleBlobDetector_create(params)

# ArUco detector params
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
aruco_detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Board layout
frquency = 1000
aruco_marker_id = 0
board_size = 640
period = 100
led_size = 8
ir_radius = 280
visible_radius = 294

ir_corners = np.array(
    [
        [13, 13],
        [board_size - 14, 13],
        [board_size - 14, board_size - 13],
        [13, board_size - 14],
    ],
    dtype=np.float32,
)

corner_dots = np.array(
    [
        [51, 51],
        [board_size - 52, 51],
        [board_size - 52, board_size - 52],
        [51, board_size - 52],
    ],
    dtype=np.float32,
)

aruco_corners_coords = np.array(
    [
        [202, 202],
        [board_size - 203, 202],
        [board_size - 203, board_size - 203],
        [202, board_size - 203],
    ],
    dtype=np.float32,
)


def draw_polygon(points, image, color):
    for i in range(len(points)):
        cv2.line(
            image,
            tuple(map(int, points[i])),
            tuple(map(int, points[(i + 1) % len(points)])),
            color,
            2,
        )


def read_led(img, x, y):
    led_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(led_mask, (x, y), led_size, (255), -1)
    led_intensity = np.quantile(img[led_mask > 0], 0.75)
    return led_intensity


def find_optimal_ring_start_end(scores):
    """
        Find the most likely window of leds turned ON, while allowing for false detections.
        Computes the window [start, end) that maximizes the sum of values within the window minus the sum of values outside,
        while allowing for wrap-around windows.

        Args:
            scores: A list of numeric values where positive means likely ON and negative means likely OFF.

        Returns:
            A tuple containing:
            - A tuple of [start, end) indices for the optimal window.
              If start > end, the window wraps around.
              If start == end, the window is empty.
            - The maximum possible score.
        """
    nums = list(scores)
    n = len(nums)
    total_sum = sum(nums)

    # --- Find max non-wrapping subarray and its indices (Kadane's Algorithm) ---
    max_score = -float('inf')
    max_start, max_end = 0, 0
    current_max = 0
    current_start_max = 0
    for i, x in enumerate(nums):
        if current_max <= 0:
            current_start_max = i
            current_max = x
        else:
            current_max += x

        if current_max > max_score:
            max_score = current_max
            max_start = current_start_max
            max_end = i + 1

    # --- Find min non-wrapping subarray and its indices ---
    min_score = float('inf')
    min_start, min_end = 0, 0
    current_min = 0
    current_start_min = 0
    for i, x in enumerate(nums):
        if current_min >= 0:
            current_start_min = i
            current_min = x
        else:
            current_min += x

        if current_min < min_score:
            min_score = current_min
            min_start = current_start_min
            min_end = i + 1

    # --- Determine the optimal window and score, now considering the empty set ---
    max_wrap_sum = -float('inf')
    if n > 1:  # A wrapping window needs at least 2 elements
        max_wrap_sum = total_sum - min_score

    # The three candidates for the best window sum are:
    # 1. The best non-wrapping sum (max_kadane)
    # 2. The best wrapping sum (max_wrap_sum)
    # 3. The empty set sum (0)

    if max_score > max_wrap_sum and max_score > 0:
        max_window_sum = max_score
        final_window = (max_start, max_end)
    elif max_wrap_sum > 0:
        max_window_sum = max_wrap_sum
        final_window = (min_end, min_start)
    else:
        # If both wrapping and non-wrapping sums are negative or zero,
        # the empty window (sum=0) is the best choice.
        max_window_sum = 0
        final_window = (0, 0)

    final_score = 2 * max_window_sum - total_sum

    return final_window, final_score


def read_ring(extracted_board, camera_type, draw_result=False, return_leds=False):
    radius = visible_radius if camera_type == CameraType.RGB else ir_radius

    # Collect mean LED intensities relative to local background
    led_intensities = np.zeros(period, dtype=np.float64)
    for i in range(period):
        angle = -(i / period + 0.25) * 2 * math.pi

        x = int(board_size / 2 + radius * math.cos(angle))
        y = int(board_size / 2 + radius * math.sin(angle))
        led_intensity = read_led(extracted_board, x, y)

        x_bg = int(board_size / 2 + (radius - 25) * math.cos(angle))
        y_bg = int(board_size / 2 + (radius - 25) * math.sin(angle))
        bg_intensity = read_led(extracted_board, x_bg, y_bg)

        led_intensities[i] = max(led_intensity - bg_intensity, 0.0)

    # Center around median so ON LEDs are positive, OFF LEDs are negative
    scores = led_intensities - np.median(led_intensities)

    # Find most likely segment of enabled LEDs, allowing for false detections
    (start, end), score = find_optimal_ring_start_end(scores)

    # Derive binary LED states from the arc boundaries
    leds = [False] * period
    if start != end:
        if start < end:
            for i in range(start, end):
                leds[i] = True
        else:  # wraps around
            for i in range(start, period):
                leds[i] = True
            for i in range(0, end):
                leds[i] = True

    if start == end:
        # no segment found
        if return_leds:
            return None, leds
        return None

    if draw_result:
        for i in range(period):
            angle = -(i / period + 0.25) * 2 * math.pi
            x = int(board_size / 2 + radius * math.cos(angle))
            y = int(board_size / 2 + radius * math.sin(angle))
            color = (0, 0, 255) if leds[i] else (255, 0, 0)
            cv2.circle(extracted_board, (x, y), led_size, color, 1)
    # return inclusive bounds (i.e. start is the first led ON, end -1 is the last led ON)
    result = (start, (end - 1) % period)
    if return_leds:
        return result, leds
    return result


def read_counter(extracted_board, camera_type, draw_result=False, return_leds=False):
    y = int(53 / 250 * 640) if camera_type == CameraType.RGB else int(48 / 250 * 640)

    # Collect mean LED intensities relative to local background
    led_intensities = np.zeros(16, dtype=np.uint8)
    for i in range(0, 16):
        x = int((65 + i * 8) / 250 * 640)
        led_intensity = read_led(extracted_board, x, y)
        bg_intensity = read_led(extracted_board, x, y - 25)
        led_intensities[i] = np.clip(led_intensity - bg_intensity, 0, 255)

    # Apply Otsu's thresholding to led_intensities
    _, otsu_thresh = cv2.threshold(led_intensities, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    leds = otsu_thresh.astype(bool)

    counter = 0
    for i in range(16):
        if leds[i]:
            counter += 2 ** (15 - i)
        if draw_result:
            x = int((65 + i * 8) / 250 * 640)
            color = (0, 0, 255) if leds[i] else (255, 0, 0)
            cv2.circle(extracted_board, (x, y), led_size, color, 1)
    if return_leds:
        return counter, leds.flatten()
    return counter


def find_corners_convexhull(mask, frame_number, debug_dir=None):
    points = blob_detector.detect(mask)

    # Draw detected blobs as red circles
    debug_image = cv2.drawKeypoints(
        mask.copy(),
        points,
        np.array([]),
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    points = [kp.pt for kp in points]

    # Find the convex hull and identify the corners
    if len(points) >= 4:
        hull = cv2.convexHull(np.array(points, dtype=np.float32))
        corners = hull.reshape(-1, 2)
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

            # Draw the approxcuimated convex hull
            corners = approx_hull.reshape(-1, 2)
            draw_polygon(corners, debug_image, (255, 0, 0))

    if debug_dir:
        cv2.imwrite(f"{debug_dir}/convexhull_{frame_number}.png", debug_image)
    if corners is not None and len(corners) == 4:
        return corners


def find_corners_dots(mask, frame_number, debug_dir=None):
    """Find the 4 corner LEDs in the rough-rectified image.

    Returns a list of 4 elements in the same order as corner_dots.
    Each element is a (x, y) tuple for matched corners, or None if unmatched.
    """
    points = blob_detector.detect(mask)
    if not points:
        return [None, None, None, None]
    if debug_dir:
        debug_image = cv2.drawKeypoints(
            mask.copy(),
            points,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv2.imwrite(f"{debug_dir}/corner_{frame_number}.png", debug_image)

    closest_points = [
        min(points, key=lambda p: np.linalg.norm(p.pt - target)).pt for target in corner_dots
    ]
    distances = [np.linalg.norm(act - exp) for act, exp in zip(closest_points, corner_dots)]

    return [pt if dist <= 50 else None for pt, dist in zip(closest_points, distances)]


def find_corners_aruco(mask, frame_number, debug_dir=None, brightness_boost=None):
    if brightness_boost is not None:
        mask = np.clip(mask * brightness_boost, 0, 255).astype(np.uint8)

    markers, marker_ids, _ = aruco_detector.detectMarkers(mask)
    if debug_dir:
        debug_image = mask.copy()
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(debug_image, markers, marker_ids)
        cv2.imwrite(f"{debug_dir}/aruco_{frame_number}.png", debug_image)

    if marker_ids is None:
        return None, None
    marker_dict = {id[0]: marker for id, marker in zip(marker_ids, markers)}
    if aruco_marker_id in marker_dict.keys():
        return marker_dict[aruco_marker_id], int(aruco_marker_id)
    return None, None


def process_frame(image, camera_type, frame_number, debug_dir=None, brightness_boost=None, try_hard=False, stats=None):
    total_start = time.perf_counter()
    if stats is not None:
        stats["steps"] = {}

    clahe = cv2.createCLAHE(tileGridSize=(8, 8))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    match camera_type:
        case CameraType.RGB:
            # First extract course PCB using ArUco marker
            t0 = time.perf_counter()
            gray_img_clahe = clahe.apply(gray_image)
            aruco_corners, aruco_id = find_corners_aruco(gray_img_clahe, frame_number, debug_dir, brightness_boost)
            _record_step(stats, "aruco_detection", t0, success=aruco_corners is not None, count=1 if aruco_corners is not None else 0)

            if stats is not None:
                stats["aruco_id"] = aruco_id
                stats["aruco_corners"] = aruco_corners.tolist() if aruco_corners is not None else None

            if aruco_corners is None:
                if stats is not None:
                    stats["rectified"] = None
                    stats["corner_positions"] = None
                _finalize_stats(stats, total_start, False, None)
                return False, None

            mask = image[:, :, 2]

            # Use course PCB to accurately extract corner
            rough_transformation_matrix = cv2.getPerspectiveTransform(
                aruco_corners, aruco_corners_coords
            )
            rough_pcb = cv2.warpPerspective(
                mask, rough_transformation_matrix, (board_size, board_size)
            )

            t0 = time.perf_counter()
            corners = find_corners_dots(rough_pcb, frame_number, debug_dir)
            matched = [(pt, exp) for pt, exp in zip(corners, corner_dots) if pt is not None]
            corner_count = len(matched)

            if stats is not None:
                stats["corner_positions"] = [list(pt) if pt is not None else None for pt in corners]

            if try_hard:
                if corner_count == 0:
                    # 0 corner LEDs found — use coarse ArUco homography as-is
                    transformation_matrix = rough_transformation_matrix
                else:
                    detected_pts = np.array([pt for pt, _ in matched], dtype=np.float32)
                    expected_pts = np.array([exp for _, exp in matched], dtype=np.float32)

                    # Back-project detected corner LEDs from rough-rectified → original image space
                    inv_rough = np.linalg.inv(rough_transformation_matrix)
                    src_corners = cv2.perspectiveTransform(
                        detected_pts.reshape(-1, 1, 2).astype(np.float64), inv_rough
                    ).reshape(-1, 2)

                    # Pool with ArUco correspondences
                    src_all = np.vstack([aruco_corners.reshape(-1, 2), src_corners]).astype(np.float64)
                    dst_all = np.vstack([aruco_corners_coords, expected_pts]).astype(np.float64)

                    transformation_matrix, _ = cv2.findHomography(src_all, dst_all)
                _record_step(stats, "corner_detection", t0, success=corner_count > 0, count=corner_count)
            else:
                _record_step(stats, "corner_detection", t0, success=corner_count == 4, count=corner_count)
                if corner_count < 4:
                    if stats is not None:
                        stats["rectified"] = None
                    _finalize_stats(stats, total_start, True, None)
                    return True, None

                all_corners = np.array([pt for pt in corners], dtype=np.float32)
                # NOTE: only four points are needed/allowed to calculate the perspective transform.
                transformation_matrix = np.dot(
                    cv2.getPerspectiveTransform(all_corners, corner_dots),
                    rough_transformation_matrix,
                )

            t0 = time.perf_counter()
            pcb = cv2.warpPerspective(mask, transformation_matrix, (board_size, board_size))
            _record_step(stats, "fine_rectification", t0)

            if stats is not None:
                stats["rectified"] = pcb.copy()
                stats["homography"] = transformation_matrix

            if debug_dir:
                cv2.imwrite(f"{debug_dir}/rectified_pcb_{frame_number}.png", pcb)
        case CameraType.INFRARED:
            _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            corners = find_corners_convexhull(mask, frame_number, debug_dir)
            if corners is None:
                _finalize_stats(stats, total_start, False, None)
                return False, None
            transformation_matrix = cv2.getPerspectiveTransform(corners, ir_corners)
            pcb = cv2.warpPerspective(mask, transformation_matrix, (board_size, board_size))

            # Find correct rotation
            for _ in range(4):
                if read_counter(pcb, CameraType.INFRARED) == 0:
                    pcb = cv2.rotate(pcb, cv2.ROTATE_90_CLOCKWISE)
            if read_counter(pcb, CameraType.INFRARED) == 0:
                _finalize_stats(stats, total_start, True, None)
                return True, None  # Counter was actually 0, can't determine orientation

    if debug_dir:  # For RGB debug output
        pcb = cv2.cvtColor(pcb, cv2.COLOR_GRAY2BGR)

    t0 = time.perf_counter()
    if stats is not None:
        counter, counter_leds = read_counter(pcb, camera_type, draw_result=True, return_leds=True)
        stats["counter_leds"] = counter_leds
    else:
        counter = read_counter(pcb, camera_type, draw_result=True)
    _record_step(stats, "counter_reading", t0, value=int(counter))

    t0 = time.perf_counter()
    if stats is not None:
        ring, ring_leds = read_ring(pcb, camera_type, draw_result=True, return_leds=True)
        stats["ring_leds"] = ring_leds
    else:
        ring = read_ring(pcb, camera_type, draw_result=True)
    _record_step(stats, "ring_reading", t0, success=ring is not None)

    if debug_dir:
        cv2.imwrite(f"{debug_dir}/leds_{frame_number}.png", pcb)

    if ring is not None:
        start, end = ring
        if start > end or start <= 1 or period - end <= 1:
            _finalize_stats(stats, total_start, True, None)
            return True, None # Counter increment during exposure

        start += counter * period
        end += counter * period
        _finalize_stats(stats, total_start, True, (start, end))
        return True, (start, end)

    _finalize_stats(stats, total_start, True, None)
    return True, None
