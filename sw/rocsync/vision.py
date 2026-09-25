import math
import time
from functools import cache

import cv2
import numpy as np

from rocsync.board_detection import find_corners_layout
from rocsync.board_profiles import (
    DEFAULT_BOARD_SIZE,
    PROFILES_BY_ARUCO,
    RING_BG_OFFSET_MM,
)
from rocsync.camera import CameraType
from rocsync.decode import NO_BOARD, NO_CORNERS, Decode, decode_reading
from rocsync.printer import print

MIN_ARUCO_AREA_FRACTION = 0.002  # smallest marker area, as a fraction of the frame

ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

ARUCO_PRIOR_MARGIN = 1.0  # region around a prior marker, in marker sizes per side
ARUCO_THRESHOLD_REACH = 32  # px beyond a tile edge the marker thresholding may look

# Keys `rocsync.benchmark` expects to find in a stats dict, whether or not the frame decoded.
_STATS_KEYS = (
    "aruco_id",
    "aruco_corners",
    "corner_positions",
    "rough_homography",
    "homography",
    "rectified",
    "counter_leds",
    "ring_leds",
    "ring_window",
    "timestamp",
    "reject",
)


def _init_stats(stats):
    """Give the benchmark a fully populated dict even when the frame fails early."""
    if stats is not None:
        stats.setdefault("steps", {})
        for key in _STATS_KEYS:
            stats.setdefault(key, None)


def _record_step(stats, name, t0, **fields):
    """Record one pipeline step's wall time and outcome."""
    if stats is not None:
        stats["steps"][name] = {"time_ms": (time.perf_counter() - t0) * 1000, **fields}


def _finalize_stats(stats, t0, decode):
    """Close out a stats dict with the total wall time and the frame's result."""
    if stats is not None:
        stats["total_time_ms"] = (time.perf_counter() - t0) * 1000
        stats["success"] = decode.board_seen
        stats["reject"] = decode.reject
        timestamp = decode.board_time
        # int(): the readers work in numpy scalars, which json cannot serialize
        stats["timestamp"] = [int(v) for v in timestamp] if timestamp else None


def _make_blob_detector():
    """Blob detector tuned for the board's lit LEDs."""
    params = cv2.SimpleBlobDetector.Params()

    # Detect white blobs
    params.filterByColor = True
    params.blobColor = 255

    # Exclude elongated blobs caused by motion blur
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    return cv2.SimpleBlobDetector.create(params)


def _make_aruco_detector():
    """ArUco detector for the board's identifying marker."""
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    return cv2.aruco.ArucoDetector(ARUCO_DICTIONARY, parameters)


blob_detector = _make_blob_detector()
aruco_detector = _make_aruco_detector()
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)


@cache
def _clahe(grid):
    """CLAHE over `grid` tiles, for a crop cut along the full frame's tile edges."""
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=grid)


def read_led(img, x, y, radius):
    """Intensity of one LED: the 0.75 quantile over a disc, robust to partial blur."""
    led_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(led_mask, (x, y), radius, (255), -1)
    led_intensity = np.quantile(img[led_mask > 0], 0.75)
    return led_intensity


@cache
def _disc_offsets(radius):
    """(dy, dx) of the pixels `cv2.circle` fills around an integer centre."""
    disc = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    cv2.circle(disc, (radius, radius), radius, (255), -1)
    dy, dx = np.nonzero(disc)
    return dy - radius, dx - radius


def read_leds(img, coords, radius):
    """`read_led` for every (x, y) row of `coords` at once."""
    coords = np.asarray(coords, dtype=int).reshape(-1, 2)
    dy, dx = _disc_offsets(radius)
    xs = coords[:, :1] + dx
    ys = coords[:, 1:] + dy
    height, width = img.shape[:2]
    inside = np.full(len(coords), img.ndim == 2)
    inside &= (xs.min(axis=1) >= 0) & (ys.min(axis=1) >= 0)
    inside &= (xs.max(axis=1) < width) & (ys.max(axis=1) < height)
    intensities = np.empty(len(coords))
    if inside.any():
        intensities[inside] = np.quantile(img[ys[inside], xs[inside]], 0.75, axis=1)
    # Discs the image edge clips, and multi-channel images, take the single-LED path
    for i in np.flatnonzero(~inside):
        intensities[i] = read_led(img, coords[i, 0], coords[i, 1], radius)
    return intensities


def read_ring(extracted_board, camera_type, board, draw_on=None, stats=None):
    """Ring reading of a rectified board: first and last lit LED, or None."""
    t_start = time.perf_counter()
    radius = board.led_sample_radius
    led_coords = board.ring_led_coords(camera_type).astype(int)
    bg_coords = board.ring_led_coords(camera_type, RING_BG_OFFSET_MM).astype(int)

    # Collect LED intensities relative to local background
    led_intensities = np.zeros(board.period, dtype=np.uint8)
    contrast = read_leds(extracted_board, led_coords, radius) - read_leds(
        extracted_board, bg_coords, radius
    )
    led_intensities[: len(contrast)] = np.clip(contrast, 0, 255)

    # Apply Otsu's thresholding to led_intensities
    _, otsu_thresh = cv2.threshold(led_intensities, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    leds = otsu_thresh.astype(bool).flatten()

    if draw_on is not None:
        for state, (x, y) in zip(leds, led_coords, strict=True):
            color = (0, 0, 255) if state else (255, 0, 0)
            cv2.circle(draw_on, (x, y), radius, color, 1)

    ring = board.decode_ring(leds)
    if stats is not None:
        stats["ring_leds"] = leds
        # The arc itself, so the benchmark can score reading it apart from the
        # timestamp: an arc that wraps the period end is read correctly and still
        # yields no time, because the counter changed while it was being exposed.
        stats["ring_window"] = tuple(int(v) for v in ring) if ring is not None else None
    _record_step(stats, "ring_reading", t_start, success=ring is not None)
    return ring


def read_counter(extracted_board, camera_type, board, draw_on=None, stats=None):
    """Counter reading of a rectified board."""
    t_start = time.perf_counter()
    led_coords = board.counter_led_coords[camera_type].astype(int)
    bg_y = int(board.counter_bg_y[camera_type])
    radius = board.led_sample_radius

    # Collect LED intensities relative to local background
    bg_coords = np.column_stack([led_coords[:, 0], np.full(len(led_coords), bg_y)])
    contrast = read_leds(extracted_board, led_coords, radius) - read_leds(
        extracted_board, bg_coords, radius
    )
    led_intensities = np.clip(contrast, 0, 255).astype(np.uint8)

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

    counter = board.decode_counter(leds)
    if stats is not None:
        stats["counter_leds"] = leds
    _record_step(stats, "counter_reading", t_start, value=counter)
    return counter


def find_corners_dots(mask, frame_number, board, debug_dir=None):
    """Match always-on LEDs to their expected spots.

    Returns one row per always-on LED of the board, in board order, holding the
    matched position in `mask` coordinates or NaN where nothing landed close enough.
    """
    corner_dots = board.always_on_leds[CameraType.RGB]
    points = blob_detector.detect(mask)
    if debug_dir:
        debug_image = cv2.drawKeypoints(
            mask,
            points,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv2.imwrite(f"{debug_dir}/corner_{frame_number}.png", debug_image)
    if not points:
        return np.full((len(corner_dots), 2), np.nan, dtype=np.float32)

    # For each target, find its closest available blob; commit the globally closest
    # target/blob pair first and remove that blob, so no blob is claimed by two targets.
    available = list(points)
    assigned = {}
    while available and len(assigned) < len(corner_dots):
        best = None
        for i, target in enumerate(corner_dots):
            if i in assigned:
                continue
            blob = min(available, key=lambda p: np.linalg.norm(p.pt - target))
            dist = np.linalg.norm(blob.pt - target)
            if best is None or dist < best[2]:
                best = (i, blob, dist)
        i, blob, dist = best
        assigned[i] = (blob.pt, dist)
        available.remove(blob)

    return np.array(
        [
            assigned[i][0]
            if i in assigned and assigned[i][1] <= board.rough_corner_tol
            else (np.nan, np.nan)
            for i in range(len(corner_dots))
        ],
        dtype=np.float32,
    )


def _aruco_in_region(image, prior, board_ids, recrop=True):
    """`find_corners_aruco` on the box around the `prior` marker corners, or {} if no marker
    with an id in `board_ids` is there.

    The box snaps outward to the full frame's CLAHE tiles, at least half a tile past the
    marker, so the region around it equalizes exactly as on the full frame. A marker found
    outside that zone is looked for again in a box around where it was found.
    """
    height, width = image.shape[:2]
    corners = prior.reshape(-1, 2)
    (x0, y0), (x1, y1) = corners.min(axis=0), corners.max(axis=0)
    pad = ARUCO_PRIOR_MARGIN * max(x1 - x0, y1 - y0)
    columns, rows = CLAHE_TILE_GRID_SIZE
    tile_w, tile_h = math.ceil(width / columns), math.ceil(height / rows)
    # Past half a tile no pixel near the marker interpolates towards a tile outside the box
    reach_x, reach_y = tile_w / 2 + ARUCO_THRESHOLD_REACH, tile_h / 2 + ARUCO_THRESHOLD_REACH
    pad_x, pad_y = max(pad, reach_x), max(pad, reach_y)
    left = max(0, int((x0 - pad_x) // tile_w) * tile_w)
    top = max(0, int((y0 - pad_y) // tile_h) * tile_h)
    right = min(width, math.ceil((x1 + pad_x) / tile_w) * tile_w)
    bottom = min(height, math.ceil((y1 + pad_y) / tile_h) * tile_h)
    grid = (math.ceil((right - left) / tile_w), math.ceil((bottom - top) / tile_h))
    crop = image[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    markers, marker_ids, _ = aruco_detector.detectMarkers(_clahe(grid).apply(gray))
    if marker_ids is None:
        return {}
    offset = np.array([left, top], dtype=np.float32)
    found = {i.item(): m + offset for i, m in zip(marker_ids, markers, strict=True)}
    marker = next((c for i, c in found.items() if i in board_ids), None)
    if marker is None:
        return {}
    if recrop:
        (u0, v0), (u1, v1) = marker.reshape(-1, 2).min(axis=0), marker.reshape(-1, 2).max(axis=0)
        exact = (
            (left == 0 or u0 - left >= reach_x)
            and (top == 0 or v0 - top >= reach_y)
            and (right == width or right - u1 >= reach_x)
            and (bottom == height or bottom - v1 >= reach_y)
        )
        if not exact:
            return _aruco_in_region(image, marker, board_ids, recrop=False)
    return found


def find_corners_aruco(mask, frame_number, debug_dir=None, prior=None, board_ids=None):
    """Locate the board's ArUco marker, equalizing the image first so it survives
    under- and over-exposure.

    With `prior` corners, only the region around them is searched first; the full frame
    is searched if no marker with an id in `board_ids` is found there.
    """
    if prior is not None and not debug_dir:
        found = _aruco_in_region(mask, prior, PROFILES_BY_ARUCO if board_ids is None else board_ids)
        if found:
            return found
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.ndim == 3 else mask
    normalized = clahe.apply(gray)
    markers, marker_ids, _ = aruco_detector.detectMarkers(normalized)
    if debug_dir:
        debug_image = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        cv2.aruco.drawDetectedMarkers(debug_image, markers, marker_ids)
        cv2.imwrite(f"{debug_dir}/aruco_{frame_number}.png", debug_image)

    if marker_ids is None:
        return {}
    return {id.item(): marker for id, marker in zip(marker_ids, markers, strict=True)}


def _warp_red(image, homography, board_size):
    """Red channel of `image` warped onto the board grid, without copying a strided channel."""
    warped = cv2.warpPerspective(image, homography, (board_size, board_size))
    return np.ascontiguousarray(warped[:, :, 2])


def rectify_board(
    image,
    camera_type,
    frame_number,
    board=None,
    debug_dir=None,
    board_size=DEFAULT_BOARD_SIZE,
    stats=None,
    min_aruco_area_fraction=MIN_ARUCO_AREA_FRACTION,
    try_hard=False,
):
    """Locate the board in a frame and warp it onto a square pixel grid.

    Returns (detected, pcb, board): whether the board was seen at all, the
    rectified single-channel image or None if it could not be squared up, and the
    RectifiedBoard the reading should be decoded against.

    `min_aruco_area_fraction` rejects frames where the board was held too far away; pass 0
    to read whatever the marker detector found, however small. `try_hard` forces it to 0.
    """
    return _rectify_board(
        image,
        camera_type,
        frame_number,
        board,
        debug_dir,
        board_size,
        stats,
        min_aruco_area_fraction,
        try_hard,
    )[:3]


def _rectify_board(
    image,
    camera_type,
    frame_number,
    board=None,
    debug_dir=None,
    board_size=DEFAULT_BOARD_SIZE,
    stats=None,
    min_aruco_area_fraction=MIN_ARUCO_AREA_FRACTION,
    try_hard=False,
    prior=None,
):
    """`rectify_board`, searching for the marker around `prior` corners first, and also
    returning the board's marker corners in the image, or None where it was not found."""
    _init_stats(stats)
    if try_hard:
        min_aruco_area_fraction = 0.0
    match camera_type:
        case CameraType.RGB:
            # Detect ArUco markers
            t0 = time.perf_counter()
            board_ids = None if board is None else (board.aruco_marker_id,)
            markers = find_corners_aruco(image, frame_number, debug_dir, prior, board_ids)
            _record_step(stats, "aruco_detection", t0, success=bool(markers), count=len(markers))
            if not markers:
                return False, None, None, None

            # Resolve board profile
            if board is None:
                for marker_id, corners in markers.items():
                    if marker_id in PROFILES_BY_ARUCO:
                        board = PROFILES_BY_ARUCO[marker_id].rectify(board_size)
                        aruco_corners = corners
                        break
                else:
                    return False, None, None, None
            else:
                board = board.rectify(board_size)
                if board.aruco_marker_id not in markers:
                    return False, None, board, None
                aruco_corners = markers[board.aruco_marker_id]

            if stats is not None:
                stats["aruco_id"] = board.aruco_marker_id
                stats["aruco_corners"] = aruco_corners.tolist()

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
            if stats is not None:
                stats["aruco_area_fraction"] = area_percentage
            if area_percentage < min_aruco_area_fraction:
                print(
                    f"Rejected {frame_number}: aruco marker only fills {area_percentage:.2%} of the image"
                )
                return False, None, board, aruco_corners

            # Use coarse PCB to accurately extract corners
            rough_transformation_matrix = cv2.getPerspectiveTransform(
                aruco_corners, board.aruco_corners_coords
            )
            rough_pcb = _warp_red(image, rough_transformation_matrix, board_size)
            if stats is not None:
                # Corners are detected in this grid; the benchmark needs it to get back to image space
                stats["rough_homography"] = rough_transformation_matrix
            t0 = time.perf_counter()
            rough_corners = find_corners_dots(rough_pcb, frame_number, board, debug_dir)
            found = np.isfinite(rough_corners).all(axis=1)
            _record_step(
                stats,
                "corner_detection",
                t0,
                success=bool(found.any()),
                count=int(found.sum()),
            )

            # Corners are matched in the rough-rectified grid; un-warp them back to image
            # space so everything this function stores is in one coordinate space.
            inv_rough = np.linalg.inv(rough_transformation_matrix)
            image_corners = np.full_like(rough_corners, np.nan)
            if found.any():
                image_corners[found] = cv2.perspectiveTransform(
                    rough_corners[found].reshape(1, -1, 2), inv_rough
                ).reshape(-1, 2)
            if stats is not None:
                stats["corner_positions"] = image_corners.tolist()

            # The first four always-on LEDs are the perspective-transform anchors; without
            # all four there's nothing to fit the fine homography from, unless try_hard falls
            # back to whichever LEDs (plus the ArUco corners) were found.
            if found[:4].all():
                transformation_matrix = cv2.getPerspectiveTransform(
                    image_corners[:4], board.transform_corners(CameraType.RGB)
                )
            elif not try_hard:
                return True, None, board, aruco_corners
            elif not found.any():
                # No always-on LED matched; fall back to the coarse ArUco homography
                transformation_matrix = rough_transformation_matrix
            else:
                # Pool every matched LED with the ArUco corners
                corner_dots = board.always_on_leds[CameraType.RGB]
                src = np.vstack([aruco_corners.reshape(-1, 2), image_corners[found]]).astype(
                    np.float64
                )
                dst = np.vstack([board.aruco_corners_coords, corner_dots[found]]).astype(np.float64)
                transformation_matrix, _ = cv2.findHomography(src, dst)
                if transformation_matrix is None:
                    return True, None, board, aruco_corners
            t0 = time.perf_counter()
            pcb = _warp_red(image, transformation_matrix, board_size)
            _record_step(stats, "fine_rectification", t0)
            if stats is not None:
                stats["homography"] = transformation_matrix

        case CameraType.INFRARED:
            if board is None:
                raise ValueError("IR mode requires an explicit board version (--board-version)")
            board = board.rectify(board_size)
            aruco_corners = None

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            t0 = time.perf_counter()
            corners = find_corners_layout(mask, board, frame_number, debug_dir)
            _record_step(stats, "corner_detection", t0, success=corners is not None)
            if corners is not None and stats is not None:
                stats["corner_positions"] = corners.tolist()
            if corners is None:
                return False, None, board, None
            # find_corners_layout settles the orientation, so this warp is upright
            transformation_matrix = cv2.getPerspectiveTransform(
                corners, board.transform_corners(CameraType.INFRARED)
            )
            t0 = time.perf_counter()
            pcb = cv2.warpPerspective(mask, transformation_matrix, (board_size, board_size))
            _record_step(stats, "fine_rectification", t0)
            if stats is not None:
                stats["homography"] = transformation_matrix

        case _:
            raise ValueError(f"Unsupported camera type: {camera_type!r}")

    if stats is not None:
        stats["rectified"] = pcb
    return True, pcb, board, aruco_corners


def process_frame(
    image,
    camera_type,
    frame_number,
    board=None,
    debug_dir=None,
    board_size=DEFAULT_BOARD_SIZE,
    stats=None,
    min_aruco_area_fraction=MIN_ARUCO_AREA_FRACTION,
    try_hard=False,
    prior=None,
):
    """Decode the board time off one image.

    `prior` is the board marker's corners from the previous frame, around which the
    marker is searched for first.
    """
    t_start = time.perf_counter()
    _init_stats(stats)
    detected, pcb, board, aruco_corners = _rectify_board(
        image,
        camera_type,
        frame_number,
        board,
        debug_dir,
        board_size,
        stats,
        min_aruco_area_fraction,
        try_hard,
        prior,
    )
    if pcb is None or board is None:
        decode = Decode(reject=NO_CORNERS if detected else NO_BOARD, aruco_corners=aruco_corners)
        _finalize_stats(stats, t_start, decode)
        return decode

    # Sample the pristine board; overlays go onto a separate canvas
    debug_canvas = cv2.cvtColor(pcb, cv2.COLOR_GRAY2BGR) if debug_dir else None

    counter = read_counter(pcb, camera_type, board, draw_on=debug_canvas, stats=stats)
    ring = (
        read_ring(pcb, camera_type, board, draw_on=debug_canvas, stats=stats) if counter else None
    )

    if debug_canvas is not None:
        cv2.imwrite(f"{debug_dir}/leds_{frame_number}.png", debug_canvas)

    decode = decode_reading(board, counter, ring)
    decode.aruco_corners = aruco_corners
    _finalize_stats(stats, t_start, decode)
    return decode
