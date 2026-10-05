"""Locate the board's four IR corner LEDs among unrelated 2D points.

The points may include other bright objects, so the board is selected, not assumed to be
the only thing in view:

1. Candidate quads pair up segments whose midpoints coincide, since a square's diagonals
   bisect each other under modest perspective. This is O(n^2) rather than O(n^4).
2. Each candidate's homography maps the points onto the board's layout map, and the
   candidate is scored by which LED classes they land on. Points inside the outline that
   match no LED count against it.
3. All four 90-degree rotations are scored; the best wins and a tie is rejected.

Masks are split by connected components rather than SimpleBlobDetector, whose default
minimum area drops small IR LEDs.
"""

import math
from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np

from rocsync.board_profiles import LED_SAMPLE_RADIUS_MM
from rocsync.camera import CameraType

# Blob extraction
MIN_BLOB_AREA = 2  # px; drops single-pixel sensor noise
MAX_BLOBS = 160  # keep the largest when a frame thresholds into noise

# Candidate generation
MIN_SAMPLE_RADIUS_PX = 1.0  # source pixels per LED sample disc radius
MIN_DIAGONAL_RATIO = 0.35  # shorter/longer diagonal, ~65 degrees of tilt
MID_TOL_FRAC = 0.15  # midpoint offset, relative to the mean diagonal
MIN_CROSS_SIN = 0.15  # sine of the angle between the diagonals; keeps H stable
MAX_CANDIDATES = 512  # scored per frame, best-bisecting first
JOIN_BLOCK = 200_000  # segment pairs compared at once, bounds peak memory

# Acceptance
MIN_RING_LEDS = 1  # the exposure window always lights at least one
MAX_UNEXPLAINED = 0.30  # of the blobs inside the outline that the layout must explain

# Layout classes; ANCHOR is the candidate quad, EXEMPT may be bright without being an LED
ALWAYS_ON, RING, COUNTER, ANCHOR, EXEMPT = 1, 2, 3, 4, 5

_MODEL_CACHE = {}


def _board_model(board):
    """Layout map, expected always-on count and rotations for a board, built once.

    The layout map labels each pixel with the class of LED that belongs there. The ArUco
    marker footprint is exempt, since its white squares blob like LEDs in RGB images.
    """
    model = _MODEL_CACHE.get(board)
    if model is None:
        ir = CameraType.INFRARED
        size = board.board_size
        layout = np.zeros((size, size), dtype=np.uint8)
        cv2.fillConvexPoly(layout, np.rint(board.aruco_corners_coords).astype(np.int32), EXEMPT)
        # Coarsest class first, so finer labels win where tolerance discs overlap
        for label, coords in (
            (COUNTER, board.counter_led_coords[ir]),
            (RING, board.ring_led_coords(ir)),
            (ANCHOR, board.always_on_leds[ir][:4]),
            (ALWAYS_ON, board.always_on_leds[ir][4:]),
        ):
            for x, y in coords:
                cv2.circle(layout, (round(x), round(y)), board.layout_tol, label, -1)

        # rotations[r] maps corners[k] onto corners[(k + r) % 4]
        corners = board.transform_corners(ir).astype(np.float32)
        rotations = [
            cv2.getPerspectiveTransform(corners, np.roll(corners, -r, axis=0)) for r in range(4)
        ]
        n_always_on = len(board.always_on_leds[ir]) - 4
        model = (layout, n_always_on, rotations)
        _MODEL_CACHE[board] = model
    return model


def min_board_diagonal(board):
    """Shortest board diagonal, in source pixels, that can still be decoded.

    Rectifying cannot add detail. Once an LED's sample disc shrinks below a source
    pixel the warp is interpolating between neighbours that never resolved the LED,
    and the rectified image carries no information regardless of its size.
    """
    px_per_mm = MIN_SAMPLE_RADIUS_PX / LED_SAMPLE_RADIUS_MM
    return px_per_mm * board.profile.size_mm * math.sqrt(2)


def detect_blobs(mask):
    """Centroids of the lit blobs in a binary mask, as an (n, 2) float array."""
    _, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]  # label 0 is the background
    centroids = centroids[1:]

    keep = areas >= MIN_BLOB_AREA
    areas, centroids = areas[keep], centroids[keep]
    if len(centroids) > MAX_BLOBS:
        # Saturated LEDs outsize noise specks, so keep the largest blobs
        centroids = centroids[np.argsort(areas)[-MAX_BLOBS:]]
    return centroids


def _row_blocks(ahead, block):
    """Split look-ahead counts into row ranges of at most ``block`` pairs each."""
    total = int(ahead.sum())
    cuts = np.searchsorted(np.cumsum(ahead), np.arange(0, total, block), side="right")
    bounds = np.append(cuts, len(ahead))
    return [(a, b) for a, b in pairwise(bounds) if b > a]


def _candidate_quads(points, min_diagonal):
    """Quads whose diagonals bisect each other, as (m, 4) indices in convex order.

    Returned alongside the relative midpoint offset of each quad, which orders them
    from most to least square-consistent.
    """
    empty = (np.empty((0, 4), dtype=int), np.empty(0))
    i, j = np.triu_indices(len(points), k=1)
    delta = points[j] - points[i]
    length = np.hypot(delta[:, 0], delta[:, 1])

    keep = length >= min_diagonal
    i, j, length, delta = i[keep], j[keep], length[keep], delta[keep]
    if len(i) < 2:
        return empty
    mid = (points[i] + points[j]) / 2

    # Sweep by midpoint x so only segments within the widest tolerance are compared
    window = MID_TOL_FRAC * length.max()
    order = np.argsort(mid[:, 0], kind="stable")
    n_seg = len(order)
    ahead = np.searchsorted(mid[order, 0], mid[order, 0] + window, side="right")
    ahead = np.maximum(ahead - np.arange(1, n_seg + 1), 0)

    quads, offsets = [], []
    for row0, row1 in _row_blocks(ahead, JOIN_BLOCK):
        rows = np.arange(row0, row1)
        counts = ahead[rows]
        group = np.repeat(np.arange(len(rows)), counts)
        step = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
        a = order[rows[group]]
        b = order[rows[group] + 1 + step]

        gap = mid[a] - mid[b]
        offset = np.hypot(gap[:, 0], gap[:, 1])
        mean_length = (length[a] + length[b]) / 2
        ok = offset <= MID_TOL_FRAC * mean_length
        # Comparable diagonals that cross at an angle keep the homography stable
        ok &= np.minimum(length[a], length[b]) >= MIN_DIAGONAL_RATIO * np.maximum(
            length[a], length[b]
        )
        cross = delta[a, 0] * delta[b, 1] - delta[a, 1] * delta[b, 0]
        ok &= np.abs(cross) >= MIN_CROSS_SIN * length[a] * length[b]
        # Segments sharing an endpoint would collapse the quad onto a triangle.
        ok &= (i[a] != i[b]) & (i[a] != j[b]) & (j[a] != i[b]) & (j[a] != j[b])
        if not ok.any():
            continue

        a, b = a[ok], b[ok]
        # Endpoints of one diagonal go opposite each other, which is convex order.
        quads.append(np.stack([i[a], i[b], j[a], j[b]], axis=1))
        offsets.append(offset[ok] / mean_length[ok])

    if not quads:
        return empty
    quads = np.concatenate(quads)
    offsets = np.concatenate(offsets)

    best = np.argsort(offsets)[:MAX_CANDIDATES]
    return quads[best], offsets[best]


def _orient(quads, points):
    """Wind each quad like the IR anchors, i.e. clockwise in image coordinates."""
    corners = points[quads]
    x, y = corners[:, :, 0], corners[:, :, 1]
    area = np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    flipped = quads[:, [0, 3, 2, 1]]
    return np.where((area < 0)[:, None], flipped, quads)


def _project(points, homography):
    """Apply a homography to an (n, 2) array, dropping points sent to infinity."""
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        projected = projected @ homography.T
        projected = projected[:, :2] / projected[:, 2:3]
    return projected


def _score(points, homography, layout):
    """Points inside the board outline, counted by the layout class they land on.

    Index 0 holds the points that land on no class at all -- bright where the board has
    nothing to be bright. Points outside the outline are not counted either way; they
    are the rest of the scene, which the board says nothing about.
    """
    board = _project(points, homography)
    x = np.rint(board[:, 0])
    y = np.rint(board[:, 1])
    size = len(layout)
    inside = (x >= 0) & (x < size) & (y >= 0) & (y < size)  # false for NaN
    hits = layout[y[inside].astype(int), x[inside].astype(int)]
    return np.bincount(hits, minlength=EXEMPT + 1)


def _explained(score):
    """(explained, unexplained) blob counts, ignoring the exempt footprint."""
    return int(score[ALWAYS_ON:EXEMPT].sum()), int(score[0])


def _rank(score, n_always_on):
    """Ordering key for a scored orientation, best last.

    Showing every always-on LED outranks any amount of other agreement: those LEDs are
    lit in every frame, so a fit that cannot account for them is the wrong fit.
    """
    explained, unexplained = _explained(score)
    return (score[ALWAYS_ON] == n_always_on, explained - unexplained)


@dataclass
class BoardFit:
    """The best-scoring candidate quad found, whether or not it was accepted.

    Kept even when rejected so the debug renderer can show why it was rejected.
    """

    corners: np.ndarray  # (4, 2) in `points`' own coordinates, transform_corners order
    score: np.ndarray  # per-class hit counts; see _score
    n_always_on: int  # always-on LEDs the model expects, beyond the 4 anchors
    ambiguous: bool  # true when the runner-up orientation tied the winner
    accepted: bool


def _accept(score, n_always_on, ambiguous):
    explained, unexplained = _explained(score)
    return bool(
        score[ALWAYS_ON] == n_always_on  # every always-on LED accounted for
        and score[RING] >= MIN_RING_LEDS  # exposure window is on the board
        # Unexplained blobs inside the outline mean the outline is misplaced
        and unexplained <= MAX_UNEXPLAINED * max(1, explained + unexplained)
        and not ambiguous
    )


def _locate(points, board, min_diagonal):
    """The best candidate quad among ``points``, scored against ``board``, or None.

    ``None`` only when no candidate quad exists; a rejected quad is returned with
    ``accepted=False``.
    """
    layout, n_always_on, rotations = _board_model(board)
    ir_corners = board.transform_corners(CameraType.INFRARED)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    best_key, best_score, best_corners, ambiguous = None, None, None, False
    if len(points) >= 4:
        quads, _ = _candidate_quads(points, min_diagonal)
        quads = _orient(quads, points)

        for quad in quads:
            corners = points[quad].astype(np.float32)
            homography = cv2.getPerspectiveTransform(corners, ir_corners)
            # The corners land on the model by construction, so score the other points only
            others = np.delete(points, quad, axis=0)
            scores = [_score(others, rot @ homography, layout) for rot in rotations]
            keys = [_rank(score, n_always_on) for score in scores]
            rotation = max(range(4), key=keys.__getitem__)

            if best_key is None or keys[rotation] > best_key:
                best_key = keys[rotation]
                best_score = scores[rotation]
                best_corners = np.roll(corners, rotation, axis=0)
                # A tie leaves the orientation undetermined, e.g. v1 with a zero counter
                ambiguous = sorted(keys)[-2] == best_key

    if best_score is None:
        return None
    assert best_corners is not None  # set together with best_score, always
    return BoardFit(
        corners=best_corners,
        score=best_score,
        n_always_on=n_always_on,
        ambiguous=ambiguous,
        accepted=_accept(best_score, n_always_on, ambiguous),
    )


def find_board(points, board, min_diagonal=0.0):
    """The board's four corners among ``points``, in ``transform_corners`` order, or None.

    ``points`` may be image blobs or a tracker's 2D or projected-3D centroids.
    ``min_diagonal`` is the resolution floor from ``min_board_diagonal`` for points
    extracted from an image; leave it at 0 for tracker centroids.

    The corners are rotated into correspondence with the model, so the resulting warp
    is upright.
    """
    fit = _locate(points, board, min_diagonal)
    return fit.corners if fit is not None and fit.accepted else None


def find_corners_layout(mask, board, frame_number=None, debug_dir=None):
    """``find_board`` from a binary mask instead of pre-extracted points."""
    points = detect_blobs(mask)
    fit = _locate(points, board, min_board_diagonal(board))
    if debug_dir:
        _write_debug(mask, points, fit, board, frame_number, debug_dir)
    return fit.corners if fit is not None and fit.accepted else None


def _write_debug(mask, points, fit, board, frame_number, debug_dir):
    image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for x, y in points:
        cv2.circle(image, (round(x), round(y)), 6, (0, 0, 255), 1)

    if fit is not None:
        corners = fit.corners
        # Model LEDs pulled back into the image show how well the winner actually fits.
        inverse = np.linalg.inv(
            cv2.getPerspectiveTransform(corners, board.transform_corners(CameraType.INFRARED))
        )
        for x, y in _project(board.layout_coords(CameraType.INFRARED), inverse):
            if np.isfinite(x) and np.isfinite(y):
                cv2.drawMarker(image, (round(x), round(y)), (255, 128, 0), cv2.MARKER_CROSS, 4)

        colour = (0, 255, 0) if fit.accepted else (0, 165, 255)
        cv2.polylines(image, [np.rint(corners).astype(np.int32)], True, colour, 2)
        for label, (x, y) in enumerate(corners):
            cv2.putText(
                image,
                str(label),
                (round(x) + 8, round(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                1,
            )

    reason = (
        "" if fit is None or fit.accepted else " (ambiguous)" if fit.ambiguous else " (rejected)"
    )
    if fit is None:
        tally = "no quad"
    else:
        explained, unexplained = _explained(fit.score)
        tally = (
            f"on {fit.score[ALWAYS_ON]}/{fit.n_always_on}, ring {fit.score[RING]}, "
            f"counter {fit.score[COUNTER]}, unexplained {unexplained}/{explained + unexplained}"
        )
    cv2.putText(
        image,
        f"{len(points)} blobs, {tally}{reason}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    cv2.imwrite(f"{debug_dir}/board_{frame_number}.png", image)
