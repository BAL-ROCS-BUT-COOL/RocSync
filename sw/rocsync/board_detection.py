"""Locate the board's four IR corner LEDs among unrelated blobs.

``find_corners_convexhull`` takes the convex hull of *every* blob in the image, which
assumes the board is the only thing there. In a tracking volume it is not: measured on
the live FusionTrack, one frame held four retroreflective spheres of a tracked marker
at y~71-126 and the board at y~487-748. The hull spans both, and the resulting
homography is meaningless -- while still yielding four "corners", so the failure is
silent and shows up only as a counter that reads zero. The v2 board makes this worse:
its 5th IR LED sits on the hull, so even an otherwise clean frame gives a five-point
hull that ``approxPolyDP`` collapses by dropping a real corner.

The board also does not look like the spheres. Its LEDs saturate (peak 254) but are
*small*: 22-42 px in area against the spheres' 152-219. OpenCV's SimpleBlobDetector
defaults to ``minArea=25``, which upstream never overrides, so most LEDs are discarded
before any of this is reached. Since the input is already a binary mask, this module
takes connected components instead -- every lit LED, with its centroid, in one pass.

Selection, not thresholding, is the actual problem, and it is solved in three steps:

1. **Candidate quads from bisecting diagonals.** A square's diagonals bisect each
   other, and that survives modest perspective (the midpoints separate by 0.09 of a
   diagonal at 45 degrees of tilt, 0.14 at 75). Pairing up point-pairs by shared
   midpoint is O(n^2) rather than the O(n^4) of trying every 4-subset, which matters
   because a volume with several markers easily reaches 40 blobs.
2. **Scored by which LEDs land where that class of LED belongs.** Each candidate gives
   a homography; map every point through it and sort the hits into always-on, ring and
   counter positions. Two of those are evidence the board is really there: every
   always-on LED the profile declares must show up, and the ring must light somewhere.
   Both hold for any decodable frame, however little else is lit.

   Necessary is not sufficient, though, because the search maximises over hundreds of
   candidates: a coincidence rare per candidate is common across all of them. What
   settles it is the blobs a candidate *fails* to explain. A real board lights nothing
   inside its own outline that is not an LED, so a quad laid over unrelated blobs is
   betrayed by the ones falling in the gaps -- the layout accounts for under a tenth of
   the board's area, so a wrong outline leaves most of them stranded.
3. **Only the winner is decoded.** Scoring is a few hundred vectorised lookups into a
   precomputed layout mask; decoding is not, so it runs once.

Scoring also settles the board's orientation. The four corners are interchangeable
under 90-degree rotation, so all four are scored and the best wins -- for v2 the 5th
always-on LED alone decides it, for v1 the counter row does whenever any counter bit
is lit. A tie is rejected rather than guessed at.

The one thing a quad's own geometry has to answer is whether the board is resolved at
all. A rectified image is an interpolation of what the source held, so the floor is
set by the source: below roughly one pixel per LED sample disc there is nothing left
to read, however large the rectified grid is.
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

# Layout classes. ANCHOR marks the four corners, which are the candidate quad itself.
# EXEMPT marks board area that is allowed to be bright without being an LED, so a blob
# there counts neither for nor against a candidate.
ALWAYS_ON, RING, COUNTER, ANCHOR, EXEMPT = 1, 2, 3, 4, 5

_MODEL_CACHE = {}


def _board_model(board):
    """Layout map, expected always-on count and rotations for a board, built once.

    The layout map labels each pixel with the class of LED that belongs there, which is
    what lets a candidate be judged on *which* LEDs it explains rather than how many.

    The marker footprint is exempt. Nothing on an IR board lights there, but the RGB
    board carries the ArUco marker, whose white squares blob just like an LED does --
    so counting them as unexplained would reject exactly the frames that decode best.
    Exempting it costs the IR path nothing, and keeps the map honest for either camera.
    """
    model = _MODEL_CACHE.get(board)
    if model is None:
        ir = CameraType.INFRARED
        size = board.board_size
        layout = np.zeros((size, size), dtype=np.uint8)
        cv2.fillConvexPoly(layout, np.rint(board.aruco_corners_coords).astype(np.int32), EXEMPT)
        # Drawn coarsest class first, so an always-on LED close to the ring keeps its
        # own label wherever two tolerance discs would overlap.
        for label, coords in (
            (COUNTER, board.counter_led_coords[ir]),
            (RING, board.ring_led_coords(ir)),
            (ANCHOR, board.always_on_leds[ir][:4]),
            (ALWAYS_ON, board.always_on_leds[ir][4:]),
        ):
            for x, y in coords:
                cv2.circle(layout, (round(x), round(y)), board.layout_tol, label, -1)

        # rotations[r] maps corners[k] onto corners[(k + r) % 4], so composing it with
        # a candidate's homography re-labels which corner is the top-left one.
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
        # A frame without the board thresholds into noise specks; the LEDs saturate
        # and are the larger blobs, so this keeps them and bounds the pairing below.
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

    # Sorting the segments by midpoint x and only looking ahead while x stays within
    # the widest tolerance any pair could claim keeps this near-linear in the number
    # of segments, instead of comparing all of them against all of them.
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
        # Comparable diagonals, and diagonals that actually cross at an angle: both
        # hold for a square under perspective and both keep the homography stable.
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

    Kept even when rejected so a caller -- currently only the debug renderer -- can
    show *why*: the corners it tried, the tally that fell short, and whether the
    runner-up orientation tied it.
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
        # A real board lights nothing inside its own outline that is not an LED,
        # so unexplained blobs there mean the outline is in the wrong place.
        and unexplained <= MAX_UNEXPLAINED * max(1, explained + unexplained)
        and not ambiguous
    )


def _locate(points, board, min_diagonal):
    """The best candidate quad among ``points``, scored against ``board``, or None.

    ``None`` only when no candidate quad existed at all (fewer than 4 points, or none
    of them paired into one); a quad that was scored but rejected still comes back as
    a ``BoardFit`` with ``accepted=False``, which is what the debug renderer needs.
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
            # The corners themselves land on the model by construction, so score the
            # other blobs only; a quad that explains none of them scores below zero.
            others = np.delete(points, quad, axis=0)
            scores = [_score(others, rot @ homography, layout) for rot in rotations]
            keys = [_rank(score, n_always_on) for score in scores]
            rotation = max(range(4), key=keys.__getitem__)

            if best_key is None or keys[rotation] > best_key:
                best_key = keys[rotation]
                best_score = scores[rotation]
                best_corners = np.roll(corners, rotation, axis=0)
                # A tie means the lit LEDs cannot tell this orientation from the next
                # -- true of a v1 board whose counter reads zero, since the ring on
                # its own is 4-fold symmetric. Reject rather than decode a counter
                # and a ring window off the wrong edge of the board.
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

    ``points`` need not come from an image: a tracker's own 2D or projected-3D
    centroids work exactly the same way, since scoring only ever compares mapped
    points against the board's modelled LED positions. ``min_diagonal`` is the
    source-resolution floor from ``min_board_diagonal`` -- meaningful when ``points``
    were extracted from an image with finite resolution, meaningless (leave at 0) for
    points a tracker already centroided, which have no source pixels to run out of.

    The returned corners are already rotated into correspondence with the model, so
    the resulting warp is upright whenever the frame shows an LED that breaks the
    board's 4-fold symmetry.
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
