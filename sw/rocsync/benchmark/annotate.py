#!/usr/bin/env python3
"""Interactive annotation and verification tool for RocSync benchmark frames.

Runs the rocsync pipeline on validation images and videos, displays results
with LED overlays, and lets the user verify or correct the decoded values.
Produces a ground_truth.json file for benchmark evaluation.

Usage:
    python -m rocsync.benchmark.annotate [data_dir] [-o ground_truth.json]
"""

import argparse
import copy
import itertools
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from rocsync.benchmark.common import (
    MIN_REFERENCE_FRAMES,
    FrameSource,
    annotation_camera,
    collect_frames,
    corner_positions_in_image,
    fit_reference_clock,
    frame_key,
    orphaned_entries,
    parse_frame_key,
    reconstruct_timestamp,
    reference_outliers,
    reference_residual,
)
from rocsync.board_profiles import BOARD_V1, PROFILES_BY_ARUCO, RectifiedBoard
from rocsync.camera import CameraType
from rocsync.timeline import frame_pts, measured_residual_threshold_ms, source_frame_period_ms
from rocsync.vision import ARUCO_DICTIONARY, process_frame, read_counter, read_ring

# Every board rectifies to the same pixel grid, so one radius serves the whole GUI.
LED_RADIUS_PX = BOARD_V1.rectify().led_sample_radius

# Slack warped in around the board, so LEDs that a coarse fit pushes off the edge stay visible.
BOARD_MARGIN_PX = 10

# Length of the tick marks drawn along the board edge at each board corner
BOARD_TICK_PX = 10


# ── Board geometry helpers ──────────────────────────────────────────────────


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


# A homography follows from four point pairs whose board coordinates are in general
# position. v2's LEDs 0, 1 and 4 are collinear, so a fifth-LED set needs a corner off
# that line. The area threshold separates the collinear zero from the smallest triangle
# the layout actually forms (~10000 px² on v2) with room to spare.
MIN_HOMOGRAPHY_CORNERS = 4
MIN_CORNER_TRIANGLE_FRACTION = 0.02  # of board_size, squared, as an area
FIT_WARNING = "Rectified view needs"  # prefix of the message a failed fit leaves


def _min_triangle_area(points):
    """Smallest triangle area over all triples of points — zero if three are collinear."""
    return min(
        abs(np.linalg.det(np.array([points[j] - points[i], points[k] - points[i]]))) / 2
        for i, j, k in itertools.combinations(range(len(points)), 3)
    )


def _determines_homography(points, min_area):
    """Whether four of the points sit in general position, i.e. no three of them collinear."""
    return any(
        _min_triangle_area([points[i] for i in quad]) >= min_area
        for quad in itertools.combinations(range(len(points)), MIN_HOMOGRAPHY_CORNERS)
    )


def fit_corner_homography(corners, board, camera):
    """Fit original image → rectified board from the visible corner LEDs, or None.

    None means the visible corners do not determine a map: too few of them, board
    coordinates that are degenerate, or a fit that came out singular.
    """
    board_pos = board.always_on_leds[camera]
    visible = [i for i, c in enumerate(corners) if c["visible"]]
    if len(visible) < MIN_HOMOGRAPHY_CORNERS:
        return None

    dst = np.array([board_pos[i] for i in visible], dtype=np.float32)
    min_area = (MIN_CORNER_TRIANGLE_FRACTION * board.board_size) ** 2
    if not _determines_homography(dst, min_area):
        return None

    src = np.array([corners[i]["position"] for i in visible], dtype=np.float32)
    try:
        if len(visible) == MIN_HOMOGRAPHY_CORNERS:
            H = cv2.getPerspectiveTransform(src, dst)
        else:
            H, _ = cv2.findHomography(src, dst)
    except cv2.error:
        return None
    if H is None or not np.isfinite(H).all():
        return None
    return H


def coarse_homography_from_aruco(stats, board):
    """Fit original image → rectified board from the ArUco marker alone, or None.

    Recomputed rather than taken from the pipeline's `rough_homography`: the annotator may
    have resolved a different board profile, whose marker sits elsewhere in board coords.
    """
    corners = stats.get("aruco_corners")
    if corners is None:
        return None
    src = np.array(corners, dtype=np.float32).reshape(4, 2)
    try:
        H = cv2.getPerspectiveTransform(src, board.aruco_corners_coords)
    except cv2.error:
        return None
    if H is None or not np.isfinite(H).all():
        return None
    return H


def warp_board_view(image, H, board, margin=BOARD_MARGIN_PX):
    """Rectify `image` into a board grid padded by `margin` px on every side."""
    T = np.array([[1, 0, margin], [0, 1, margin], [0, 0, 1]], dtype=np.float64)
    side = board.board_size + 2 * margin
    return cv2.warpPerspective(image[:, :, 2], T @ np.asarray(H, dtype=np.float64), (side, side))


def board_from_view(view, board, margin=BOARD_MARGIN_PX):
    """The board itself, cropped back out of a padded view."""
    bs = board.board_size
    return view[margin : margin + bs, margin : margin + bs].copy()


def decode_clock(rectified, board, camera):
    """(counter_leds, counter_value, ring_start, ring_end) read off a rectified board.

    The ring is returned in the annotation's half-open ascending form; start == end means
    no arc was found.
    """
    stats = {"steps": {}}
    counter = read_counter(rectified, camera, board, stats=stats)
    ring = read_ring(rectified, camera, board, stats=stats)
    leds = [bool(v) for v in stats["counter_leds"]]
    if ring is None:
        return leds, counter, 0, 0
    return leds, counter, int(ring[0]), int(ring[1] + 1) % board.period


# Known ArUco marker IDs to cycle through
ARUCO_IDS = [p.aruco_marker_id for p in PROFILES_BY_ARUCO.values()]

# Hit-test radii (in board coordinates)
HIT_CORNER = 20
HIT_COUNTER = LED_RADIUS_PX
HIT_RING = 25
DRAG_THRESHOLD = 3  # px before a click becomes a drag


# ── Annotation data ────────────────────────────────────────────────────────


@dataclass
class ImageAnnotation:
    """Mutable annotation state for a single image.

    Corner LED positions are stored in original image coordinates (float).
    The homography maps original image → rectified board space and is needed
    to display / edit positions on the rectified view.

    Ring uses half-open semantics: ring_start = first ON LED,
    ring_end = first OFF LED.  ring_start == ring_end means undecodable.
    """

    board: RectifiedBoard  # not serialized
    camera: CameraType  # not serialized directly; drives which LED layout applies

    aruco_visible: bool = False
    aruco_id: int = 0

    # Each corner: {"visible": bool, "position": [x, y]}
    # position is in *original image* coordinates (float)
    corners: list = field(default_factory=list)

    # Homography: original image → rectified board (3×3, stored as list-of-lists)
    homography: list | None = None

    counter_visible: bool = False
    counter_leds: list = field(default_factory=list)
    counter_value: int = 0

    ring_start: int = 0  # first ON LED index
    ring_end: int = 0  # first OFF LED index (exclusive); == start → undecodable

    def __post_init__(self):
        if not self.corners:
            corner_pos = corner_led_positions(self.board, self.camera)
            self.corners = [{"visible": False, "position": list(p)} for p in corner_pos]
        if not self.counter_leds:
            self.counter_leds = [False] * self.board.counter_bits

    @classmethod
    def from_stats(cls, stats, board, camera):
        """Pre-populate annotation from pipeline stats dict."""
        ann = cls(board=board, camera=camera)

        # Homography (original → rectified)
        H = stats.get("homography")
        if H is not None:
            H = np.array(H, dtype=np.float64)
            ann.homography = H.tolist()

        # ArUco
        aruco_id = stats.get("aruco_id")
        ann.aruco_visible = aruco_id is not None
        ann.aruco_id = aruco_id if aruco_id is not None else board.aruco_marker_id

        # Corner LEDs — the pipeline reports these in original image space
        for i, pos in enumerate(corner_positions_in_image(stats)):
            if i < len(ann.corners) and pos is not None:
                ann.corners[i]["visible"] = True
                ann.corners[i]["position"] = [float(pos[0]), float(pos[1])]

        # Counter
        counter_leds = stats.get("counter_leds")
        if counter_leds is not None:
            ann.counter_visible = True
            ann.counter_leds = list(counter_leds)
            n = board.counter_bits
            ann.counter_value = sum(2 ** (n - 1 - i) for i in range(n) if counter_leds[i])

        # Ring — the decoded arc is inclusive (first ON, last ON). Convert to half-open
        # [start, end) in ascending index order. An arc wrapping the period end is kept as
        # seen; whether it yields a timestamp is reconstruct_timestamp's call, not ours.
        ring = stats.get("ring_window")
        if ring is not None:
            ann.ring_start = int(ring[0])
            ann.ring_end = int(ring[1] + 1) % board.period

        return ann

    def recompute_counter(self):
        n = self.board.counter_bits
        self.counter_value = sum(2 ** (n - 1 - i) for i in range(n) if self.counter_leds[i])

    def to_original(self, board_x, board_y):
        """Transform a point from rectified board coords to original image coords."""
        if self.homography is None:
            return [float(board_x), float(board_y)]
        inv_H = np.linalg.inv(np.array(self.homography, dtype=np.float64))
        pt = np.array([[[board_x, board_y]]], dtype=np.float64)
        orig = cv2.perspectiveTransform(pt, inv_H).reshape(2)
        return [float(orig[0]), float(orig[1])]

    def to_board(self, orig_x, orig_y):
        """Transform a point from original image coords to rectified board coords."""
        if self.homography is None:
            return round(orig_x), round(orig_y)
        H = np.array(self.homography, dtype=np.float64)
        pt = np.array([[[orig_x, orig_y]]], dtype=np.float64)
        board = cv2.perspectiveTransform(pt, H).reshape(2)
        return round(board[0]), round(board[1])

    def to_dict(self):
        """Serialize annotation. Invisible components are stripped to visibility only."""
        corners = []
        for c in self.corners:
            if c["visible"]:
                corners.append({"visible": True, "position": c["position"]})
            else:
                corners.append({"visible": False})

        result = {
            "camera": self.camera.value,
            "aruco": {"visible": self.aruco_visible, "id": self.aruco_id},
            "corners": corners,
            "homography": self.homography,
            "counter": {"visible": self.counter_visible},
            "ring": {},
        }

        if self.counter_visible:
            result["counter"]["value"] = self.counter_value
        if self.ring_start == self.ring_end:
            self.ring_start = 0
            self.ring_end = 0
        result["ring"]["start"] = self.ring_start
        result["ring"]["end"] = self.ring_end

        return result

    @classmethod
    def from_dict(cls, data, board, camera):
        ann = cls(board=board, camera=camera)
        aruco = data.get("aruco", {})
        ann.aruco_visible = aruco.get("visible", False)
        ann.aruco_id = aruco.get("id", board.aruco_marker_id)

        ann.homography = data.get("homography")

        corner_pos = corner_led_positions(board, camera)
        for i, c in enumerate(data.get("corners", [])):
            if i < len(ann.corners):
                if "position" in c:
                    pos = list(c["position"])
                elif ann.homography is not None:
                    pos = ann.to_original(*corner_pos[i])
                else:
                    pos = list(corner_pos[i])
                ann.corners[i] = {
                    "visible": c.get("visible", False),
                    "position": pos,
                }

        counter = data.get("counter", {})
        ann.counter_visible = counter.get("visible", False)
        ann.counter_value = counter.get("value", 0)
        n = board.counter_bits
        ann.counter_leds = [bool(ann.counter_value & (2 ** (n - 1 - i))) for i in range(n)]

        ring = data.get("ring", {})
        ann.ring_start = ring.get("start", 0)
        ann.ring_end = ring.get("end", 0)
        return ann


# ── Interaction state ───────────────────────────────────────────────────────


# ── Reference clock ─────────────────────────────────────────────────────────


def annotated_starts(images, rel_path):
    """Annotated board start time in ms per frame index, for one video."""
    starts = {}
    for key, entry in images.items():
        path, index = parse_frame_key(key)
        if index is None or path != rel_path:
            continue
        board = PROFILES_BY_ARUCO.get(entry.get("aruco", {}).get("id"))
        if board is None:
            continue
        timestamp = reconstruct_timestamp(entry, board)
        if timestamp is not None:
            starts[index] = timestamp[0]
    return starts


def video_rel_paths(frames):
    """Relative paths of the videos in a frame list, sorted."""
    return sorted({parse_frame_key(ref.key)[0] for ref in frames if ref.index is not None})


def derive_reference_clock(starts, pts, threshold_ms):
    """(clock, outliers) for one video, clock None below MIN_REFERENCE_FRAMES.

    Takes the annotated board times rather than finding them, so a caller scoring a
    retimed clip can hand over the source annotations in that clip's own numbering.
    """
    clock = fit_reference_clock(starts, pts)
    if clock is None:
        return None, []
    return clock, reference_outliers(clock, starts, pts, threshold_ms)


def describe_outliers(rel_path, outliers, threshold_ms):
    """One line per frame whose annotation the rest of the video contradicts."""
    lines = [f"{rel_path}: {len(outliers)} frame(s) beyond {threshold_ms:.2f} ms"]
    lines += [f"  {frame_key(rel_path, i)}  {residual:+.2f} ms" for i, residual in outliers]
    return "\n".join(lines)


def _group_by_path(keys):
    """{relative path: [key]} for a list of frame keys, in path order."""
    grouped = {}
    for key in keys:
        grouped.setdefault(parse_frame_key(key)[0], []).append(key)
    return dict(sorted(grouped.items()))


def describe_orphans(orphans, ground_truth):
    """One line per file the ground truth still describes and the dataset no longer backs.

    Grouped by file rather than by key: a video contributes one annotation per frame, and
    a thousand of those say nothing a single line about the file does not.
    """
    images = ground_truth.get("images") or {}
    videos = ground_truth.get("videos") or {}
    lines = []
    for rel_path, keys in _group_by_path(orphans.missing_images).items():
        detail = "reference clock and " if rel_path in orphans.missing_videos else ""
        lines.append(f"  {rel_path}: no such file ({detail}{len(keys)} annotation(s))")
    described = set(_group_by_path(orphans.missing_images))
    for rel_path in orphans.missing_videos:
        if rel_path in described:
            continue
        # A retimed clip is orphaned by its source, which is the file to name
        source = (videos.get(rel_path) or {}).get("source")
        gone = f"its source {source} is gone" if source else "no such file"
        lines.append(f"  {rel_path}: {gone} (reference clock)")
    for rel_path, keys in _group_by_path(orphans.out_of_range_images).items():
        lines.append(f"  {rel_path}: {len(keys)} annotation(s) past the last frame")
    for rel_path in orphans.unreadable_videos:
        n = sum(1 for key in images if parse_frame_key(key)[0] == rel_path)
        kept = f"{n} annotation(s) kept" if n else "no annotations"
        lines.append(f"  {rel_path}: could not be read, so nothing was pruned ({kept})")
    return "\n".join(lines)


def annotations_behind(orphans, images):
    """How many annotations sit behind files that are present but would not decode."""
    unreadable = set(orphans.unreadable_videos)
    return sum(1 for key in images if parse_frame_key(key)[0] in unreadable)


def prune(data_dir, output_path, dry_run=False):
    """Drop ground truth entries whose input is gone. Returns a process exit code.

    Removal is opt-in and never touches a file that merely failed to decode: those are
    listed and kept, and make the exit code non-zero so a partial clean-up is not read
    as a finished one.
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path) if output_path else data_dir / "ground_truth.json"
    if not output_path.exists():
        print(f"No ground truth at {output_path}", file=sys.stderr)
        return 1

    with open(output_path) as f:
        ground_truth = json.load(f)
    images = ground_truth.setdefault("images", {})
    videos = ground_truth.setdefault("videos", {})

    frames = collect_frames(data_dir, sources_only=True)
    orphans = orphaned_entries(ground_truth, data_dir, frames)
    if orphans.is_empty():
        print(f"{output_path}: every entry still has its input")
        return 0

    print(describe_orphans(orphans, ground_truth))
    if not orphans.prunable():
        return 1

    if dry_run:
        print("Dry run, nothing written")
        return 1 if orphans.unreadable_videos else 0

    for key in orphans.missing_images + orphans.out_of_range_images:
        del images[key]
    for rel_path in orphans.missing_videos:
        del videos[rel_path]
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    n_images = len(orphans.missing_images) + len(orphans.out_of_range_images)
    print(f"Removed {n_images} annotation(s) and {len(orphans.missing_videos)} video reference(s)")
    if orphans.out_of_range_images:
        # Those videos are still in the benchmark, and their reference was fitted over
        # annotations that no longer all exist
        print("Re-derive the affected clocks with --fit-clocks")
    return 1 if annotations_behind(orphans, images) else 0


def measured_video_entry(clock, source_period_ms, threshold_ms):
    """The `videos` entry for a clip whose timeline is the camera's own."""
    return {
        **clock.to_dict(threshold_ms),
        "timeline": "measured",
        "source_frame_period_ms": source_period_ms,
        "derived_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def is_synthesized(videos, rel_path):
    """Whether a stored entry describes a retimed clip, which is not ours to re-derive."""
    return (videos.get(rel_path) or {}).get("timeline") == "synthesized"


class Mode(Enum):
    IDLE = "idle"
    DRAGGING_CORNER = "dragging"
    RING_AWAITING_END = "ring_end"
    LAYOUT_CONFIRM = "layout_confirm"  # board, marker, or camera mode changed, Esc to revert


# ── Display and interaction ─────────────────────────────────────────────────

COLOR_ON = (0, 0, 255)  # red  — LED is ON
COLOR_OFF = (255, 0, 0)  # blue — LED is OFF
COLOR_NOT_VIS = (128, 128, 128)  # gray — component hidden/undecodable
COLOR_RING_SEL = (0, 255, 0)  # green — ring selection highlight
COLOR_TEXT = (0, 0, 0)  # black text on bright bg
COLOR_STATUS_BG = (220, 220, 220)  # light gray
COLOR_BOARD_TEXT = (255, 255, 255)  # white text on dark board
COLOR_MARGIN = (180, 180, 180)  # light gray — board corner ticks inside the margin

WINDOW_NAME = "RocSync Annotation"
TARGET_HEIGHT = 800

# Every label uses one size, and is drawn at display resolution so it never gets
# magnified along with the panel it sits on.
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.75
LINE_H = 32  # baseline spacing for stacked text rows


# Zoom inset shown while placing corners in the left panel.
LOUPE_SIZE = 220  # on-screen size of the inset (px)
LOUPE_SOURCE_PX = 70  # side of the original-image region it magnifies
LOUPE_MARGIN = 12

HELP_TEXT = [
    "=== Mouse (left panel) ===",
    "Click anywhere    : Place the pending corner,",
    "                    then advance to the next",
    "Drag corner LED   : Refine position",
    "                    (zoom inset follows cursor)",
    "=== Mouse (right panel) ===",
    "Click corner LED  : Toggle visible / hidden",
    "Drag corner LED   : Refine position",
    "Click counter LED : Toggle ON / OFF",
    "Click counter box : Toggle counter visibility",
    "Click ArUco area  : Cycle ID 0 visible/hidden,",
    "                    then ID 21 visible/hidden",
    "Click ring LED    : Set first ON, then first OFF",
    "                    (same LED = undecodable)",
    "=== Keys ===",
    "0-N               : Place that corner next",
    "M                 : Toggle camera mode (RGB / IR)",
    ". / ,             : Next / previous input file",
    "=== Rectified view ===",
    "Any corner edit re-fits the board view,",
    "from 4 or more visible corner LEDs, and",
    "re-reads counter and ring off the new fit",
    "until you annotate either by hand",
]


def draw_text(img, text, org, color, scale=FONT_SCALE, thickness=1):
    """Draw a label in the one global font. Hershey strokes look ragged without LINE_AA."""
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def fit_scale(text, max_w, scale=FONT_SCALE):
    """Largest font scale up to `scale` at which `text` still fits `max_w` pixels."""
    w = cv2.getTextSize(text, FONT, scale, 1)[0][0]
    return scale if w <= max_w else scale * max_w / w


class AnnotationTool:
    # Per-image state. Set by run() before any handler can fire, so it is never None.
    annotation: ImageAnnotation
    stats: dict
    _current_image: np.ndarray

    def __init__(self, data_dir, output_path=None):
        self.data_dir = Path(data_dir)
        self.output_path = Path(output_path) if output_path else self.data_dir / "ground_truth.json"
        self.ground_truth = {"images": {}, "videos": {}}
        self.frames = []
        self.source = FrameSource()
        self.current_idx = 0

        # Reference clock state. pts are read once per video and kept; a video is
        # re-derived only once one of its annotations has actually changed.
        self._video_pts: dict[str, dict[int, float]] = {}
        self._video_threshold: dict[str, tuple[float | None, float]] = {}
        self._dirty_videos: set[str] = set()

        # Board profile, camera mode, and derived geometry (set per-image)
        self.board = BOARD_V1.rectify()
        self.camera = CameraType.RGB
        self.ring_pos = ring_led_positions(self.board, self.camera)
        self.counter_pos = counter_led_positions(self.board, self.camera)
        self.corner_pos = corner_led_positions(self.board, self.camera)
        self.counter_box = counter_bbox(self.counter_pos)
        self.aruco_x1 = int(self.board.aruco_corners_coords[0][0])
        self.aruco_y1 = int(self.board.aruco_corners_coords[0][1])
        self.aruco_x2 = int(self.board.aruco_corners_coords[2][0])
        self.aruco_y2 = int(self.board.aruco_corners_coords[2][1])

        # Display state
        self.show_help = False
        self.left_panel_w = 0
        self.left_scale = 1.0
        self.board_scale = 1.0
        self._board_view = None  # rectified board plus margin, as drawn on the right
        self.status_msg = ""
        self._last_composite = None  # last frame, re-presented while work blocks the loop
        self._pending_key: int | None = None  # key caught by _pump, consumed by _image_loop

        # Interaction state
        self.mode = Mode.IDLE
        self._drag_idx: int | None = None
        self._drag_start: tuple[float, float] | None = None
        self._drag_start_original: tuple[float, float] | None = None
        self._drag_started = False
        self._drag_on_left = False
        self._ring_start_candidate: int | None = None
        # Corner the next left-panel click places; advances with each placement.
        self.placing_idx = 0
        self._cursor_left: tuple[float, float] | None = None
        # Whether the user took over the clock / ring, which stops the auto re-decode
        self._counter_edited = False
        self._ring_edited = False
        self._needs_redraw = True
        self._window_sized = False
        # Annotation snapshots keyed by (aruco_id, camera.value), taken before a layout
        # change; lets a mis-click on a board or mode change undo without losing work.
        self._layout_snapshots: dict[tuple[int, str], ImageAnnotation] = {}
        self._layout_original: tuple[ImageAnnotation, RectifiedBoard, CameraType] | None = None

    def _set_layout(self, board, camera):
        """Update board profile, camera mode, and recompute derived geometry."""
        self.board = board
        self.camera = camera
        self.ring_pos = ring_led_positions(board, camera)
        self.counter_pos = counter_led_positions(board, camera)
        self.corner_pos = corner_led_positions(board, camera)
        self.counter_box = counter_bbox(self.counter_pos)
        self.aruco_x1 = int(board.aruco_corners_coords[0][0])
        self.aruco_y1 = int(board.aruco_corners_coords[0][1])
        self.aruco_x2 = int(board.aruco_corners_coords[2][0])
        self.aruco_y2 = int(board.aruco_corners_coords[2][1])

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self):
        self.frames = collect_frames(self.data_dir, sources_only=True)
        if not self.frames:
            print("No images or videos found.", file=sys.stderr)
            return

        self._load_ground_truth()
        self._report_orphans()
        first = self._find_unannotated(-1, forward=True)
        if first == -1:
            print("All frames already annotated. Starting from the beginning.")
            self.current_idx = 0
        else:
            self.current_idx = first
            print(f"Resuming at frame {first + 1}/{len(self.frames)}: {self.frames[first].key}")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        n_frames = len(self.frames)
        unreadable = 0  # consecutive read failures; bail out rather than spin forever
        while True:
            ref = self.frames[self.current_idx]
            frame_key = ref.key

            # Decoding a video frame can seek, so present a frame before blocking on it
            self._pump()
            image = self.source.read(ref)
            if image is None:
                print(f"Cannot read {frame_key}", file=sys.stderr)
                self._pump()
                unreadable += 1
                if unreadable >= n_frames:
                    print("No readable frames.", file=sys.stderr)
                    break
                self.current_idx = (self.current_idx + 1) % n_frames
                continue
            unreadable = 0
            self._pump()

            # Resolve camera mode and a board guess from the saved annotation if there is
            # one, else stay on whatever the previous frame left `self.camera`/`self.board`
            # at — a video's frames are usually all the same mode, so this is what keeps a
            # long IR clip from needing `M` pressed once per frame.
            saved = frame_key in self.ground_truth["images"]
            entry = self.ground_truth["images"].get(frame_key)
            camera = annotation_camera(entry) if entry is not None else self.camera
            guess_profile = (
                PROFILES_BY_ARUCO.get(entry.get("aruco", {}).get("id"), self.board.profile)
                if entry is not None
                else self.board.profile
            )

            # Run pipeline
            self.stats = {}
            try:
                # No area gate: the annotator should see every marker the detector found,
                # however small — rejecting it is the benchmark's call, not the ground truth's.
                # IR has no marker to auto-resolve a board from, so it always needs an
                # explicit guess; RGB stays free to auto-resolve on an unannotated frame.
                board_hint = guess_profile if saved or camera == CameraType.INFRARED else None
                process_frame(
                    image,
                    camera,
                    0,
                    board=board_hint,
                    stats=self.stats,
                    min_aruco_area_fraction=0.0,
                    try_hard=True,
                )
            except Exception as e:  # noqa: BLE001 — a bad frame must not kill the session
                print(f"Pipeline error on {frame_key}: {e}", file=sys.stderr)
                self.stats["rectified"] = None
                self.stats["aruco_id"] = None
            self._pump()

            # Resolve board profile from pipeline detection or saved annotation
            aruco_id = self.stats.get("aruco_id")
            if entry is not None:
                aruco_id = entry.get("aruco", {}).get("id", aruco_id)
            elif aruco_id is None:
                aruco_id = guess_profile.aruco_marker_id
            profile = PROFILES_BY_ARUCO.get(aruco_id, BOARD_V1)
            board = profile.rectify()
            self._set_layout(board, camera)

            # Load existing annotation or create from pipeline
            coarse_fit = False
            if saved:
                self.annotation = ImageAnnotation.from_dict(entry, board, camera)
            else:
                self.annotation = ImageAnnotation.from_stats(self.stats, board, camera)
                # Use the pipeline homography for new annotations, falling back to the
                # marker alone when corner detection failed — a coarse view to refine
                # beats a black panel and corners parked at a guess.
                H = self.stats.get("homography")
                if H is None:
                    H = coarse_homography_from_aruco(self.stats, board)
                    if H is not None:
                        coarse_fit = True
                if H is not None:
                    self.annotation.homography = np.array(H, dtype=np.float64).tolist()

            # Ensure invisible corners have positions in original image coords
            if self.annotation.homography is not None:
                for i, c in enumerate(self.annotation.corners):
                    if not c["visible"]:
                        c["position"] = self.annotation.to_original(*self.corner_pos[i])
            else:
                # No homography — place invisible corners in a centered square
                # proportional to the board's corner layout.
                img_h, img_w = image.shape[:2]
                scale = 0.5 * min(img_w, img_h) / board.board_size
                ox = (img_w - board.board_size * scale) / 2
                oy = (img_h - board.board_size * scale) / 2
                for i, c in enumerate(self.annotation.corners):
                    if not c["visible"]:
                        bx, by = self.corner_pos[i]
                        c["position"] = [ox + bx * scale, oy + by * scale]

            # Warp with the annotation's own homography, saved or from the pipeline, so
            # the displayed image is consistent with to_original/to_board. Loading never
            # refits: a stored annotation keeps the homography it was accepted with.
            self._board_view = None
            if self.annotation.homography is not None:
                H = np.array(self.annotation.homography, dtype=np.float64)
                self._board_view = warp_board_view(image, H, board)
                self.stats["rectified"] = board_from_view(self._board_view, board)
            self._pump()

            # Demuxing for presentation timestamps blocks, so do it before the loop
            if ref.index is not None:
                self._load_pts(parse_frame_key(frame_key)[0])
            self._pump()

            self.mode = Mode.IDLE
            self._ring_start_candidate = None
            self.placing_idx = 0
            self._cursor_left = None
            self.status_msg = ""
            self._needs_redraw = True
            # A stored annotation was accepted as it stands, so it counts as hand-made:
            # a corner nudge must not silently overwrite the reading it was accepted with.
            self._counter_edited = saved
            self._ring_edited = saved

            self._current_image = image

            if coarse_fit:
                # The pipeline stopped before reading anything, so seed the reading off
                # the coarse view — wrong in detail, but a starting point to correct.
                self._redecode_clock()
                self.status_msg = "Coarse fit from ArUco marker — refine the corner LEDs"
            action = self._image_loop(image, frame_key)

            if action == "accept":
                self.ground_truth["images"][frame_key] = self.annotation.to_dict()
                self._mark_video_dirty(frame_key)
                self._save_ground_truth()
                # Show next unannotated frame, or next frame if no unannoted exists
                nxt = self._find_unannotated(self.current_idx, forward=True)
                self.current_idx = (
                    nxt if nxt != self.current_idx else (self.current_idx + 1) % n_frames
                )
            elif action == "skip":
                self.current_idx = (self.current_idx + 1) % n_frames
            elif action == "back":
                self.current_idx = (self.current_idx - 1) % n_frames
            elif action == "next_unannotated":
                self.current_idx = self._find_unannotated(self.current_idx, forward=True)
            elif action == "prev_unannotated":
                self.current_idx = self._find_unannotated(self.current_idx, forward=False)
            elif action == "next_source":
                self.current_idx = self._find_source_boundary(self.current_idx, forward=True)
            elif action == "prev_source":
                self.current_idx = self._find_source_boundary(self.current_idx, forward=False)
            elif action == "clear":
                self.ground_truth["images"].pop(frame_key, None)
                self._mark_video_dirty(frame_key)
                self._save_ground_truth()
                # stay on current frame — re-runs pipeline on next iteration
            elif action == "quit":
                break

        self.source.close()
        cv2.destroyAllWindows()

    def _pump(self):
        """Present a frame and service events so the window manager sees a live window."""
        if self._last_composite is not None:
            cv2.imshow(WINDOW_NAME, self._last_composite)
        key = cv2.waitKey(1) & 0xFF
        if key != 255 and self._pending_key is None:
            self._pending_key = key  # hold keys typed while work blocked the loop

    def _image_loop(self, image, frame_key):
        composite = None
        while True:
            if self._needs_redraw or composite is None:
                composite = self._render(image, frame_key)
                if not self._window_sized:
                    h, w = composite.shape[:2]
                    cv2.resizeWindow(WINDOW_NAME, w, h)
                    self._window_sized = True
                self._needs_redraw = False

            # A window that stops presenting frames is flagged as not responding.
            cv2.imshow(WINDOW_NAME, composite)
            self._last_composite = composite

            if self._pending_key is not None:
                key = self._pending_key
                self._pending_key = None
            else:
                key = cv2.waitKey(30) & 0xFF

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return "quit"

            if key == 255:
                continue

            action = self._process_key(key)
            if action is not None:
                return action

    # ── Keyboard handling ───────────────────────────────────────────────

    def _process_key(self, key):
        if key in (13, 10, 32):  # Enter or Space
            self._commit_layout_change()
            return "accept"
        elif key == 83 or key == ord("d"):  # Right arrow
            self._commit_layout_change()
            return "skip"
        elif key == 81 or key == ord("a"):  # Left arrow
            self._commit_layout_change()
            return "back"
        elif key in (ord("n"), ord("N")):
            self._commit_layout_change()
            return "next_unannotated"
        elif key in (ord("b"), ord("B")):
            self._commit_layout_change()
            return "prev_unannotated"
        elif key == ord("."):
            self._commit_layout_change()
            return "next_source"
        elif key == ord(","):
            self._commit_layout_change()
            return "prev_source"
        elif key in (ord("c"), ord("C"), 8, 127):  # C or Backspace
            self._commit_layout_change()
            return "clear"
        elif key in (ord("q"), ord("Q")):
            self._commit_layout_change()
            return "quit"
        elif key in (ord("h"), ord("H")):
            self.show_help = not self.show_help
            self._needs_redraw = True
        elif key in (ord("m"), ord("M")):
            self._cycle_camera()
        elif ord("0") <= key <= ord("9"):
            self._select_placing_corner(key - ord("0"))
        elif key == 27:  # Escape
            if self.mode == Mode.LAYOUT_CONFIRM:
                self._revert_layout_change()
            elif self.mode == Mode.RING_AWAITING_END:
                self.mode = Mode.IDLE
                self._ring_start_candidate = None
                self.status_msg = ""
                self._needs_redraw = True
        return None

    # ── Mouse handling ──────────────────────────────────────────────────

    def _display_to_board(self, x, y):
        """Convert display pixel coords to board coords. Returns None if outside board panel."""
        if x < self.left_panel_w:
            return None, None
        panel_x = x - self.left_panel_w
        m = BOARD_MARGIN_PX
        bx = int(panel_x / self.board_scale) - m
        by = int(y / self.board_scale) - m
        bs = self.board.board_size
        if bx < -m or bx >= bs + m or by < -m or by >= bs + m:
            return None, None
        return bx, by

    def _display_to_original(self, x, y):
        """Convert display pixel coords to original image coords. Returns None if outside left panel."""
        if x >= self.left_panel_w:
            return None, None
        ox = x / self.left_scale
        oy = y / self.left_scale
        return ox, oy

    def _hit_test(self, bx, by):
        """Returns (element_type, index) or (None, None)."""
        # 1. Corner LEDs
        for i, c in enumerate(self.annotation.corners):
            cx, cy = self.annotation.to_board(*c["position"])
            if (bx - cx) ** 2 + (by - cy) ** 2 <= HIT_CORNER**2:
                return "corner", i

        # 2. Counter LEDs (individual)
        for i, (cx, cy) in enumerate(self.counter_pos):
            if (bx - cx) ** 2 + (by - cy) ** 2 <= HIT_COUNTER**2:
                return "counter", i

        # 3. Counter bounding box (outside LED circles → toggle visibility)
        x1, y1, x2, y2 = self.counter_box
        if x1 <= bx <= x2 and y1 <= by <= y2:
            return "counter_bbox", 0

        # 4. ArUco region
        if self.aruco_x1 <= bx <= self.aruco_x2 and self.aruco_y1 <= by <= self.aruco_y2:
            return "aruco", 0

        # 5. Ring LEDs (nearest within radius)
        best_i, best_d = 0, float("inf")
        for i, (rx, ry) in enumerate(self.ring_pos):
            d = (bx - rx) ** 2 + (by - ry) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        if best_d <= HIT_RING**2:
            return "ring", best_i

        return None, None

    def _hit_test_corner_original(self, ox, oy):
        """Hit-test corners in original image coordinates."""
        img_h, img_w = self._current_image.shape[:2]
        hit_r = 2 * int(max(0.01 * min(img_w, img_h), 2))
        for i, c in enumerate(self.annotation.corners):
            cx, cy = c["position"]
            if (ox - cx) ** 2 + (oy - cy) ** 2 <= hit_r**2:
                return i
        return None

    def _mouse_callback(self, event, x, y, flags, param):
        # Try left panel (original image) — corners only
        ox, oy = self._display_to_original(x, y)
        if ox is not None and oy is not None:
            self._cursor_left = (ox, oy)
            if event == cv2.EVENT_MOUSEMOVE:
                self._needs_redraw = True
            if event == cv2.EVENT_LBUTTONDOWN:
                self._commit_layout_change()
                idx = self._hit_test_corner_original(ox, oy)
                if idx is not None:
                    # On a corner overlay: hold for a drag, release without one to place.
                    self._drag_idx = idx
                    self._drag_start_original = (ox, oy)
                    self._drag_started = False
                    self._drag_on_left = True
                else:
                    self._place_corner(ox, oy)
            elif event == cv2.EVENT_MOUSEMOVE and self._drag_on_left:
                if (
                    self._drag_idx is not None
                    and self._drag_start_original is not None
                    and (flags & cv2.EVENT_FLAG_LBUTTON)
                ):
                    dx = ox - self._drag_start_original[0]
                    dy = oy - self._drag_start_original[1]
                    img_h, img_w = self._current_image.shape[:2]
                    threshold = int(max(0.01 * min(img_w, img_h), DRAG_THRESHOLD))
                    if not self._drag_started and (dx * dx + dy * dy) > threshold**2:
                        self._drag_started = True
                        self.mode = Mode.DRAGGING_CORNER
                    if self._drag_started:
                        self.annotation.corners[self._drag_idx]["position"] = [ox, oy]
                        self._needs_redraw = True
            elif event == cv2.EVENT_LBUTTONUP and self._drag_on_left:
                if self._drag_idx is not None:
                    if self._drag_started:
                        self._recompute_homography(self._current_image)
                    else:
                        self._place_corner(ox, oy)
                    self._drag_idx = None
                    self._drag_started = False
                    self._drag_on_left = False
                    self.mode = Mode.IDLE
                    self._needs_redraw = True
            return

        # Right panel (rectified board)
        if self._cursor_left is not None:
            self._cursor_left = None  # cursor left the loupe's panel
            self._needs_redraw = True

        bx, by = self._display_to_board(x, y)
        if bx is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_on_left = False
            self._on_left_down(bx, by)
        elif event == cv2.EVENT_MOUSEMOVE:
            self._on_mouse_move(bx, by, flags)
        elif event == cv2.EVENT_LBUTTONUP:
            self._on_left_up(bx, by)

    def _on_left_down(self, bx, by):
        if self.mode == Mode.RING_AWAITING_END:
            elem, idx = self._hit_test(bx, by)
            if elem == "ring" and idx is not None and self._ring_start_candidate is not None:
                # Convert CW clicks to ascending [start, end):
                # CW first ON  → ascending end (exclusive)
                # CW first OFF → ascending start
                self.annotation.ring_start = (idx + 1) % self.board.period
                self.annotation.ring_end = (self._ring_start_candidate + 1) % self.board.period
                self._ring_edited = True
                self.mode = Mode.IDLE
                self._ring_start_candidate = None
                self.status_msg = ""
                self._needs_redraw = True
            return

        elem, idx = self._hit_test(bx, by)
        if elem != "aruco":
            self._commit_layout_change()
        if elem == "corner":
            self._drag_idx = idx
            self._drag_start = (bx, by)
            self._drag_started = False
        elif elem == "counter":
            self._toggle_counter_led(idx)
        elif elem == "counter_bbox":
            self._toggle_counter_visibility()
        elif elem == "aruco":
            self._cycle_aruco()
        elif elem == "ring":
            self._handle_ring_first_click(idx)

    def _on_mouse_move(self, bx, by, flags):
        if (
            self._drag_idx is not None
            and self._drag_start is not None
            and (flags & cv2.EVENT_FLAG_LBUTTON)
        ):
            dx = bx - self._drag_start[0]
            dy = by - self._drag_start[1]
            if not self._drag_started and (dx * dx + dy * dy) > DRAG_THRESHOLD**2:
                self._drag_started = True
                self.mode = Mode.DRAGGING_CORNER
            if self._drag_started:
                bs = self.board.board_size
                m = BOARD_MARGIN_PX
                cbx = max(-m, min(bs - 1 + m, bx))
                cby = max(-m, min(bs - 1 + m, by))
                self.annotation.corners[self._drag_idx]["position"] = self.annotation.to_original(
                    cbx, cby
                )
                self._needs_redraw = True

    def _on_left_up(self, bx, by):
        if self._drag_idx is not None:
            if not self._drag_started:
                self._cycle_corner_state(self._drag_idx)
            self._recompute_homography(self._current_image)
            self._drag_idx = None
            self._drag_started = False
            self.mode = Mode.IDLE
            self._needs_redraw = True

    # ── Direct corner placement (left panel) ────────────────────────────

    def _select_placing_corner(self, idx):
        """Choose which corner the next left-panel click places."""
        if not 0 <= idx < len(self.annotation.corners):
            return
        self.placing_idx = idx
        self._needs_redraw = True

    def _place_corner(self, ox, oy):
        """Set the pending corner to the clicked point and advance to the next one."""
        corner = self.annotation.corners[self.placing_idx]
        corner["position"] = [ox, oy]
        corner["visible"] = True
        self._recompute_homography(self._current_image)
        self.placing_idx = (self.placing_idx + 1) % len(self.annotation.corners)
        self._needs_redraw = True

    def _cycle_corner_state(self, idx):
        c = self.annotation.corners[idx]
        c["visible"] = not c["visible"]
        self._needs_redraw = True

    def _toggle_counter_led(self, idx):
        self._counter_edited = True
        if self.annotation.counter_visible:
            self.annotation.counter_leds[idx] = not self.annotation.counter_leds[idx]
            self.annotation.recompute_counter()
        else:
            self.annotation.counter_visible = True
        self._needs_redraw = True

    def _toggle_counter_visibility(self):
        self._counter_edited = True
        self.annotation.counter_visible = not self.annotation.counter_visible
        self._needs_redraw = True

    def _cycle_aruco(self):
        """Cycle (marker id, visible) through every board with the marker shown or hidden.

        Four states for two board revisions: id0 visible → id0 hidden → id1 visible →
        id1 hidden → … A board must stay selectable with the marker hidden — the ArUco
        marker itself is never visible in IR — so `visible` toggles independently of which
        board is picked, rather than `none` being a single shared third state as before.
        """
        ann = self.annotation
        states = [(marker_id, visible) for marker_id in ARUCO_IDS for visible in (True, False)]
        try:
            idx = states.index((ann.aruco_id, ann.aruco_visible))
        except ValueError:
            idx = -1
        new_id, new_visible = states[(idx + 1) % len(states)]
        new_board = PROFILES_BY_ARUCO[new_id].rectify() if new_id in PROFILES_BY_ARUCO else self.board
        self._switch_layout(new_board, self.camera, new_visible, new_id)

    def _cycle_camera(self):
        """Toggle RGB / IR, keeping the current board and marker selection."""
        new_camera = CameraType.INFRARED if self.camera == CameraType.RGB else CameraType.RGB
        ann = self.annotation
        self._switch_layout(self.board, new_camera, ann.aruco_visible, ann.aruco_id)

    def _switch_layout(self, new_board, new_camera, new_aruco_visible, new_aruco_id):
        """Switch to a different board / camera / marker layout.

        Snapshotted per (board id, camera), so returning to a layout already visited this
        frame restores the annotation made under it rather than resetting it — a mis-click
        on `M` or the ArUco region must not lose work. RGB and IR corner LEDs are different
        physical LEDs, and so are v1's and v2's, so a layout that has not been visited yet
        starts from a fresh annotation rather than carrying positions over.
        """
        ann = self.annotation
        geometry_changes = new_board is not self.board or new_camera != self.camera

        # Snapshot on the first change this frame; later changes update in place
        if self._layout_original is None:
            self._layout_original = (copy.deepcopy(ann), self.board, self.camera)

        if not geometry_changes:
            ann.aruco_visible = new_aruco_visible
            ann.aruco_id = new_aruco_id
            self.status_msg = "ArUco changed — Esc to undo"
            self.mode = Mode.LAYOUT_CONFIRM
            self._needs_redraw = True
            return

        old_key = (self.board.aruco_marker_id, self.camera.value)
        new_key = (new_board.aruco_marker_id, new_camera.value)
        self._layout_snapshots[old_key] = copy.deepcopy(ann)
        self._set_layout(new_board, new_camera)

        if new_key in self._layout_snapshots:
            # Layout visited before this frame — restore its annotation
            self.annotation = self._layout_snapshots.pop(new_key)
            self.annotation.aruco_visible = new_aruco_visible
            self.annotation.aruco_id = new_aruco_id
        else:
            new_ann = ImageAnnotation(board=new_board, camera=new_camera)
            new_ann.aruco_visible = new_aruco_visible
            new_ann.aruco_id = new_aruco_id
            # Carry over layout-independent state
            new_ann.homography = ann.homography
            if new_board.period == ann.board.period:
                new_ann.ring_start = ann.ring_start
                new_ann.ring_end = ann.ring_end
            # Place corners at the new layout's default positions.
            # If a homography is available, convert to original image coords; otherwise
            # center a scaled copy of the board layout on the image, matching the
            # centering a freshly loaded frame gets in `run()`.
            new_corner_pos = corner_led_positions(new_board, new_camera)
            if new_ann.homography is None:
                img_h, img_w = self._current_image.shape[:2]
                scale = 0.5 * min(img_w, img_h) / new_board.board_size
                ox = (img_w - new_board.board_size * scale) / 2
                oy = (img_h - new_board.board_size * scale) / 2
            for i, c in enumerate(new_ann.corners):
                if new_ann.homography is not None:
                    c["position"] = new_ann.to_original(*new_corner_pos[i])
                else:
                    bx, by = new_corner_pos[i]
                    c["position"] = [ox + bx * scale, oy + by * scale]
                c["visible"] = False
            self.annotation = new_ann

        self.status_msg = f"Layout changed to {new_board.profile.name}/{new_camera.value} — Esc to undo"
        self.mode = Mode.LAYOUT_CONFIRM
        self._needs_redraw = True

    def _commit_layout_change(self):
        """Discard layout snapshots, committing the board / camera / marker in place."""
        if self._layout_original is not None:
            self._layout_original = None
            self._layout_snapshots.clear()
            if self.mode == Mode.LAYOUT_CONFIRM:
                self.mode = Mode.IDLE
                self.status_msg = ""
                self._needs_redraw = True

    def _revert_layout_change(self):
        """Restore annotation, board and camera from the pre-change snapshot."""
        if self._layout_original is None:
            return
        self.annotation, old_board, old_camera = self._layout_original
        self._layout_original = None
        self._layout_snapshots.clear()
        self._set_layout(old_board, old_camera)
        self.mode = Mode.IDLE
        self.status_msg = ""
        self._needs_redraw = True

    def _recompute_homography(self, original):
        """Refit the homography and re-warp the board from the visible corners.

        Returns whether the fit succeeded; a failed one leaves the previous homography
        and rectified view in place, so the board stays usable while corners are edited.
        """
        ann = self.annotation
        board = self.board
        H = fit_corner_homography(ann.corners, board, self.camera)
        if H is None:
            n = sum(1 for c in ann.corners if c["visible"])
            self.status_msg = (
                f"{FIT_WARNING} {MIN_HOMOGRAPHY_CORNERS} non-collinear corner LEDs ({n} visible)"
            )
            return False
        if self.status_msg.startswith(FIT_WARNING):
            self.status_msg = ""
        ann.homography = H.tolist()
        # Hidden corners have no annotated position, so park them where the fit predicts.
        for i, c in enumerate(ann.corners):
            if not c["visible"]:
                c["position"] = ann.to_original(*self.corner_pos[i])
        self._board_view = warp_board_view(original, H, board)
        self.stats["rectified"] = board_from_view(self._board_view, board)
        self._redecode_clock()
        return True

    def _redecode_clock(self):
        """Re-read counter and ring off the current rectified view.

        Skips whichever the user has annotated by hand, so the auto-decode only ever fills
        in what nobody has decided yet.
        """
        rectified = self.stats.get("rectified")
        if rectified is None or (self._counter_edited and self._ring_edited):
            return
        leds, value, start, end = decode_clock(rectified, self.board, self.camera)
        ann = self.annotation
        if not self._counter_edited:
            ann.counter_visible = True
            ann.counter_leds = leds
            ann.counter_value = value
        if not self._ring_edited:
            ann.ring_start = start
            ann.ring_end = end

    def _handle_ring_first_click(self, idx):
        """First ring click (CW first ON LED), then wait for CW first OFF LED."""
        self._ring_start_candidate = idx
        self.mode = Mode.RING_AWAITING_END
        self.status_msg = f"Ring: (clock-wise) first ON={idx}, click first OFF LED..."
        self._needs_redraw = True

    # ── Rendering ───────────────────────────────────────────────────────

    def _render(self, original, frame_key):
        """Build the composite side-by-side display image."""
        ann = self.annotation

        # Left panel: original image, downscaled first so the overlays are drawn
        # at display resolution — sharp labels, and no full-res copies per redraw.
        h = TARGET_HEIGHT
        orig_h, orig_w = original.shape[:2]
        self.left_scale = h / orig_h
        left = cv2.resize(original, (round(orig_w * self.left_scale), h))
        left_h, left_w = left.shape[:2]
        corner_radius = int(max(0.01 * min(left_w, left_h), 2))
        overlay = left.copy()
        for i, c in enumerate(ann.corners):
            cx = round(c["position"][0] * self.left_scale)
            cy = round(c["position"][1] * self.left_scale)
            color = COLOR_ON if c["visible"] else COLOR_NOT_VIS
            cv2.circle(overlay, (cx, cy), corner_radius, (0, 255, 255), -1)
            cv2.circle(left, (cx, cy), corner_radius, color, 2)
            if i == self.placing_idx:  # the corner the next click places
                cv2.circle(left, (cx, cy), corner_radius + 5, COLOR_RING_SEL, 2)
            draw_text(left, str(i), (cx + corner_radius + 6, cy + 7), color, thickness=2)
        cv2.addWeighted(overlay, 0.35, left, 0.65, 0, dst=left)

        # Which corner the next click places, and how to pick another. On a darkened
        # strip so it stays readable over whatever the image happens to show.
        hint = f"Click places corner {self.placing_idx}   (keys 0-{len(ann.corners) - 1})"
        (tw, th), _ = cv2.getTextSize(hint, FONT, FONT_SCALE, 2)
        strip = left[0 : th + 20, 0 : tw + 24]
        cv2.addWeighted(strip, 0.25, np.zeros_like(strip), 0.75, 0, dst=strip)
        draw_text(left, hint, (12, th + 9), COLOR_RING_SEL, thickness=2)

        # Right panel: rectified board with overlays
        bs = self.board.board_size
        m = BOARD_MARGIN_PX
        cs = bs + 2 * m
        view = self._board_view
        if view is not None:
            board_img = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
        else:
            board_img = np.zeros((cs, cs, 3), dtype=np.uint8)
        # Ticks marking where the board proper ends and the margin begins: an L at each
        # corner, plus one centred on each edge.
        x0, x1 = m, m + bs - 1
        y0, y1 = m, m + bs - 1
        half = BOARD_TICK_PX // 2
        xc, yc = m + bs // 2, m + bs // 2
        for cx, sx in ((x0, 1), (x1, -1)):
            for cy, sy in ((y0, 1), (y1, -1)):
                cv2.line(board_img, (cx, cy), (cx + sx * BOARD_TICK_PX, cy), COLOR_MARGIN, 1)
                cv2.line(board_img, (cx, cy), (cx, cy + sy * BOARD_TICK_PX), COLOR_MARGIN, 1)
        for y in (y0, y1):
            cv2.line(board_img, (xc - half, y), (xc - half + BOARD_TICK_PX, y), COLOR_MARGIN, 1)
        for x in (x0, x1):
            cv2.line(board_img, (x, yc - half), (x, yc - half + BOARD_TICK_PX), COLOR_MARGIN, 1)

        # ArUco overlay (always shown)
        ax1, ay1 = self.aruco_x1 + m, self.aruco_y1 + m
        ax2, ay2 = self.aruco_x2 + m, self.aruco_y2 + m
        if ann.aruco_visible:
            marker_size = ax2 - ax1
            marker = cv2.aruco.generateImageMarker(ARUCO_DICTIONARY, ann.aruco_id, marker_size)
            marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            board_img[ay1:ay2, ax1:ax2] = cv2.addWeighted(
                board_img[ay1:ay2, ax1:ax2], 0.5, marker_bgr, 0.5, 0
            )
        else:
            # Grey box with diagonal cross
            cv2.rectangle(board_img, (ax1, ay1), (ax2, ay2), COLOR_NOT_VIS, 2)
            cv2.line(board_img, (ax1, ay1), (ax2, ay2), COLOR_NOT_VIS, 2)
            cv2.line(board_img, (ax2, ay1), (ax1, ay2), COLOR_NOT_VIS, 2)

        # Corner LEDs (positions stored in original image space, transform to board).
        # Without a homography there is nothing to project through — `to_board` would
        # fall back to treating the image-space position as board coordinates, which
        # scrambles the layout — so draw the board's own default corner positions instead.
        corner_labels = []
        for i, c in enumerate(ann.corners):
            if ann.homography is None:
                cx, cy = self.corner_pos[i]
            else:
                cx, cy = ann.to_board(*c["position"])
            cx, cy = cx + m, cy + m
            color = COLOR_ON if c["visible"] else COLOR_NOT_VIS
            cv2.circle(board_img, (cx, cy), LED_RADIUS_PX + 4, color, 2)
            corner_labels.append((str(i), cx + LED_RADIUS_PX + 8, cy + 6, color))

        # Counter bounding box
        bx1, by1, bx2, by2 = (v + m for v in self.counter_box)
        bbox_color = COLOR_BOARD_TEXT if ann.counter_visible else COLOR_NOT_VIS
        cv2.rectangle(board_img, (bx1, by1), (bx2, by2), bbox_color, 1)

        # Counter LEDs
        for i, (cx, cy) in enumerate(self.counter_pos):
            cx, cy = cx + m, cy + m
            if not ann.counter_visible:
                color = COLOR_NOT_VIS
            elif ann.counter_leds[i]:
                color = COLOR_ON
            else:
                color = COLOR_OFF
            cv2.circle(board_img, (cx, cy), LED_RADIUS_PX, color, 1)

        # Ring LEDs
        for i, (rx, ry) in enumerate(self.ring_pos):
            rx, ry = rx + m, ry + m
            if ann.ring_start == ann.ring_end:
                color = COLOR_NOT_VIS
            else:
                in_arc = self._led_in_ring_arc(i, ann.ring_start, ann.ring_end)
                color = COLOR_ON if in_arc else COLOR_OFF
            if self.mode == Mode.RING_AWAITING_END and i == self._ring_start_candidate:
                color = COLOR_RING_SEL
            cv2.circle(board_img, (rx, ry), LED_RADIUS_PX, color, 1)

        # The left panel is already at the common height; scale the board to match.
        left_scaled = left
        board_scaled = cv2.resize(board_img, (h, h))

        self.left_panel_w = left_scaled.shape[1]
        self.board_scale = h / cs

        # Board labels go on after the resize, so they are drawn at display
        # resolution rather than magnified along with the board.
        def board_text(text, x, y, color):
            pos = (round(x * self.board_scale), round(y * self.board_scale))
            draw_text(board_scaled, text, pos, color)

        marker_state = f"ArUco ID {ann.aruco_id}" if ann.aruco_visible else "ArUco: hidden"
        aruco_label = f"{self.camera.value.upper()}  {self.board.profile.name}  {marker_state}"
        board_text(aruco_label, ax1, ay1 - 8, (128, 128, 128))
        counter_text = f"Counter: {ann.counter_value}" if ann.counter_visible else "Counter: n/a"
        board_text(counter_text, bx1, by1 - 8, COLOR_BOARD_TEXT)
        for text, cx, cy, color in corner_labels:
            board_text(text, cx, cy, color)

        self._draw_loupe(left_scaled, original)

        # Status bar (3 rows). Left of the panel split: progress, shortcuts, legend.
        # Right of it, under the board: the reference-clock fit and status messages.
        row_h = LINE_H
        status_h = row_h * 3 + 10
        total_w = left_scaled.shape[1] + board_scaled.shape[1]
        status_bar = np.full((status_h, total_w, 3), COLOR_STATUS_BG, dtype=np.uint8)
        right_x = left_scaled.shape[1] + 8

        # Row 1: progress + file path, plus this frame's annotated state on the right
        n_annotated = len(self.ground_truth["images"])
        progress = f"[{self.current_idx + 1}/{len(self.frames)}] ({n_annotated} annotated)"
        info = f"{progress}  {frame_key}"
        draw_text(status_bar, info, (8, row_h - 10), COLOR_TEXT, fit_scale(info, left_w - 16))

        is_annotated = frame_key in self.ground_truth["images"]
        annotation_label = "Annotated" if is_annotated else "UNANNOTATED"
        thickness = 1 if is_annotated else 2
        (tw, _), _ = cv2.getTextSize(annotation_label, FONT, FONT_SCALE, thickness)
        draw_text(
            status_bar,
            annotation_label,
            (total_w - tw - 10, row_h - 10),
            (0, 255, 0) if is_annotated else (0, 0, 255),
            thickness=thickness,
        )

        # Row 2: keyboard shortcuts, shrunk if they would reach past the panel split
        shortcuts = (
            "Enter/Space=Accept  Left/Right=Prev/Next  N/B=Skip to unannotated  "
            ",/.=Prev/Next file  0-N=Pick corner  M=Camera mode  C=Clear  Q=Quit  H=Help"
        )
        draw_text(
            status_bar,
            shortcuts,
            (8, row_h * 2 - 10),
            (80, 80, 80),
            fit_scale(shortcuts, left_w - 16),
        )

        # Row 3: color legend
        legend_y = row_h * 3 - 10
        legend_items = [
            (COLOR_ON, "ON"),
            (COLOR_OFF, "OFF"),
            (COLOR_NOT_VIS, "Hidden"),
            (COLOR_RING_SEL, "Selection"),
        ]
        lx = 8
        for color, label in legend_items:
            cv2.circle(status_bar, (lx + 8, legend_y - 6), 7, color, -1)
            draw_text(status_bar, label, (lx + 24, legend_y), color)
            lx += 24 + cv2.getTextSize(label, FONT, FONT_SCALE, 1)[0][0] + 22

        # How this frame's timestamp sits against the clock the rest of its video implies
        for i, (text, color) in enumerate(self._clock_overlay(frame_key), start=1):
            draw_text(status_bar, text, (right_x, row_h * i - 10), color)

        if self.status_msg:
            draw_text(
                status_bar,
                self.status_msg,
                (right_x, row_h * 3 - 10),
                (0, 128, 0),
                fit_scale(self.status_msg, board_scaled.shape[1] - 16),
            )

        composite = np.vstack(
            [
                np.hstack([left_scaled, board_scaled]),
                status_bar,
            ]
        )

        if self.show_help:
            self._draw_help(composite)

        return composite

    def _draw_loupe(self, panel, original):
        """Overlay a magnified inset of the region under the cursor on the scaled left panel."""
        if self._cursor_left is None:
            return
        ph, pw = panel.shape[:2]
        if pw < LOUPE_SIZE + 2 * LOUPE_MARGIN or ph < LOUPE_SIZE + 2 * LOUPE_MARGIN:
            return

        cx, cy = self._cursor_left
        # getRectSubPix replicates the border, so cursors near the edge still work.
        patch = cv2.getRectSubPix(
            original, (LOUPE_SOURCE_PX, LOUPE_SOURCE_PX), (float(cx), float(cy))
        )
        loupe = cv2.resize(patch, (LOUPE_SIZE, LOUPE_SIZE), interpolation=cv2.INTER_NEAREST)
        zoom = LOUPE_SIZE / LOUPE_SOURCE_PX
        mid = LOUPE_SIZE // 2

        # Corner markers falling inside the magnified region
        for i, c in enumerate(self.annotation.corners):
            px = round((c["position"][0] - cx) * zoom + mid)
            py = round((c["position"][1] - cy) * zoom + mid)
            if 0 <= px < LOUPE_SIZE and 0 <= py < LOUPE_SIZE:
                color = COLOR_ON if c["visible"] else COLOR_NOT_VIS
                cv2.circle(loupe, (px, py), 8, color, 1)
                if i == self.placing_idx:
                    cv2.circle(loupe, (px, py), 12, COLOR_RING_SEL, 1)
                draw_text(loupe, str(i), (px + 14, py + 6), color)

        # Crosshair marking the exact click point, with a gap so the pixel stays visible
        for dx0, dy0, dx1, dy1 in ((-16, 0, -4, 0), (4, 0, 16, 0), (0, -16, 0, -4), (0, 4, 0, 16)):
            cv2.line(loupe, (mid + dx0, mid + dy0), (mid + dx1, mid + dy1), COLOR_RING_SEL, 1)
        cv2.rectangle(loupe, (0, 0), (LOUPE_SIZE - 1, LOUPE_SIZE - 1), COLOR_RING_SEL, 2)

        # Park the inset opposite the cursor so it never hides the target
        dx = cx * self.left_scale
        dy = cy * self.left_scale
        x0 = LOUPE_MARGIN if dx > pw / 2 else pw - LOUPE_SIZE - LOUPE_MARGIN
        y0 = LOUPE_MARGIN if dy > ph / 2 else ph - LOUPE_SIZE - LOUPE_MARGIN
        panel[y0 : y0 + LOUPE_SIZE, x0 : x0 + LOUPE_SIZE] = loupe

    @staticmethod
    def _led_in_ring_arc(idx, start, end):
        """Check if LED idx is ON in the half-open arc [start, end).

        Ascending index order: start, start+1, ..., end-1.
        LED at index `end` is the first OFF and is excluded.
        """
        if start == end:
            return False  # undecodable
        if start < end:
            # Non-wrapping: ON indices are start, start+1, ..., end-1
            return start <= idx < end
        else:
            # Wrapping past 99→0: ON indices are start, ..., 99, 0, ..., end-1
            return idx >= start or idx < end

    @staticmethod
    def _draw_help(img):
        """Draw semi-transparent help overlay."""
        overlay = img.copy()
        h, w = img.shape[:2]
        pad = 20
        text_w = max(cv2.getTextSize(line, FONT, FONT_SCALE, 1)[0][0] for line in HELP_TEXT)
        box_w, box_h = text_w + 2 * pad, len(HELP_TEXT) * LINE_H + 2 * pad
        x0 = (w - box_w) // 2
        y0 = (h - box_h) // 2
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, dst=img)
        for i, line in enumerate(HELP_TEXT):
            draw_text(img, line, (x0 + pad, y0 + pad + 22 + i * LINE_H), (255, 255, 255))

    # ── Persistence ─────────────────────────────────────────────────────

    def _load_ground_truth(self):
        if self.output_path.exists():
            with open(self.output_path) as f:
                self.ground_truth = json.load(f)
            if "images" not in self.ground_truth:
                self.ground_truth = {"images": {}}
            self.ground_truth.setdefault("videos", {})
            n = len(self.ground_truth["images"])
            print(f"Loaded {n} existing annotations from {self.output_path}")

    def _report_orphans(self):
        """Say which entries no longer have an input, without touching any of them.

        The GUI walks the disk, so an entry whose file is gone is otherwise invisible
        here -- and saving would write it straight back out.
        """
        orphans = orphaned_entries(self.ground_truth, self.data_dir, self.frames)
        if orphans.is_empty():
            return
        print(f"Ground truth entries with no input in {self.data_dir}:", file=sys.stderr)
        print(describe_orphans(orphans, self.ground_truth), file=sys.stderr)
        if orphans.prunable():
            print("Run with --prune to remove them.", file=sys.stderr)

    def _save_ground_truth(self):
        self._update_references()
        with open(self.output_path, "w") as f:
            json.dump(self.ground_truth, f, indent=2)

    # ── Reference clock ─────────────────────────────────────────────────

    def _mark_video_dirty(self, frame_key_str):
        """Note that a video's annotations changed, so its reference is re-derived."""
        rel_path, index = parse_frame_key(frame_key_str)
        if index is not None:
            self._dirty_videos.add(rel_path)

    def _load_pts(self, rel_path):
        """Presentation timestamps of a video, keyed by frame index.

        Reading them demuxes the whole file, so it happens once per video and off
        the render path.
        """
        if rel_path not in self._video_pts:
            self._video_pts[rel_path] = dict(enumerate(frame_pts(self.data_dir / rel_path)))
        return self._video_pts[rel_path]

    def _load_threshold(self, rel_path):
        """(source frame period, residual tolerance) of a video, both in ms.

        The period comes from ffprobe, so like the timestamps it is read once per
        video and kept off the render path.
        """
        if rel_path not in self._video_threshold:
            period = source_frame_period_ms(self.data_dir / rel_path)
            self._video_threshold[rel_path] = (period, measured_residual_threshold_ms(period))
        return self._video_threshold[rel_path]

    def _update_references(self):
        """Re-derive the reference clock of every video that needs one.

        Gated so a frozen reference is never silently replaced: a video is derived
        only when it has none yet or one of its annotations just changed. A video
        the rest of the session never touched keeps the exact number it had.
        """
        videos = self.ground_truth.setdefault("videos", {})
        for rel_path in video_rel_paths(self.frames):
            if is_synthesized(videos, rel_path):
                continue
            if rel_path in videos and rel_path not in self._dirty_videos:
                continue
            period, threshold = self._load_threshold(rel_path)
            clock, outliers = derive_reference_clock(
                annotated_starts(self.ground_truth["images"], rel_path),
                self._load_pts(rel_path),
                threshold,
            )
            if clock is None:
                continue
            if outliers:
                # Refusing to store is the point; losing the annotation work is not
                described = describe_outliers(rel_path, outliers, threshold)
                self.status_msg = described.splitlines()[0]
                print(described, file=sys.stderr)
                continue
            videos[rel_path] = measured_video_entry(clock, period, threshold)
        self._dirty_videos.clear()

    def _clock_overlay(self, frame_key_str):
        """Lines describing how this frame's timestamp sits against its video's clock.

        The residual is leave-one-out: the clock is fitted over the video's *other*
        annotated frames and asked to predict this one, so a frame cannot drag the
        line towards itself and hide its own mistake. It reads the live annotation,
        so a correction shows up before it is saved.
        """
        rel_path, index = parse_frame_key(frame_key_str)
        pts = self._video_pts.get(rel_path)
        if index is None or not pts:
            return []

        starts = annotated_starts(self.ground_truth["images"], rel_path)
        live = reconstruct_timestamp(self.annotation.to_dict(), self.board)
        if live is None:
            starts.pop(index, None)
        else:
            starts[index] = live[0]

        lines = []
        residual = reference_residual(
            fit_reference_clock(starts, pts, exclude=index), index, starts, pts
        )
        if residual is not None:
            within = abs(residual) <= self._load_threshold(rel_path)[1]
            lines.append((f"dt {residual:+.2f} ms", (0, 128, 0) if within else (0, 0, 200)))
        clock = fit_reference_clock(starts, pts)
        if clock is not None:
            lines.append(
                (
                    f"fit {clock.n_frames_fitted} frames  RMSE {clock.rmse_ms:.2f}"
                    f"  max {clock.max_residual_ms:.2f} ms",
                    COLOR_TEXT,
                )
            )
        elif len(starts) < MIN_REFERENCE_FRAMES:
            lines.append((f"fit needs {MIN_REFERENCE_FRAMES} annotated frames", COLOR_NOT_VIS))
        return lines

    def _find_unannotated(self, from_idx, forward=True):
        """Find the next/previous unannotated frame index, wrapping around the list.

        Returns from_idx if every frame is annotated.
        """
        n = len(self.frames)
        step = 1 if forward else -1
        for k in range(1, n + 1):
            idx = (from_idx + k * step) % n
            if self.frames[idx].key not in self.ground_truth["images"]:
                return idx
        return from_idx

    def _find_source_boundary(self, from_idx, forward=True):
        """First frame of the next/previous input file, wrapping around the list.

        A video contributes one frame per key, so stepping past one otherwise costs a
        keypress per frame, and jump-to-unannotated only helps while frames are still
        unannotated.
        """
        n = len(self.frames)
        current = self.frames[from_idx].path
        step = 1 if forward else -1
        for k in range(1, n + 1):
            idx = (from_idx + k * step) % n
            if self.frames[idx].path == current:
                continue
            if forward:
                return idx
            # Going back lands on that file's last frame; rewind to its first
            found = self.frames[idx].path
            while self.frames[(idx - 1) % n].path == found:
                idx = (idx - 1) % n
            return idx
        return from_idx


def fit_clocks(data_dir, output_path):
    """Re-derive every video's reference clock without opening the GUI.

    Unlike the annotator this recomputes unconditionally -- it is an explicit request
    to re-derive, which is what to run after editing a ground truth file by hand.
    Returns a process exit code.
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path) if output_path else data_dir / "ground_truth.json"
    if not output_path.exists():
        print(f"No ground truth at {output_path}", file=sys.stderr)
        return 1

    with open(output_path) as f:
        ground_truth = json.load(f)
    ground_truth.setdefault("images", {})
    videos = ground_truth.setdefault("videos", {})

    frames = collect_frames(data_dir, sources_only=True)
    failed = False
    for rel_path in video_rel_paths(frames):
        if is_synthesized(videos, rel_path):
            continue
        period = source_frame_period_ms(data_dir / rel_path)
        threshold = measured_residual_threshold_ms(period)
        pts = dict(enumerate(frame_pts(data_dir / rel_path)))
        starts = annotated_starts(ground_truth["images"], rel_path)
        clock, outliers = derive_reference_clock(starts, pts, threshold)
        if clock is None:
            print(f"{rel_path}: fewer than {MIN_REFERENCE_FRAMES} annotated frames, skipped")
            continue
        if outliers:
            print(describe_outliers(rel_path, outliers, threshold), file=sys.stderr)
            failed = True
            continue

        previous = videos.get(rel_path)
        videos[rel_path] = measured_video_entry(clock, period, threshold)
        if previous is None:
            print(f"{rel_path}: derived {clock.clock_rate:.7f}x, {clock.clock_offset_ms:.2f} ms")
        elif any(previous.get(k) != videos[rel_path][k] for k in ("clock_rate", "clock_offset_ms")):
            print(
                f"{rel_path}: {previous['clock_rate']:.7f}x -> {clock.clock_rate:.7f}x, "
                f"{previous['clock_offset_ms']:.2f} -> {clock.clock_offset_ms:.2f} ms"
            )
        else:
            print(f"{rel_path}: unchanged")

    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    return 1 if failed else 0


# ── CLI entry point ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Annotate and verify RocSync benchmark frames")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="validation_data",
        help="Path to validation data directory (default: validation_data)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output ground truth JSON (default: <data_dir>/ground_truth.json)",
    )
    parser.add_argument(
        "--fit-clocks",
        action="store_true",
        help="Re-derive every video's reference clock and exit, without opening the GUI",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Remove ground truth entries whose image or video is gone, and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --prune, list what would be removed without writing",
    )
    args = parser.parse_args()

    if args.prune:
        sys.exit(prune(args.data_dir, args.output, dry_run=args.dry_run))

    if args.fit_clocks:
        sys.exit(fit_clocks(args.data_dir, args.output))

    tool = AnnotationTool(args.data_dir, output_path=args.output)
    tool.run()


if __name__ == "__main__":
    main()
