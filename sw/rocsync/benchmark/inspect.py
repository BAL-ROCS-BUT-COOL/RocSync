#!/usr/bin/env python3
"""Side-by-side viewer for benchmark results.

Shows, for each benchmark frame, the original input image next to the rectified
board view reconstructed from one or more `benchmark_results.json` (or
`ground_truth.json`) files -- one column per file, so a checkout's output can be
eyeballed against another checkout, or against ground truth, frame by frame.

The rectified view is reconstructed from the file's own stored corner and ArUco
positions (the same fit `rocsync-annotate` uses), not by re-running the pipeline:
a results file is a record of what a checkout saw, and this tool shows exactly
that record back.

Ground truth is read from `<data_dir>/ground_truth.json` (or `-g`) so a retimed clip's
frames -- keyed to the recording it was cut from in every results file -- line up with the
column they belong under, whether or not the ground truth itself is shown as a column.

Usage:
    python -m rocsync.benchmark.inspect results.json [other.json ...] \
        [--data-dir DIR] [-g ground_truth.json]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from rocsync.benchmark.annotate import (
    BOARD_MARGIN_PX,
    COLOR_NOT_VIS,
    COLOR_OFF,
    COLOR_ON,
    LED_RADIUS_PX,
    counter_bbox,
    counter_led_positions,
    draw_text,
    fit_scale,
    ring_led_positions,
)
from rocsync.benchmark.common import (
    FrameRef,
    FrameSource,
    annotation_camera,
    parse_frame_key,
    retimed_videos,
)
from rocsync.benchmark.evaluate import resolve_retimed_keys
from rocsync.board_profiles import DEFAULT_BOARD_SIZE, PROFILES_BY_ARUCO
from rocsync.vision import ARUCO_DICTIONARY

WINDOW_NAME = "RocSync Inspect"

PANEL_SIDE = DEFAULT_BOARD_SIZE + 2 * BOARD_MARGIN_PX  # rectified panels are fixed at this size
HEADER_H = 52
TITLE_H = 30
DIVIDER_PX = 2

FONT_SCALE = 0.6
LABEL_SCALE = 0.65

COLOR_TEXT = (255, 255, 255)
COLOR_TITLE_BG = (60, 60, 60)
COLOR_HEADER_BG = (40, 40, 40)
COLOR_MISSING = (55, 55, 55)

TARGET_ASPECT = 16 / 9  # grid shape is chosen to make the window roughly this wide


def load_columns(paths):
    """{label: parsed json} for each results file, keyed by filename stem.

    A single directory is expanded to every `.json` file in it, matching
    `rocsync-evaluate`'s convention for the same `output/benchmark/` layout.
    """
    if len(paths) == 1 and Path(paths[0]).is_dir():
        files = sorted(Path(paths[0]).glob("*.json"))
    else:
        files = [Path(p) for p in paths]
    columns = {}
    for path in files:
        with open(path) as f:
            columns[path.stem] = json.load(f)
    return columns


def resolve_data_dir(columns, override):
    """The validation data directory: `--data-dir`, or the first `config.data_dir` found.

    A ground truth file carries no `config`, so it never supplies one -- one of the
    result files, or an explicit override, always has to.
    """
    if override is not None:
        return Path(override)
    for data in columns.values():
        data_dir = (data.get("config") or {}).get("data_dir")
        if data_dir:
            return Path(data_dir)
    return None


def frame_keys(columns):
    """Every frame key named by any column, sorted so a video's frames stay in order."""
    keys = set()
    for data in columns.values():
        keys.update((data.get("images") or {}).keys())
    return sorted(keys)


def _frame_signature(entry):
    """The decoded fields that matter for spotting a cross-column disagreement, or None."""
    if entry is None:
        return None
    aruco = entry.get("aruco") or {}
    counter = entry.get("counter") or {}
    ring = entry.get("ring") or {}
    return (
        aruco.get("visible", False),
        aruco.get("id"),
        counter.get("visible", False),
        counter.get("value"),
        ring.get("start", 0),
        ring.get("end", 0),
    )


def has_disagreement(key, columns):
    """Whether any two columns decode this frame differently (including one missing it)."""
    sigs = {_frame_signature((data.get("images") or {}).get(key)) for data in columns.values()}
    return len(sigs) > 1


def _find_source_boundary(keys, from_idx, forward=True):
    """Index of the first frame of the next/previous video, wrapping around the list.

    Mirrors `AnnotationTool._find_source_boundary`: a video contributes one frame per
    key, so stepping past one otherwise costs a keypress per frame.
    """
    n = len(keys)
    current = parse_frame_key(keys[from_idx])[0]
    step = 1 if forward else -1
    for k in range(1, n + 1):
        idx = (from_idx + k * step) % n
        path = parse_frame_key(keys[idx])[0]
        if path == current:
            continue
        if not forward:
            # Going back lands on that file's last frame; rewind to its first
            found = path
            while parse_frame_key(keys[(idx - 1) % n])[0] == found:
                idx = (idx - 1) % n
        return idx
    return from_idx


def _find_disagreement(keys, columns, from_idx, forward=True):
    """Index of the next/previous frame where columns disagree, wrapping around."""
    n = len(keys)
    step = 1 if forward else -1
    for k in range(1, n + 1):
        idx = (from_idx + k * step) % n
        if has_disagreement(keys[idx], columns):
            return idx
    return from_idx


def resolve_board(entry):
    """The `RectifiedBoard` an entry's ArUco reading identifies, or None.

    `aruco.id` is the board identifier even when the marker itself is not visible -- IR
    footage never sees it -- so this looks the id up regardless of `visible`, matching
    every other resolver (`annotate.py`, `evaluate.py`, `common.annotated_board_time`).
    """
    board_id = ((entry or {}).get("aruco") or {}).get("id")
    profile = PROFILES_BY_ARUCO.get(board_id)
    return None if profile is None else profile.rectify(DEFAULT_BOARD_SIZE)


def reconstruct_view(entry, image, margin=BOARD_MARGIN_PX):
    """Rectified color view for one column's entry: (view, board, failure reason).

    Uses the entry's own stored `homography` as-is, never re-fit from its corner or
    ArUco positions: those are a record of what was seen, not a recipe for reproducing
    the fit, and re-deriving it here would show a homography the annotation or pipeline
    run never actually used (and, for a corner re-fit, requires 4 visible corners this
    file may not have even when its own stored fit is a good one). `view` and `reason`
    are mutually exclusive.
    """
    board = resolve_board(entry)
    if board is None:
        return None, None, "no aruco"

    H = entry.get("homography")
    if H is None:
        return None, board, "no fit"

    T = np.array([[1, 0, margin], [0, 1, margin], [0, 0, 1]], dtype=np.float64)
    side = board.board_size + 2 * margin
    view = cv2.warpPerspective(image, T @ np.asarray(H, dtype=np.float64), (side, side))
    return view, board, None


def _ring_arc_color(idx, start, end):
    """Color for ring LED `idx` in the half-open arc `[start, end)`, wraparound included.

    Mirrors `AnnotationTool._led_in_ring_arc`: the annotator's own reading of the arc.
    """
    if start == end:
        return COLOR_NOT_VIS
    in_arc = start <= idx < end if start < end else (idx >= start or idx < end)
    return COLOR_ON if in_arc else COLOR_OFF


def _counter_bit_colors(board, value, visible):
    """Per-LED color for the counter, decoded from `value` -- most significant bit first."""
    if not visible or value is None:
        return [COLOR_NOT_VIS] * board.counter_bits
    return [
        COLOR_ON if (value >> (board.counter_bits - 1 - i)) & 1 else COLOR_OFF
        for i in range(board.counter_bits)
    ]


def draw_overlay(view, entry, board, margin=BOARD_MARGIN_PX):
    """Draw the entry's decoded aruco/corner/counter/ring reading onto its rectified view.

    Same overlay `rocsync-annotate` shows on the right panel, so a decoded timestamp --
    or a wrong one -- reads the same way here as it does while annotating.
    """
    aruco = entry.get("aruco") or {}
    counter = entry.get("counter") or {}
    ring = entry.get("ring") or {}
    corners = entry.get("corners") or []
    camera = annotation_camera(entry)

    # ArUco marker
    ax1, ay1 = (int(v) + margin for v in board.aruco_corners_coords[0])
    ax2, ay2 = (int(v) + margin for v in board.aruco_corners_coords[2])
    if aruco.get("visible") and aruco.get("id") is not None:
        marker = cv2.aruco.generateImageMarker(ARUCO_DICTIONARY, aruco["id"], ax2 - ax1)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        region = view[ay1:ay2, ax1:ax2]
        view[ay1:ay2, ax1:ax2] = cv2.addWeighted(region, 0.5, marker_bgr, 0.5, 0)
        draw_text(view, f"ArUco {aruco['id']}", (ax1, ay1 - 8), (180, 180, 180))
    else:
        cv2.rectangle(view, (ax1, ay1), (ax2, ay2), COLOR_NOT_VIS, 2)
        cv2.line(view, (ax1, ay1), (ax2, ay2), COLOR_NOT_VIS, 2)
        cv2.line(view, (ax2, ay1), (ax1, ay2), COLOR_NOT_VIS, 2)

    # Corner LEDs, at the fit's own expected positions
    for i, (x, y) in enumerate(board.always_on_leds[camera]):
        cx, cy = int(x) + margin, int(y) + margin
        visible = i < len(corners) and corners[i].get("visible", False)
        color = COLOR_ON if visible else COLOR_NOT_VIS
        cv2.circle(view, (cx, cy), LED_RADIUS_PX + 4, color, 2)
        draw_text(view, str(i), (cx + LED_RADIUS_PX + 6, cy + 6), color)

    # Counter
    bx1, by1, bx2, by2 = (
        round(v) + margin for v in counter_bbox(counter_led_positions(board, camera))
    )
    counter_visible = counter.get("visible", False)
    cv2.rectangle(view, (bx1, by1), (bx2, by2), COLOR_TEXT if counter_visible else COLOR_NOT_VIS, 1)
    value = counter.get("value")
    for (cx, cy), color in zip(
        counter_led_positions(board, camera),
        _counter_bit_colors(board, value, counter_visible),
        strict=True,
    ):
        cv2.circle(view, (cx + margin, cy + margin), LED_RADIUS_PX, color, 1)
    counter_text = f"Counter: {value}" if counter_visible else "Counter: n/a"
    draw_text(view, counter_text, (bx1, by1 - 8), COLOR_TEXT)

    # Ring
    start, end = ring.get("start", 0), ring.get("end", 0)
    for i, (rx, ry) in enumerate(ring_led_positions(board, camera)):
        color = _ring_arc_color(i, start, end)
        cv2.circle(view, (rx + margin, ry + margin), LED_RADIUS_PX, color, 1)


def status_text(entry):
    """One-line summary of an entry's decoded values, for the panel header."""
    if entry is None:
        return "missing"
    aruco = entry.get("aruco") or {}
    counter = entry.get("counter") or {}
    ring = entry.get("ring") or {}
    parts = [f"aruco={aruco['id']}" if aruco.get("visible") else "aruco=none"]
    if "success" in entry:
        parts.append("OK" if entry["success"] else "FAIL")
    if counter.get("visible"):
        parts.append(f"cnt={counter.get('value')}")
    if ring.get("start", 0) != ring.get("end", 0):
        parts.append(f"ring={ring.get('start')}-{ring.get('end')}")
    return "  ".join(parts)


def fit_into_square(image, side):
    """`image` letterboxed onto a `side`x`side` black square, aspect ratio preserved.

    Lets the input image join the grid as a cell the same shape as every rectified
    panel, rather than needing a differently-shaped slot of its own.
    """
    h, w = image.shape[:2]
    scale = min(side / h, side / w)
    resized = cv2.resize(image, (round(w * scale), round(h * scale)))
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y0 = (side - resized.shape[0]) // 2
    x0 = (side - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def choose_grid_cols(n, cell_w, cell_h, target_aspect=TARGET_ASPECT):
    """Column count whose grid best matches `target_aspect`, ties broken by fewest gaps.

    Every cell is the same fixed size, so the window's shape is entirely a function of
    how many columns it wraps at -- chosen here instead of a plain ceil(sqrt(n)), which
    packs cells into a square regardless of what shape actually fits the screen.
    """
    best_cols, best_score = 1, None
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        aspect = (cols * cell_w) / (rows * cell_h)
        score = (abs(math.log(aspect / target_aspect)), rows * cols - n)
        if best_score is None or score < best_score:
            best_cols, best_score = cols, score
    return best_cols


def make_panel(body, header_lines, width, height):
    """A header bar over an image (or a blank body when there is none)."""
    canvas = np.zeros((HEADER_H + height, width, 3), dtype=np.uint8)
    canvas[:HEADER_H] = COLOR_HEADER_BG
    for i, (text, scale) in enumerate(header_lines):
        y = 20 + i * 22
        draw_text(canvas, text, (6, y), COLOR_TEXT, fit_scale(text, width - 12, scale))
    if body is not None:
        canvas[HEADER_H : HEADER_H + height, :width] = body
    else:
        canvas[HEADER_H:] = COLOR_MISSING
    return canvas


def build_grid(panels, cols):
    """Panels of equal size, tiled into `cols` columns wrapping to as many rows as needed.

    A leftover cell in the last row is filled in blank, so panel count need not be a
    multiple of `cols` -- e.g. 5 results tile as a 3x2 grid with one empty cell, not a
    single row 5 panels wide.
    """
    rows = math.ceil(len(panels) / cols)
    cell_h, cell_w = panels[0].shape[:2]
    blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    cells = list(panels) + [blank] * (rows * cols - len(panels))

    v_divider = np.zeros((cell_h, DIVIDER_PX, 3), dtype=np.uint8)
    row_imgs = []
    for r in range(rows):
        row_cells = cells[r * cols : (r + 1) * cols]
        row = row_cells[0]
        for cell in row_cells[1:]:
            row = np.hstack([row, v_divider, cell])
        row_imgs.append(row)

    h_divider = np.zeros((DIVIDER_PX, row_imgs[0].shape[1], 3), dtype=np.uint8)
    grid = row_imgs[0]
    for row_img in row_imgs[1:]:
        grid = np.vstack([grid, h_divider, row_img])
    return grid


def render_frame(image, key, columns, idx, total):
    """The full composite: title bar over a grid of the input image and every rectified panel."""
    input_body = fit_into_square(image, PANEL_SIDE) if image is not None else None
    panels = [make_panel(input_body, [("input", LABEL_SCALE)], PANEL_SIDE, PANEL_SIDE)]

    for label in sorted(columns):
        entry = ((columns[label].get("images") or {}).get(key)) if image is not None else None
        if image is None:
            view, reason = None, "no image"
        elif entry is None:
            view, reason = None, "missing"
        else:
            view, board, reason = reconstruct_view(entry, image)
            if view is not None:
                draw_overlay(view, entry, board)

        header = [(label, LABEL_SCALE), (status_text(entry), FONT_SCALE)]
        panel = make_panel(view, header, PANEL_SIDE, PANEL_SIDE)
        if view is None:
            reason = reason or "unknown"
            size = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            org = ((PANEL_SIDE - size[0]) // 2, HEADER_H + PANEL_SIDE // 2)
            draw_text(panel, reason, org, COLOR_TEXT, 0.8, 2)
        panels.append(panel)

    cols = choose_grid_cols(len(panels), PANEL_SIDE, HEADER_H + PANEL_SIDE)
    grid = build_grid(panels, cols)

    title = np.full((TITLE_H, grid.shape[1], 3), COLOR_TITLE_BG, dtype=np.uint8)
    draw_text(
        title,
        f"[{idx + 1}/{total}] {key}    "
        "a/d: prev/next frame   ,/.: prev/next video   b/n: prev/next disagreement   q: quit",
        (6, 21),
        COLOR_TEXT,
        FONT_SCALE,
    )
    return np.vstack([title, grid])


def run_viewer(data_dir, keys, columns):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    idx = 0
    composite = None
    window_sized = False
    with FrameSource() as source:
        while True:
            if composite is None:
                key = keys[idx]
                rel_path, frame_index = parse_frame_key(key)
                ref = FrameRef(data_dir / rel_path, frame_index, key)
                image = source.read(ref)
                composite = render_frame(image, key, columns, idx, len(keys))
                if not window_sized:
                    h, w = composite.shape[:2]
                    cv2.resizeWindow(WINDOW_NAME, w, h)
                    window_sized = True

            cv2.imshow(WINDOW_NAME, composite)
            key_code = cv2.waitKey(30) & 0xFF

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key_code == 255:
                continue
            # Arrow codes are checked before q/Q: on Linux, left arrow truncates to the
            # same byte (81) as ord("Q"), same collision `annotate.py` resolves this way.
            if key_code in (ord("d"), ord("D"), 83):  # right arrow: next frame
                idx = (idx + 1) % len(keys)
                composite = None
            elif key_code in (ord("a"), ord("A"), 81):  # left arrow: previous frame
                idx = (idx - 1) % len(keys)
                composite = None
            elif key_code in (ord("n"), ord("N")):  # next disagreement
                idx = _find_disagreement(keys, columns, idx, forward=True)
                composite = None
            elif key_code in (ord("b"), ord("B")):  # previous disagreement
                idx = _find_disagreement(keys, columns, idx, forward=False)
                composite = None
            elif key_code == ord("."):  # next video
                idx = _find_source_boundary(keys, idx, forward=True)
                composite = None
            elif key_code == ord(","):  # previous video
                idx = _find_source_boundary(keys, idx, forward=False)
                composite = None
            elif key_code in (ord("q"), ord("Q"), 27):
                break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side viewer: input image vs. rectified benchmark results"
    )
    parser.add_argument(
        "results", nargs="+", help="benchmark_results.json file(s), or a directory of them"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Validation data directory (default: read from a result file's config.data_dir)",
    )
    parser.add_argument(
        "-g",
        "--ground-truth",
        default=None,
        help="Ground truth JSON to show as an extra column "
        "(default: <data_dir>/ground_truth.json, read either way to resolve retimed clips)",
    )
    args = parser.parse_args()

    columns = load_columns(args.results)

    data_dir = resolve_data_dir(columns, args.data_dir)
    if data_dir is None:
        print("Could not determine data_dir; pass --data-dir explicitly", file=sys.stderr)
        sys.exit(1)
    if not data_dir.is_dir():
        print(f"data_dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    gt_path = Path(args.ground_truth) if args.ground_truth else data_dir / "ground_truth.json"
    ground_truth = {}
    if gt_path.is_file():
        with open(gt_path) as f:
            ground_truth = json.load(f)
    elif args.ground_truth:
        print(f"No ground truth at {gt_path}", file=sys.stderr)
        sys.exit(1)

    # Results are keyed to the retimed clip the validator walked; annotations to the recording
    retimed = retimed_videos(ground_truth)
    columns = {label: resolve_retimed_keys(data, retimed) for label, data in columns.items()}
    if args.ground_truth:
        columns["ground_truth"] = ground_truth

    keys = frame_keys(columns)
    if not keys:
        print("No frames found in the given result files", file=sys.stderr)
        sys.exit(1)

    run_viewer(data_dir, keys, columns)


if __name__ == "__main__":
    main()
