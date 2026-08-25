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

Usage:
    python -m rocsync.benchmark.inspect results.json [other.json ...] \
        [--data-dir DIR] [-g ground_truth.json]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from rocsync.benchmark.annotate import (
    BOARD_MARGIN_PX,
    coarse_homography_from_aruco,
    draw_text,
    fit_corner_homography,
    fit_scale,
)
from rocsync.benchmark.common import FrameRef, FrameSource, parse_frame_key
from rocsync.board_profiles import DEFAULT_BOARD_SIZE, PROFILES_BY_ARUCO
from rocsync.camera import CameraType

WINDOW_NAME = "RocSync Inspect"

PANEL_SIDE = DEFAULT_BOARD_SIZE + 2 * BOARD_MARGIN_PX  # rectified panels are fixed at this size
HEADER_H = 52
TITLE_H = 30

FONT_SCALE = 0.6
LABEL_SCALE = 0.65

COLOR_TEXT = (255, 255, 255)
COLOR_TITLE_BG = (60, 60, 60)
COLOR_HEADER_BG = (40, 40, 40)
COLOR_MISSING = (55, 55, 55)
COLOR_LED_VISIBLE = (0, 255, 0)
COLOR_LED_HIDDEN = (0, 0, 255)


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


def resolve_board(entry):
    """The `RectifiedBoard` an entry's ArUco reading identifies, or None."""
    aruco = (entry or {}).get("aruco") or {}
    if not aruco.get("visible"):
        return None
    profile = PROFILES_BY_ARUCO.get(aruco.get("id"))
    return None if profile is None else profile.rectify(DEFAULT_BOARD_SIZE)


def reconstruct_view(entry, image, margin=BOARD_MARGIN_PX):
    """Rectified color view for one column's entry: (view, board, failure reason).

    Prefers the fit from corner LEDs -- the same one `rocsync-annotate` shows --
    and falls back to the ArUco marker alone, exactly as `rectify_board` does when
    corner detection comes up empty. `view` and `reason` are mutually exclusive.
    """
    board = resolve_board(entry)
    if board is None:
        return None, None, "no aruco"

    H = fit_corner_homography(entry.get("corners") or [], board)
    if H is None:
        H = coarse_homography_from_aruco({"aruco_corners": entry["aruco"].get("corners")}, board)
    if H is None:
        return None, board, "no fit"

    T = np.array([[1, 0, margin], [0, 1, margin], [0, 0, 1]], dtype=np.float64)
    side = board.board_size + 2 * margin
    view = cv2.warpPerspective(image, T @ np.asarray(H, dtype=np.float64), (side, side))
    return view, board, None


def draw_corner_overlay(view, entry, board, margin=BOARD_MARGIN_PX):
    """Mark each corner LED where the fit says it should land -- green if it was
    actually seen there, red if the position is only the board's expectation."""
    corners = (entry or {}).get("corners") or []
    for i, (x, y) in enumerate(board.always_on_leds[CameraType.RGB]):
        visible = i < len(corners) and corners[i].get("visible", False)
        color = COLOR_LED_VISIBLE if visible else COLOR_LED_HIDDEN
        cv2.circle(view, (int(x + margin), int(y + margin)), 5, color, 1, cv2.LINE_AA)


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


def render_frame(image, key, columns, idx, total):
    """The full composite: title bar, input panel, one rectified panel per column."""
    if image is not None:
        h, w = image.shape[:2]
        scale = PANEL_SIDE / h
        left_body = cv2.resize(image, (round(w * scale), PANEL_SIDE))
    else:
        left_body = None
    left_w = left_body.shape[1] if left_body is not None else PANEL_SIDE
    panels = [make_panel(left_body, [("input", LABEL_SCALE)], left_w, PANEL_SIDE)]

    for label in sorted(columns):
        entry = ((columns[label].get("images") or {}).get(key)) if image is not None else None
        if image is None:
            view, reason = None, "no image"
        elif entry is None:
            view, reason = None, "missing"
        else:
            view, board, reason = reconstruct_view(entry, image)
            if view is not None:
                draw_corner_overlay(view, entry, board)

        header = [(label, LABEL_SCALE), (status_text(entry), FONT_SCALE)]
        panel = make_panel(view, header, PANEL_SIDE, PANEL_SIDE)
        if view is None:
            size = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            org = ((PANEL_SIDE - size[0]) // 2, HEADER_H + PANEL_SIDE // 2)
            draw_text(panel, reason, org, COLOR_TEXT, 0.8, 2)
        panels.append(panel)

    divider = np.zeros((panels[0].shape[0], 2, 3), dtype=np.uint8)
    row = panels[0]
    for panel in panels[1:]:
        row = np.hstack([row, divider, panel])

    title = np.full((TITLE_H, row.shape[1], 3), COLOR_TITLE_BG, dtype=np.uint8)
    draw_text(
        title,
        f"[{idx + 1}/{total}] {key}    a/d: prev/next   q: quit",
        (6, 21),
        COLOR_TEXT,
        FONT_SCALE,
    )
    return np.vstack([title, row])


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
            if key_code in (ord("q"), ord("Q"), 27):
                break
            elif key_code in (ord("d"), ord("D"), 83):  # right arrow
                idx = (idx + 1) % len(keys)
                composite = None
            elif key_code in (ord("a"), ord("A"), 81):  # left arrow
                idx = (idx - 1) % len(keys)
                composite = None

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
        "-g", "--ground-truth", default=None, help="Ground truth JSON to show as an extra column"
    )
    args = parser.parse_args()

    columns = load_columns(args.results)
    if args.ground_truth:
        with open(args.ground_truth) as f:
            columns["ground_truth"] = json.load(f)

    data_dir = resolve_data_dir(columns, args.data_dir)
    if data_dir is None:
        print("Could not determine data_dir; pass --data-dir explicitly", file=sys.stderr)
        sys.exit(1)
    if not data_dir.is_dir():
        print(f"data_dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    keys = frame_keys(columns)
    if not keys:
        print("No frames found in the given result files", file=sys.stderr)
        sys.exit(1)

    run_viewer(data_dir, keys, columns)


if __name__ == "__main__":
    main()
