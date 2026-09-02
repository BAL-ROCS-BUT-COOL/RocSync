#!/usr/bin/env python3
"""Compare benchmark results against ground truth annotations.

Loads validation result JSON files and compares them against
validation_data/ground_truth.json. Accepts either a directory (loads all .json
files in it) or one or more explicit .json filepaths. Prints detection metrics
(TPR, FPR, F1), result accuracy, and per-step timing statistics.

Positive/negative annotation is determined per-step from the ground truth
visible flags, not from a global status field.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from rocsync.benchmark.common import (
    STEP_ORDER,
    ReferenceClock,
    confusion_metrics,
    descriptive_stats,
    parse_frame_key,
    reconstruct_timestamp,
    residual_threshold_ms,
    retimed_videos,
    ring_visible,
    source_key,
)
from rocsync.board_profiles import BOARD_V1, PROFILES_BY_ARUCO

CORNER_IMAGE_SPACE_THRESHOLD_PX = 3
# A corner counts as found when it lands within one LED sampling disc of where it belongs.
CORNER_BOARD_SPACE_THRESHOLD_PX = BOARD_V1.rectify().led_sample_radius
LABEL_WIDTH_DEFAULT = 40


# Map pipeline steps to ground truth visibility checks.
# Each returns True if the ground truth considers that step's output "positive".
STEP_GT_POSITIVE = {
    "aruco": lambda gt: gt.get("aruco", {}).get("visible", False),
    "corners": lambda gt: any(c.get("visible", False) for c in gt.get("corners", [])),
    "counter": lambda gt: gt.get("counter", {}).get("visible", False),
    "ring": ring_visible,
    # Defined below, since deciding this needs the annotation's board geometry
    "overall": lambda gt: _timestamp_extractable(gt),
}

# Map pipeline steps to prediction visibility checks.
STEP_PRED_POSITIVE = {
    "aruco": lambda p: p.get("aruco", {}).get("visible", False),
    "corners": lambda p: any(c.get("visible", False) for c in p.get("corners", [])),
    "counter": lambda p: p.get("counter", {}).get("visible", False),
    "ring": ring_visible,
    "overall": lambda p: p.get("success", False),
}


def load_benchmarks(paths):
    """Load benchmark JSON files, keyed by stem name.

    If *paths* is a single directory, loads all .json files from it.
    Otherwise, each element is treated as a filepath to a .json file.
    """
    benchmarks = {}
    if len(paths) == 1 and Path(paths[0]).is_dir():
        files = sorted(Path(paths[0]).glob("*.json"))
    else:
        files = [Path(p) for p in paths]
    for path in files:
        with open(path) as f:
            benchmarks[path.stem] = json.load(f)
    return benchmarks


def load_ground_truth(path):
    """Load ground truth JSON file."""
    with open(path) as f:
        return json.load(f)


def coverage_report(gt_images, benchmarks):
    """Which ground truth frames each column scores, and where the columns disagree.

    Returns `(scored, uncovered, missing)`: the key set per column, the ground truth
    frames no column scores at all, and per column the frames another column scores and
    it does not. A prediction a run never made is skipped by every metric below, so a
    column with a different key set answers a different question than its neighbours.
    """
    scored = {method: set(gt_images) & set(bm["images"]) for method, bm in benchmarks.items()}
    covered = set().union(*scored.values()) if scored else set()
    missing = {method: covered - keys for method, keys in scored.items() if covered - keys}
    return scored, set(gt_images) - covered, missing


def describe_keys(keys, indent="    "):
    """One line per input file behind a set of frame keys, with how many it contributes."""
    counts = Counter(parse_frame_key(key)[0] for key in keys)
    return "\n".join(f"{indent}{path}: {n} frame(s)" for path, n in sorted(counts.items()))


def resolve_retimed_keys(benchmark, retimed):
    """Move a run's per-frame predictions onto the keys the annotations use.

    A retimed clip carries no annotations of its own, so a prediction about its frame
    j is a prediction about frame j + offset of the recording it was cut from. Doing
    this once at load leaves every per-frame metric comparing keys as it always has.
    The `videos` section needs no such move: both sides key it by the retimed clip.
    """
    if not retimed:
        return benchmark
    images = {source_key(key, retimed): value for key, value in benchmark["images"].items()}
    return {**benchmark, "images": images}


# ── Geometry helpers ─────────────────────────────────────────────────────────


def _board_for_gt(gt):
    """Rectified board for a ground truth entry's ArUco marker ID.

    Rectified, because every comparison below is in board pixel space.
    """
    aruco_id = gt.get("aruco", {}).get("id")
    if aruco_id is not None and aruco_id in PROFILES_BY_ARUCO:
        return PROFILES_BY_ARUCO[aruco_id].rectify()
    return None


def _gt_aruco_corners(gt):
    """Derive ground-truth ArUco corners in original image space.

    Inverse-transforms the known board-space ArUco corner coordinates through
    the ground-truth homography.  Returns a (4, 2) array or None.
    """
    H = gt.get("homography")
    if H is None or not gt.get("aruco", {}).get("visible", False):
        return None
    board = _board_for_gt(gt)
    if board is None:
        return None
    inv_H = np.linalg.inv(np.array(H, dtype=np.float64))
    pts = np.array([board.aruco_corners_coords], dtype=np.float64)
    return cv2.perspectiveTransform(pts, inv_H).reshape(4, 2)


def _timestamp_extractable(gt):
    """Whether a timestamp follows from an annotation at all.

    Counter and ring both being visible is not enough: an arc that wraps the end of
    the period was exposed across a counter increment, so it names no single time.
    The pipeline refuses those, and scoring the refusal as a miss would penalise it
    for the one correct answer available.
    """
    board = _board_for_gt(gt)
    return board is not None and reconstruct_timestamp(gt, board) is not None


def _corner_positions(entry):
    """Annotated or predicted corner LED positions in image space, None per invisible corner."""
    return [
        c["position"] if c.get("visible") and c.get("position") is not None else None
        for c in entry.get("corners", [])
    ]


def _to_board_space(positions, gt):
    """Map image-space positions into the reference board grid via the GT homography.

    Both sides of the comparison go through the same transform, so a board-space
    error stays meaningful against a threshold expressed in LED sample radii.
    """
    H = gt.get("homography")
    if H is None:
        return [None] * len(positions)
    H = np.array(H, dtype=np.float64)
    out = []
    for pos in positions:
        if pos is None:
            out.append(None)
        else:
            pt = np.array([[pos]], dtype=np.float64)
            out.append(cv2.perspectiveTransform(pt, H).reshape(2).tolist())
    return out


# ── Per-step metric computation ──────────────────────────────────────────────


def compute_step_detection(benchmark_images, gt_images, step):
    """Compute TP/FP/FN/TN and derived rates for a single step.

    Corner detection uses per-corner matching against the annotated positions,
    with a threshold of CORNER_IMAGE_SPACE_THRESHOLD_PX in image coordinates.
    """
    gt_pos_fn = STEP_GT_POSITIVE[step]
    pred_pos_fn = STEP_PRED_POSITIVE[step]
    tp = fp = fn = tn = 0

    for img_key, gt in gt_images.items():
        pred = benchmark_images.get(img_key)
        if pred is None:
            continue

        if step == "corners":
            gt_corners = gt.get("corners", [])
            pred_corners = pred.get("corners", [])
            gt_positions = _corner_positions(gt)
            pred_positions = _corner_positions(pred)

            gt_any_visible = any(c.get("visible", False) for c in gt_corners)
            pred_any_visible = any(c.get("visible", False) for c in pred_corners)
            any_tp = False

            for i in range(min(len(gt_positions), len(pred_positions), 4)):
                if gt_positions[i] is None or pred_positions[i] is None:
                    continue
                err = np.linalg.norm(np.array(pred_positions[i]) - np.array(gt_positions[i]))
                if err <= CORNER_IMAGE_SPACE_THRESHOLD_PX:
                    any_tp = True

            if gt_any_visible and any_tp:
                tp += 1
            elif not gt_any_visible and pred_any_visible:
                fp += 1
            elif gt_any_visible and not any_tp:
                fn += 1
            else:
                tn += 1
        else:
            gt_positive = gt_pos_fn(gt)
            pred_positive = pred_pos_fn(pred)

            if gt_positive and pred_positive:
                tp += 1
            elif not gt_positive and pred_positive:
                fp += 1
            elif gt_positive and not pred_positive:
                fn += 1
            else:
                tn += 1

    return confusion_metrics(tp, fp, fn, tn)


def compute_aruco_metrics(benchmark_images, gt_images):
    """Compute ArUco-specific metrics: detection + corner pixel error."""
    detection = compute_step_detection(benchmark_images, gt_images, "aruco")

    mean_errors = []
    for img_key, gt in gt_images.items():
        pred = benchmark_images.get(img_key)
        if pred is None:
            continue
        if not gt.get("aruco", {}).get("visible", False):
            continue
        if not pred.get("aruco", {}).get("visible", False):
            continue

        gt_corners = _gt_aruco_corners(gt)
        pred_corners = pred.get("aruco", {}).get("corners")
        if gt_corners is None or pred_corners is None:
            continue

        pred_corners = np.array(pred_corners, dtype=np.float64).reshape(4, 2)
        dists = np.linalg.norm(pred_corners - gt_corners, axis=1)
        mean_errors.append(float(np.mean(dists)))

    return {
        "detection": detection,
        "corner_error_px": descriptive_stats(mean_errors),
    }


def compute_corner_metrics(benchmark_images, gt_images):
    """Compute corner LED metrics: image-space and board-space detection + errors."""
    detection_image = compute_step_detection(benchmark_images, gt_images, "corners")

    # Per-corner board-space detection and pixel errors in both spaces
    tp_board = fp_board = fn_board = tn_board = 0
    errors_image = []
    errors_board = []

    for img_key, gt in gt_images.items():
        pred = benchmark_images.get(img_key)
        if pred is None:
            continue

        board = _board_for_gt(gt)
        gt_corners = gt.get("corners", [])
        pred_corners = pred.get("corners", [])
        gt_positions = _corner_positions(gt)
        pred_positions = _corner_positions(pred)
        # Board space normalises the threshold to LED sample radii, independent of apparent board size
        gt_board = _to_board_space(gt_positions, gt)
        pred_board = _to_board_space(pred_positions, gt)
        n_corners = min(len(gt_corners), len(pred_corners))

        for i in range(n_corners):
            gt_vis = gt_corners[i].get("visible", False)
            pred_vis = pred_corners[i].get("visible", False)
            both = gt_positions[i] is not None and pred_positions[i] is not None

            # Image-space pixel error (on TPs where both are visible with positions)
            if both:
                err_img = np.linalg.norm(np.array(pred_positions[i]) - np.array(gt_positions[i]))
                errors_image.append(float(err_img))

            if both and board is not None and gt_board[i] is not None:
                err_board = np.linalg.norm(np.array(pred_board[i]) - np.array(gt_board[i]))
                errors_board.append(float(err_board))
                if err_board <= CORNER_BOARD_SPACE_THRESHOLD_PX:
                    tp_board += 1
                else:
                    fn_board += 1
            elif gt_vis:
                fn_board += 1
            elif pred_vis:
                fp_board += 1
            else:
                tn_board += 1

    return {
        "detection_image": detection_image,
        "detection_board": confusion_metrics(tp_board, fp_board, fn_board, tn_board),
        "error_image_px": descriptive_stats(errors_image),
        "error_board_px": descriptive_stats(errors_board),
    }


def compute_counter_metrics(benchmark_images, gt_images):
    """Compute counter detection + value accuracy."""
    detection = compute_step_detection(benchmark_images, gt_images, "counter")

    n_compared = 0
    n_correct = 0
    for img_key, gt in gt_images.items():
        pred = benchmark_images.get(img_key)
        if pred is None:
            continue
        if not gt.get("counter", {}).get("visible", False):
            continue
        if not pred.get("counter", {}).get("visible", False):
            continue
        n_compared += 1
        if pred.get("counter", {}).get("value") == gt.get("counter", {}).get("value"):
            n_correct += 1

    return {
        "detection": detection,
        "value_correct": n_correct,
        "value_compared": n_compared,
    }


def compute_ring_metrics(benchmark_images, gt_images):
    """Compute ring detection + start/end value accuracy."""
    detection = compute_step_detection(benchmark_images, gt_images, "ring")

    n_compared = 0
    start_correct = 0
    end_correct = 0
    for img_key, gt in gt_images.items():
        pred = benchmark_images.get(img_key)
        if pred is None:
            continue
        if not ring_visible(gt) or not ring_visible(pred):
            continue
        n_compared += 1
        if pred.get("ring", {}).get("start") == gt.get("ring", {}).get("start"):
            start_correct += 1
        if pred.get("ring", {}).get("end") == gt.get("ring", {}).get("end"):
            end_correct += 1

    return {
        "detection": detection,
        "start_correct": start_correct,
        "end_correct": end_correct,
        "value_compared": n_compared,
    }


def compute_overall_metrics(benchmark_images, gt_images):
    """Compute overall detection + timestamp accuracy + error stats."""
    detection = compute_step_detection(benchmark_images, gt_images, "overall")

    n_compared = 0
    n_correct = 0
    start_errors = []
    end_errors = []
    exposure_errors = []

    for img_key, gt in gt_images.items():
        pred = benchmark_images.get(img_key)
        if pred is None:
            continue
        if not STEP_GT_POSITIVE["overall"](gt) or not STEP_PRED_POSITIVE["overall"](pred):
            continue

        board = _board_for_gt(gt)
        if board is None:
            continue

        n_compared += 1
        gt_ts = reconstruct_timestamp(gt, board)
        pred_ts = reconstruct_timestamp(pred, board)
        if pred_ts == gt_ts:
            n_correct += 1
        if pred_ts is not None and gt_ts is not None:
            start_errors.append(abs(pred_ts[0] - gt_ts[0]))
            end_errors.append(abs(pred_ts[1] - gt_ts[1]))
            pred_exposure = pred_ts[1] - pred_ts[0]
            gt_exposure = gt_ts[1] - gt_ts[0]
            exposure_errors.append(abs(pred_exposure - gt_exposure))

    return {
        "detection": detection,
        "timestamp_correct": n_correct,
        "timestamp_compared": n_compared,
        "timestamp_error": {
            "start": descriptive_stats(start_errors),
            "end": descriptive_stats(end_errors),
            "exposure": descriptive_stats(exposure_errors),
        },
    }


# ── Clock fit ────────────────────────────────────────────────────────────────


def compute_clock_metrics(benchmark, gt) -> dict[str, dict]:
    """Score each video's fitted clock against the reference frozen in the ground truth.

    The reference is read, never re-fitted: deriving it here with the code under test
    would let a change to that code move the ground truth along with the errors
    measured against it.
    """
    references = gt.get("videos", {})
    predictions = benchmark.get("videos")
    if not references:
        return {}

    def described(reference, **fields):
        """Metrics for one video, always carrying what the reference itself says."""
        return {
            # A reference cut from another video's annotations describes a retimed clip
            "group": "retimed" if reference.get("source") else "measured",
            "timeline": reference.get("timeline", "measured"),
            "residual_threshold_ms": residual_threshold_ms(reference),
            **fields,
        }

    if predictions is None:
        return {
            rel_path: described(reference, status="no video timeline recorded")
            for rel_path, reference in references.items()
        }

    # A recording whose retimed clip this run scored is not separately unscored
    shadowed = {
        reference["source"]
        for path, reference in references.items()
        if reference.get("source") and path in predictions
    }

    gt_images = gt["images"]
    bm_images = benchmark["images"]
    metrics: dict[str, dict] = {}
    for rel_path, reference in references.items():
        if rel_path in shadowed:
            continue
        pred = predictions.get(rel_path)
        if pred is None:
            metrics[rel_path] = described(reference, status="not in this run")
            continue
        if pred.get("error") is not None:
            metrics[rel_path] = described(reference, status=pred["error"])
            continue

        ref = ReferenceClock.from_dict(reference)
        rate, offset = pred["clock_rate"], pred["clock_offset_ms"]

        def predict(pts, rate=rate, offset=offset):
            return rate * pts + offset

        first = predict(ref.pts_min_ms) - float(ref.predict(ref.pts_min_ms))
        last = predict(ref.pts_max_ms) - float(ref.predict(ref.pts_max_ms))

        # Per-frame accuracy against the annotations themselves, and whether the fit's
        # own outlier rejection agrees with them
        residuals = []
        false_rejections = false_acceptances = n_flagged = 0
        annotated_path = reference.get("source", rel_path)  # retimed clips borrow theirs
        for key, annotation in gt_images.items():
            path, index = parse_frame_key(key)
            if index is None or path != annotated_path:
                continue
            prediction = bm_images.get(key)
            if prediction is None:
                continue

            board = _board_for_gt(annotation)
            gt_ts = reconstruct_timestamp(annotation, board) if board is not None else None
            pts = prediction.get("pts_ms")
            if gt_ts is not None and pts is not None:
                residuals.append(predict(pts) - gt_ts[0])

            fit = prediction.get("fit")
            if fit is None or gt_ts is None:
                continue
            n_flagged += 1
            decoded_correctly = reconstruct_timestamp(prediction, board) == gt_ts
            if decoded_correctly and not fit["inlier"]:
                false_rejections += 1
            elif not decoded_correctly and fit["inlier"]:
                false_acceptances += 1

        metrics[rel_path] = described(
            reference,
            status=None,
            clock_rate_error_ppm=(rate - ref.clock_rate) * 1e6,
            clock_offset_error_ms=offset - ref.clock_offset_ms,
            sync_error_first_ms=first,
            sync_error_last_ms=last,
            sync_error_max_ms=max(abs(first), abs(last)),
            residuals_ms=residuals,
            false_rejections=false_rejections,
            false_acceptances=false_acceptances,
            n_flagged=n_flagged,
            rmse_after=pred.get("rmse_after"),
            r2_after=pred.get("r2_after"),
            n_considered_frames=pred.get("n_considered_frames"),
            n_rejected_frames=pred.get("n_rejected_frames"),
            n_dropped_frames=pred.get("n_dropped_frames"),
        )
    return metrics


# ── Clock fit aggregation ────────────────────────────────────────────────────

CLOCK_GROUPS = ("measured", "retimed")

# Frame counters describe how much work a fit did, so a group reports their totals.
CLOCK_COUNT_KEYS = (
    "false_rejections",
    "false_acceptances",
    "n_flagged",
    "n_considered_frames",
    "n_rejected_frames",
    "n_dropped_frames",
)


def _mean(values):
    return float(np.mean(values))


def aggregate_clock_metrics(per_video) -> dict[str, dict]:
    """Summarize the per-video clock scores once per group of videos.

    Per-video rows say little a reader can act on once there are more than a handful of
    videos, and the two groups answer different questions: a measured timeline scores the
    whole chain including the camera's own timestamps, a retimed one only the parts under
    test. Videos are the samples for the fit errors, frames for the residuals.
    """
    groups: dict[str, list] = {}
    for metrics in per_video.values():
        groups.setdefault(metrics["group"], []).append(metrics)
    return {group: _aggregate_group(videos) for group, videos in groups.items()}


def _aggregate_group(videos):
    """Video-level spreads, frame-level totals, and residuals pooled over every frame."""
    scored = [v for v in videos if v.get("status") is None]
    unscored = Counter(v["status"] for v in videos if v.get("status") is not None)

    def over_videos(key, reduce, transform=lambda value: value):
        values = [transform(v[key]) for v in scored if v.get(key) is not None]
        return reduce(values) if values else None

    aggregate = {
        "n_videos": len(videos),
        "n_scored": len(scored),
        "unscored": "; ".join(f"{status} ({n})" for status, n in unscored.most_common()),
        "residual_threshold_ms_max": over_videos("residual_threshold_ms", max),
        "clock_rate_error_ppm_mean_abs": over_videos("clock_rate_error_ppm", _mean, abs),
        "clock_rate_error_ppm_max_abs": over_videos("clock_rate_error_ppm", max, abs),
        "clock_offset_error_ms_mean_abs": over_videos("clock_offset_error_ms", _mean, abs),
        "clock_offset_error_ms_max_abs": over_videos("clock_offset_error_ms", max, abs),
        "sync_error_mean_ms": over_videos("sync_error_max_ms", _mean),
        "sync_error_max_ms": over_videos("sync_error_max_ms", max),
        "rmse_after_mean": over_videos("rmse_after", _mean),
        "rmse_after_max": over_videos("rmse_after", max),
        "r2_after_min": over_videos("r2_after", min),
        "residual_vs_gt_ms": descriptive_stats([r for v in scored for r in v["residuals_ms"]]),
    }
    for key in CLOCK_COUNT_KEYS:
        aggregate[key] = over_videos(key, sum)
    return aggregate


# ── Timing statistics ────────────────────────────────────────────────────────


def compute_timing(benchmark_images, gt_images):
    """Compute per-step timing statistics for all / positive / negative subsets."""
    step_to_gt_key = {
        "aruco_detection": "aruco",
        "corner_detection": "corners",
        "fine_rectification": "corners",
        "counter_reading": "counter",
        "ring_reading": "ring",
    }

    def collect_values(timing_key, gt_positive_fn, subset):
        """Collect timing values filtered by ground-truth subset."""
        values = []
        for img_key, pred in benchmark_images.items():
            value = pred.get("timing", {}).get(timing_key)
            if value is None:
                continue
            gt = gt_images.get(img_key)
            if gt is None:
                continue
            gt_positive = gt_positive_fn(gt)
            if subset == "all" or (subset == "positive") == gt_positive:
                values.append(value)
        return values

    result = {}
    for subset in ["all", "positive", "negative"]:
        step_stats = {}
        for step in STEP_ORDER:
            gt_key = step_to_gt_key[step]
            step_stats[step] = descriptive_stats(
                collect_values(f"{step}_ms", STEP_GT_POSITIVE[gt_key], subset)
            )
        step_stats["total"] = descriptive_stats(
            collect_values("total_ms", STEP_GT_POSITIVE["overall"], subset)
        )
        result[subset] = step_stats

    return result


# ── Printing ─────────────────────────────────────────────────────────────────

GREEN = "\033[32m"
RESET = "\033[0m"

# Which direction wins a row. LOWER ranks by magnitude, so a signed error is judged by
# how far from zero it sits rather than by its sign.
LOWER, HIGHER = "lower", "higher"

_use_color = False


def set_color(mode, stream=None):
    """Turn ANSI coloring on, off, or on only when a terminal is reading."""
    global _use_color
    stream = stream or sys.stdout
    if mode == "always":
        _use_color = True
    elif mode == "never":
        _use_color = False
    else:
        _use_color = stream.isatty() and not os.environ.get("NO_COLOR")


def highlight(text):
    return f"{GREEN}{text}{RESET}" if _use_color else text


def _best_index(values, better):
    """The one method that beats every other on this row, or None if none stands alone.

    A tie leaves the row unmarked rather than painting every column: the mark says
    "pick this one", which a tie cannot. So does a row where only one method has a
    value — there is nothing it won against.
    """
    rank = abs if better == LOWER else (lambda v: -v)
    candidates = sorted((rank(v), i) for i, v in enumerate(values) if v is not None)
    if better is None or len(candidates) < 2 or candidates[0][0] == candidates[1][0]:
        return None
    return candidates[0][1]


def format_value(val, width=10):
    if val is None:
        return "-".rjust(width)
    if isinstance(val, float):
        return f"{val:.2f}".rjust(width)
    return str(val).rjust(width)


def format_percent(val, width=10):
    if val is None:
        return "-".rjust(width)
    return f"{val:.1%}".rjust(width)


def print_row(
    label,
    values,
    col_width,
    label_width=LABEL_WIDTH_DEFAULT,
    formatter=format_value,
    better=None,
    texts=None,
):
    """Print one labelled row of per-method values, the leading method in green."""
    best = _best_index(values, better)
    print(f"  {label:>{label_width}}", end="")
    for i, value in enumerate(values):
        text = texts[i].rjust(col_width) if texts else formatter(value, col_width)
        print(f"  {highlight(text) if i == best else text}", end="")
    print()


def print_header(methods, col_width, label_width=LABEL_WIDTH_DEFAULT, label=""):
    print(f"  {'':>{label_width}}", end="")
    for m in methods:
        print(f"  {m:>{col_width}}", end="")
    print()
    print(f"    {label.upper():-^{label_width - 2}}", end="")
    for _ in methods:
        print(f"  {'-' * col_width}", end="")
    print()


def _print_detection(methods, get_det, col_width, label_width=LABEL_WIDTH_DEFAULT):
    """Print TP/FP/FN/TN and derived rates from a detection dict."""
    for key, label, better in [
        ("tp", "TP", HIGHER),
        ("fp", "FP", LOWER),
        ("fn", "FN", LOWER),
        ("tn", "TN", HIGHER),
    ]:
        print_row(label, [get_det(m)[key] for m in methods], col_width, label_width, better=better)
    for key, label, better in [
        ("recall", "TPR (Recall)", HIGHER),
        ("fpr", "FPR", LOWER),
        ("precision", "Precision", HIGHER),
        ("f1", "F1 Score", HIGHER),
    ]:
        print_row(
            label,
            [get_det(m)[key] for m in methods],
            col_width,
            label_width,
            formatter=format_percent,
            better=better,
        )


def _print_error_stats(methods, get_stats, col_width, label_width=LABEL_WIDTH_DEFAULT):
    """Print descriptive statistics (mean/median/std/min/max/n)."""
    for stat in ["mean", "std", "min", "median", "max", "n"]:
        print_row(
            stat,
            [get_stats(m).get(stat) for m in methods],
            col_width,
            label_width,
            # The smallest error wins; a sample count is not a score, and neither is the
            # best case a method happened to get.
            better=None if stat in ("min", "n") else LOWER,
        )


def _print_value_accuracy(
    methods, get_n, get_correct, label, col_width, label_width=LABEL_WIDTH_DEFAULT
):
    """Print a single value accuracy row (e.g. '5/7 (71%)')."""
    fractions, texts = [], []
    for m in methods:
        n, nc = get_n(m), get_correct(m)
        fractions.append(nc / n if n > 0 else None)
        texts.append(f"{nc}/{n} ({nc / n:.0%})" if n > 0 else "-")
    print_row(
        label,
        fractions,
        col_width,
        label_width,
        better=HIGHER,
        texts=texts,
    )


def _print_metric_rows(methods, rows, get_video, col_width, label_width=LABEL_WIDTH_DEFAULT):
    """Print one row per (key, label, formatter, better), '-' where a column has no value."""
    for key, label, formatter, better in rows:
        print_row(
            label,
            [get_video(m).get(key) for m in methods],
            col_width,
            label_width,
            formatter=formatter,
            better=better,
        )


CLOCK_GROUP_LABELS = {"measured": "measured videos", "retimed": "retimed videos"}

# The last column says which direction wins the row, or None for a row that reports what
# a fit did rather than how well it did it.
CLOCK_AGGREGATE_ROWS = [
    ("residual_threshold_ms_max", "reference tolerance, loosest [ms]", format_value, None),
    ("clock_rate_error_ppm_mean_abs", "clock rate error, mean |err| [ppm]", format_value, LOWER),
    ("clock_rate_error_ppm_max_abs", "clock rate error, worst |err| [ppm]", format_value, LOWER),
    ("clock_offset_error_ms_mean_abs", "clock offset error, mean |err| [ms]", format_value, LOWER),
    ("clock_offset_error_ms_max_abs", "clock offset error, worst |err| [ms]", format_value, LOWER),
    ("sync_error_mean_ms", "sync error, mean over videos [ms]", format_value, LOWER),
    ("sync_error_max_ms", "sync error, worst over videos [ms]", format_value, LOWER),
    ("rmse_after_mean", "fit RMSE, mean over videos [ms]", format_value, LOWER),
    ("rmse_after_max", "fit RMSE, worst over videos [ms]", format_value, LOWER),
    ("r2_after_min", "fit R2, worst over videos", format_value, HIGHER),
    ("false_rejections", "outliers rejected in error", format_value, LOWER),
    ("false_acceptances", "misdecodes kept as inliers", format_value, LOWER),
    ("n_flagged", "frames with both a fit and an annotation", format_value, None),
    ("n_considered_frames", "frames in the fit", format_value, None),
    ("n_rejected_frames", "frames rejected by the fit", format_value, None),
    ("n_dropped_frames", "frames missing from the container", format_value, None),
]


def print_clock_report(methods, clock_metrics, col_width, label_width=LABEL_WIDTH_DEFAULT):
    """Print the clock-fit section, one aggregated block per group of videos."""
    aggregates = {m: aggregate_clock_metrics(clock_metrics[m]) for m in methods}
    groups = [g for g in CLOCK_GROUPS if any(g in aggregates[m] for m in methods)]
    if not groups:
        return

    print(f"{'=' * 100}")
    print("  CLOCK FIT (aggregated over videos)")
    print(f"{'=' * 100}")

    for group in groups:

        def get_group(m, group=group):
            return aggregates[m].get(group, {})

        print_header(methods, col_width, label_width, CLOCK_GROUP_LABELS[group])

        print(f"  {'videos scored':>{label_width}}", end="")
        for m in methods:
            aggregate = get_group(m)
            n_videos = aggregate.get("n_videos", 0)
            text = f"{aggregate['n_scored']}/{n_videos}" if n_videos else "-"
            print(f"  {text:>{col_width}}", end="")
        print()

        # Reasons are sentences, so they get their own line rather than a clipped column
        for m in methods:
            unscored = get_group(m).get("unscored")
            if unscored:
                print(f"  {'unscored':>{label_width}}  {m}: {unscored}")

        _print_metric_rows(methods, CLOCK_AGGREGATE_ROWS, get_group, col_width, label_width)

        print()
        print_header(methods, col_width, label_width, "Residual vs annotations (ms), pooled")
        _print_error_stats(
            methods,
            lambda m: get_group(m).get("residual_vs_gt_ms") or {},
            col_width,
            label_width,
        )
        print()


def print_report(methods, all_metrics, col_width, label_width=LABEL_WIDTH_DEFAULT):
    """Print the full evaluation report, grouped by pipeline step."""

    # ── ArUco detection ──────────────────────────────────────────────────
    print(f"{'=' * 100}")
    print("  ARUCO DETECTION")
    print(f"{'=' * 100}")

    print_header(methods, col_width, label_width, "Detection")
    _print_detection(
        methods, lambda m: all_metrics[m]["aruco"]["detection"], col_width, label_width
    )

    print()
    print_header(methods, col_width, label_width, "Corner error (px, image space)")
    _print_error_stats(
        methods,
        lambda m: all_metrics[m]["aruco"]["corner_error_px"],
        col_width,
        label_width,
    )

    # ── Corner LED detection ─────────────────────────────────────────────
    print(f"{'=' * 100}")
    print("  CORNER LED DETECTION")
    print(f"{'=' * 100}")

    print_header(
        methods,
        col_width,
        label_width,
        f"Detection (thres: {CORNER_IMAGE_SPACE_THRESHOLD_PX}px in image space)",
    )
    _print_detection(
        methods, lambda m: all_metrics[m]["corners"]["detection_image"], col_width, label_width
    )

    print()
    print_header(
        methods,
        col_width,
        label_width,
        f"Detection (thres: {CORNER_BOARD_SPACE_THRESHOLD_PX}px in board space)",
    )
    _print_detection(
        methods, lambda m: all_metrics[m]["corners"]["detection_board"], col_width, label_width
    )

    print()
    print_header(methods, col_width, label_width, "Pixel error — image space")
    _print_error_stats(
        methods,
        lambda m: all_metrics[m]["corners"]["error_image_px"],
        col_width,
        label_width,
    )

    print()
    print_header(methods, col_width, label_width, "Pixel error — board space")
    _print_error_stats(
        methods,
        lambda m: all_metrics[m]["corners"]["error_board_px"],
        col_width,
        label_width,
    )

    # ── Counter reading ──────────────────────────────────────────────────
    print(f"{'=' * 100}")
    print("  COUNTER READING")
    print(f"{'=' * 100}")

    print_header(methods, col_width, label_width, "Detection")
    _print_detection(
        methods, lambda m: all_metrics[m]["counter"]["detection"], col_width, label_width
    )

    print()
    print_header(methods, col_width, label_width, "Value accuracy")
    _print_value_accuracy(
        methods,
        lambda m: all_metrics[m]["counter"]["value_compared"],
        lambda m: all_metrics[m]["counter"]["value_correct"],
        "counter.value correct",
        col_width,
        label_width,
    )

    # ── Ring reading ─────────────────────────────────────────────────────
    print(f"{'=' * 100}")
    print("  RING READING")
    print(f"{'=' * 100}")

    print_header(methods, col_width, label_width, "Detection")
    _print_detection(methods, lambda m: all_metrics[m]["ring"]["detection"], col_width, label_width)

    print()
    print_header(methods, col_width, label_width, "Value accuracy")
    _print_value_accuracy(
        methods,
        lambda m: all_metrics[m]["ring"]["value_compared"],
        lambda m: all_metrics[m]["ring"]["start_correct"],
        "ring.start correct",
        col_width,
        label_width,
    )
    _print_value_accuracy(
        methods,
        lambda m: all_metrics[m]["ring"]["value_compared"],
        lambda m: all_metrics[m]["ring"]["end_correct"],
        "ring.end correct",
        col_width,
        label_width,
    )

    # ── Overall ──────────────────────────────────────────────────────────
    print(f"{'=' * 100}")
    print("  OVERALL (timestamp)")
    print(f"{'=' * 100}")

    print_header(methods, col_width, label_width, "Detection")
    _print_detection(
        methods, lambda m: all_metrics[m]["overall"]["detection"], col_width, label_width
    )

    print()
    print_header(methods, col_width, label_width, "Timestamp accuracy")
    _print_value_accuracy(
        methods,
        lambda m: all_metrics[m]["overall"]["timestamp_compared"],
        lambda m: all_metrics[m]["overall"]["timestamp_correct"],
        "timestamp exact match",
        col_width,
        label_width,
    )

    print()
    print_header(methods, col_width, label_width, "Timestamp error (ms)")
    for err_key, err_label in [
        ("start", "start error"),
        ("end", "end error"),
        ("exposure", "exposure error"),
    ]:
        print(f"  {err_label.upper():>{label_width}}")
        for stat in ["mean", "std", "min", "median", "max"]:
            values = [
                all_metrics[m]["overall"]["timestamp_error"].get(err_key, {}).get(stat)
                for m in methods
            ]
            print_row(
                "  " + stat,
                values,
                col_width,
                label_width,
                better=LOWER,
            )


def print_timing(methods, timing, col_width, label_width=LABEL_WIDTH_DEFAULT):
    subset_labels = {
        "all": "ALL IMAGES",
        "positive": "POSITIVE ANNOTATIONS (per-step)",
        "negative": "NEGATIVE ANNOTATIONS (per-step)",
    }
    stat_keys = ["mean", "std", "min", "median", "max"]

    for subset_name in ["all", "positive", "negative"]:
        any_data = any(timing[m][subset_name].get("total", {}).get("n", 0) > 0 for m in methods)
        if not any_data:
            continue

        print(f"{'=' * 100}")
        print(f"  TIMING — {subset_labels[subset_name]} (ms)")
        print(f"{'=' * 100}")
        print_header(methods, col_width, label_width, STEP_ORDER[0])

        for i, step in enumerate([*STEP_ORDER, "total"]):
            if i > 0:
                print(f"  {step.upper():-^{label_width}}")
            for stat in stat_keys:
                print_row(
                    "  " + stat,
                    [timing[m][subset_name].get(step, {}).get(stat) for m in methods],
                    col_width,
                    label_width,
                    # The best case a method got says little; the rest is speed, so faster wins
                    better=None if stat == "min" else LOWER,
                )


def _describe_run(config):
    """One-line provenance for a result file, so a column says which checkout produced it."""
    if not config:
        return "no provenance recorded"
    branch = config.get("branch") or "?"
    commit = config.get("commit") or "?"
    dirty = "+dirty" if config.get("dirty") else ""
    parts = [f"{branch}@{commit}{dirty}"]
    if config.get("run_at"):
        parts.append(config["run_at"])
    if config.get("opencv"):
        parts.append(f"cv2 {config['opencv']}")
    if config.get("numpy"):
        parts.append(f"numpy {config['numpy']}")
    if config.get("n_images") is not None:
        parts.append(f"{config['n_images']} images")
    return "  ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmark results against ground truth")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["output/benchmark"],
        help="Directory with benchmark .json files, or one or more .json filepaths "
        "(default: output/benchmark)",
    )
    parser.add_argument(
        "-g",
        "--ground-truth",
        default="validation_data/ground_truth.json",
        help="Path to ground truth JSON (default: validation_data/ground_truth.json)",
    )
    parser.add_argument(
        "-t", "--timing", action="store_true", help="Include per-step timing statistics"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Report instead of refusing when columns cover different frames, scoring "
        "each over its own coverage",
    )
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Highlight the leading method for each metric in green "
        "(default: auto, i.e. only when writing to a terminal)",
    )
    args = parser.parse_args()

    set_color(args.color)

    gt = load_ground_truth(args.ground_truth)
    gt_images = gt["images"]

    benchmarks = load_benchmarks(args.paths)
    if not benchmarks:
        print("No benchmark .json files found", file=sys.stderr)
        sys.exit(1)

    retimed = retimed_videos(gt)
    benchmarks = {m: resolve_retimed_keys(bm, retimed) for m, bm in benchmarks.items()}

    methods = list(benchmarks.keys())
    col_width = max(14, max(len(m) for m in methods) + 2)

    n_images = len(gt_images)
    scored, uncovered, missing = coverage_report(gt_images, benchmarks)
    print(f"Ground truth: {args.ground_truth} ({n_images} frames)")
    print(f"Methods: {', '.join(methods)}")
    for m in methods:
        # A prediction the run never made is skipped silently, so say how many it covers
        scope = f"scores {len(scored[m])}/{n_images}"
        print(f"  {m}: {_describe_run(benchmarks[m].get('config', {}))}, {scope}")

    if uncovered:
        print(
            f"\nWARNING: {len(uncovered)} ground truth frame(s) no column scores. If those "
            f"inputs are gone, prune them:\n"
            f"  rocsync-annotate <data_dir> --prune\n"
            f"{describe_keys(uncovered)}",
            file=sys.stderr,
        )

    if missing:
        detail = "\n".join(
            f"  {m} does not score {len(keys)} frame(s) another column does:\n{describe_keys(keys)}"
            for m, keys in sorted(missing.items())
        )
        # Every metric skips a frame a run has no prediction for, so columns over
        # different frame sets are different populations, counts and winner alike
        print(f"\nColumns cover different frames:\n{detail}", file=sys.stderr)
        if not args.allow_partial:
            print(
                "\nRe-run rocsync-validate for the columns above, or pass --allow-partial "
                "to score each column over its own coverage.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("--allow-partial: each column is scored over its own frames.", file=sys.stderr)

    # Compute all metrics per method
    all_metrics = {}
    for m in methods:
        bm = benchmarks[m]["images"]
        all_metrics[m] = {
            "aruco": compute_aruco_metrics(bm, gt_images),
            "corners": compute_corner_metrics(bm, gt_images),
            "counter": compute_counter_metrics(bm, gt_images),
            "ring": compute_ring_metrics(bm, gt_images),
            "overall": compute_overall_metrics(bm, gt_images),
        }

    print_report(methods, all_metrics, col_width)

    clock_metrics = {m: compute_clock_metrics(benchmarks[m], gt) for m in methods}
    print_clock_report(methods, clock_metrics, col_width)

    if args.timing:
        timing = {}
        for m in methods:
            timing[m] = compute_timing(benchmarks[m]["images"], gt_images)
        print_timing(methods, timing, col_width)

    print()


if __name__ == "__main__":
    main()
