"""The benchmark's video clock: fitting it, and scoring it against the reference.

`summarize_timeline` is the production summarization the benchmark reuses, so the
first case pins that the extraction still recovers a planted clock. The rest cover
what `rocsync-evaluate` reports about it: an error is measured against the reference
frozen in the ground truth, never against a fit recomputed here.
"""

import pytest

from rocsync.benchmark.common import fit_reference_clock, frame_key
from rocsync.benchmark.evaluate import compute_clock_metrics
from rocsync.timeline import summarize_timeline

VIDEO = "clips/take.mp4"
PERIOD = 500.0  # ms between frames, i.e. 2 fps
FPS = 2.0
CLOCK_RATE = 1.000074
CLOCK_OFFSET_MS = -4333.2
EXPOSURE_MS = 8
ARUCO_ID = 21
BOARD_PERIOD = 100  # ring LEDs, so board time is counter * 100 + ring start


def board_time(index):
    return CLOCK_RATE * index * PERIOD + CLOCK_OFFSET_MS


def as_annotation(start_ms):
    """A ground truth entry whose counter and ring reconstruct to `start_ms`."""
    counter, ring_start = divmod(round(start_ms), BOARD_PERIOD)
    return {
        "aruco": {"visible": True, "id": ARUCO_ID},
        "counter": {"visible": True, "value": counter},
        "ring": {"start": ring_start, "end": (ring_start + EXPOSURE_MS) % BOARD_PERIOD},
    }


def dataset(n=20, offset_error_ms=0.0, misdecoded=(), rejected=()):
    """A ground truth and a matching results file, keyed the way the tools key them.

    `offset_error_ms` shifts the results' fitted clock away from the reference,
    `misdecoded` names frames the run read wrongly and `rejected` frames its fit threw
    out, so outlier-rejection agreement can be checked both ways.
    """
    pts = {i: i * PERIOD for i in range(n)}
    starts = {i: board_time(i) for i in range(n)}
    reference = fit_reference_clock(starts, pts)
    assert reference is not None

    images = {}
    predictions = {}
    for i in range(n):
        key = frame_key(VIDEO, i)
        images[key] = as_annotation(starts[i])
        decoded = starts[i] + (37 if i in misdecoded else 0)
        predictions[key] = {
            **as_annotation(decoded),
            "pts_ms": pts[i],
            "fit": {"residual_ms": 0.0, "inlier": i not in rejected},
        }

    ground_truth = {"images": images, "videos": {VIDEO: reference.to_dict()}}
    benchmark = {
        "images": predictions,
        "videos": {
            VIDEO: {
                "clock_rate": reference.clock_rate,
                "clock_offset_ms": reference.clock_offset_ms + offset_error_ms,
                "rmse_after": 0.4,
                "r2_after": 1.0,
                "n_considered_frames": n - len(rejected),
                "n_rejected_frames": len(rejected),
                "n_dropped_frames": 0,
                "error": None,
            }
        },
    }
    return ground_truth, benchmark


def test_summarize_timeline_recovers_a_planted_clock():
    frame_times = {i: i * PERIOD for i in range(60)}
    timestamps = {i: (board_time(i), board_time(i) + EXPOSURE_MS) for i in range(60)}

    statistics, fit, considered, rejected, gaps = summarize_timeline(
        timestamps, frame_times, len(frame_times), FPS
    )

    assert fit.clock_rate == pytest.approx(CLOCK_RATE, abs=1e-9)
    assert statistics.clock_offset_ms == pytest.approx(CLOCK_OFFSET_MS, abs=1e-6)
    assert statistics.measured_fps == pytest.approx(FPS)
    assert statistics.mean_exposure_time == pytest.approx(EXPOSURE_MS)
    assert (len(considered), len(rejected), gaps) == (60, 0, [])


def test_summarize_timeline_needs_two_timestamped_frames():
    with pytest.raises(ValueError, match="Insufficient number of timestamped frames"):
        summarize_timeline({0: (0.0, 1.0)}, {0: 0.0, 1: PERIOD}, 2, FPS)


def test_a_matching_clock_scores_no_error():
    ground_truth, benchmark = dataset()

    metrics = compute_clock_metrics(benchmark, ground_truth)[VIDEO]

    assert metrics["status"] is None
    assert metrics["clock_rate_error_ppm"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["clock_offset_error_ms"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["sync_error_max_ms"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["residual_vs_gt_ms"]["n"] == 20


def test_a_shifted_clock_is_off_by_that_shift_everywhere():
    ground_truth, benchmark = dataset(offset_error_ms=12.5)

    metrics = compute_clock_metrics(benchmark, ground_truth)[VIDEO]

    assert metrics["clock_offset_error_ms"] == pytest.approx(12.5)
    # A pure offset error does not decay along the recording
    assert metrics["sync_error_first_ms"] == pytest.approx(12.5)
    assert metrics["sync_error_last_ms"] == pytest.approx(12.5)
    assert metrics["residual_vs_gt_ms"]["mean"] == pytest.approx(12.5, abs=0.5)


def test_outlier_rejection_is_scored_against_the_annotations():
    ground_truth, benchmark = dataset(misdecoded=(3, 4), rejected=(4, 9))

    metrics = compute_clock_metrics(benchmark, ground_truth)[VIDEO]

    assert metrics["false_rejections"] == 1  # frame 9 decoded correctly but was thrown out
    assert metrics["false_acceptances"] == 1  # frame 3 was misdecoded but kept
    assert metrics["n_flagged"] == 20


def test_a_run_without_a_timeline_is_reported_rather_than_scored():
    ground_truth, benchmark = dataset()
    del benchmark["videos"]

    assert compute_clock_metrics(benchmark, ground_truth) == {
        VIDEO: {"status": "no video timeline recorded"}
    }
