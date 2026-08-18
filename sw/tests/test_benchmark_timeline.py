"""The benchmark's video clock: fitting it, and scoring it against the reference.

`summarize_timeline` is the production summarization the benchmark reuses, so the
first case pins that the extraction still recovers a planted clock. The rest cover
what `rocsync-evaluate` reports about it: an error is measured against the reference
frozen in the ground truth, never against a fit recomputed here.
"""

import pytest

from rocsync.benchmark.common import (
    descriptive_stats,
    fit_reference_clock,
    frame_key,
    retimed_videos,
)
from rocsync.benchmark.evaluate import (
    aggregate_clock_metrics,
    compute_clock_metrics,
    resolve_retimed_keys,
)
from rocsync.timeline import summarize_timeline

VIDEO = "clips/take.mp4"
RETIMED = "clips/take.retimed.mp4"
TRIMMED = 4  # leading frames the retimed clip does not span
PERIOD = 500.0  # ms between frames, i.e. 2 fps
FPS = 2.0
CLOCK_RATE = 1.000074
CLOCK_OFFSET_MS = -4333.2
THRESHOLD_MS = 2.0
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

    ground_truth = {
        "images": images,
        "videos": {VIDEO: {**reference.to_dict(THRESHOLD_MS), "timeline": "measured"}},
    }
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


def retimed_dataset(**kwargs):
    """The same run, scored through a retimed clip that carries no annotations.

    Its frames are the recording's from `TRIMMED` on, renumbered from zero, which is
    what the stored mapping has to undo. The annotations stay exactly where they were.
    """
    ground_truth, benchmark = dataset(**kwargs)
    ground_truth["videos"] = {
        RETIMED: {
            **ground_truth["videos"][VIDEO],
            "timeline": "synthesized",
            "source": VIDEO,
            "source_frame_offset": TRIMMED,
        }
    }
    benchmark["videos"] = {RETIMED: benchmark["videos"][VIDEO]}
    source_images = benchmark["images"]
    benchmark["images"] = {
        frame_key(RETIMED, index - TRIMMED): source_images[frame_key(VIDEO, index)]
        for index in range(TRIMMED, len(source_images))
    }
    return ground_truth, resolve_retimed_keys(benchmark, retimed_videos(ground_truth))


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
    assert len(metrics["residuals_ms"]) == 20


def test_a_shifted_clock_is_off_by_that_shift_everywhere():
    ground_truth, benchmark = dataset(offset_error_ms=12.5)

    metrics = compute_clock_metrics(benchmark, ground_truth)[VIDEO]

    assert metrics["clock_offset_error_ms"] == pytest.approx(12.5)
    # A pure offset error does not decay along the recording
    assert metrics["sync_error_first_ms"] == pytest.approx(12.5)
    assert metrics["sync_error_last_ms"] == pytest.approx(12.5)
    assert descriptive_stats(metrics["residuals_ms"])["mean"] == pytest.approx(12.5, abs=0.5)


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
        VIDEO: {
            "group": "measured",
            "status": "no video timeline recorded",
            "timeline": "measured",
            "residual_threshold_ms": THRESHOLD_MS,
        }
    }


def test_a_retimed_clip_is_scored_through_the_annotations_it_was_cut_from():
    ground_truth, benchmark = retimed_dataset()

    metrics = compute_clock_metrics(benchmark, ground_truth)

    assert VIDEO not in metrics  # scored once, through the clip that stands in for it
    assert metrics[RETIMED]["timeline"] == "synthesized"
    assert metrics[RETIMED]["sync_error_max_ms"] == pytest.approx(0.0, abs=1e-9)
    # The frames the trim cut have no prediction, so they are skipped rather than missed
    assert len(metrics[RETIMED]["residuals_ms"]) == 20 - TRIMMED


def test_a_misdecode_is_attributed_through_the_retiming():
    """A wrong read has to land on the annotation it contradicts, not on its neighbour."""
    ground_truth, benchmark = retimed_dataset(misdecoded=(7,), rejected=(9,))

    metrics = compute_clock_metrics(benchmark, ground_truth)[RETIMED]

    assert metrics["false_acceptances"] == 1  # frame 7 was misdecoded but kept
    assert metrics["false_rejections"] == 1  # frame 9 decoded correctly but was thrown out
    assert metrics["n_flagged"] == 20 - TRIMMED


def test_aggregation_splits_measured_from_retimed_videos():
    """The two timelines answer different questions, so they never share a row."""
    measured_gt, measured_bm = dataset()
    retimed_gt, retimed_bm = retimed_dataset()
    per_video = {
        **compute_clock_metrics(measured_bm, measured_gt),
        **compute_clock_metrics(retimed_bm, retimed_gt),
    }

    groups = aggregate_clock_metrics(per_video)

    assert set(groups) == {"measured", "retimed"}
    assert groups["measured"]["residual_vs_gt_ms"]["n"] == 20
    assert groups["retimed"]["residual_vs_gt_ms"]["n"] == 20 - TRIMMED


def test_aggregation_keeps_the_worst_video_and_sums_the_frame_counts():
    per_video = {}
    for name, offset_error_ms in (("a", 4.0), ("b", -12.5)):
        ground_truth, benchmark = dataset(offset_error_ms=offset_error_ms, rejected=(9,))
        per_video[name] = compute_clock_metrics(benchmark, ground_truth)[VIDEO]

    group = aggregate_clock_metrics(per_video)["measured"]

    assert (group["n_scored"], group["n_videos"]) == (2, 2)
    assert group["clock_offset_error_ms_max_abs"] == pytest.approx(12.5)
    assert group["clock_offset_error_ms_mean_abs"] == pytest.approx(8.25)
    assert group["n_rejected_frames"] == 2  # one frame per video
    assert group["n_flagged"] == 40
    assert group["residual_vs_gt_ms"]["n"] == 40


def test_aggregation_names_the_videos_it_could_not_score():
    ground_truth, benchmark = dataset()
    del benchmark["videos"]

    group = aggregate_clock_metrics(compute_clock_metrics(benchmark, ground_truth))["measured"]

    assert (group["n_scored"], group["n_videos"]) == (0, 1)
    assert group["unscored"] == "no video timeline recorded (1)"
    assert group["sync_error_max_ms"] is None
