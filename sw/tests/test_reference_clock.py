"""The reference clock the benchmark scores against.

It is fitted by plain least squares rather than the production RANSAC, so the ground
truth cannot move when the production fitter changes, and no outlier is tolerated. The
annotator leans on the leave-one-out property pinned here: a frame's residual is
measured against a fit that does not contain it, so a single bad annotation shows its
full error instead of dragging the line towards itself and hiding half of it.
"""

import shutil
import subprocess

import pytest

from rocsync.benchmark.common import (
    MEASURED_RESIDUAL_MAX_MS,
    MEASURED_RESIDUAL_MIN_MS,
    MIN_REFERENCE_FRAMES,
    SYNTHESIZED_RESIDUAL_THRESHOLD_MS,
    ReferenceClock,
    fit_reference_clock,
    measured_residual_threshold_ms,
    reference_outliers,
    reference_residual,
    residual_threshold_ms,
    source_frame_period_ms,
)
from rocsync.timeline import frame_pts, median_frame_period

PERIOD = 500.0  # ms between frames
THRESHOLD_MS = 2.0
CLOCK_RATE = 1.000074
CLOCK_OFFSET_MS = -4333.2


def fitted(starts, pts, exclude=None):
    """`fit_reference_clock` where the caller has already ensured it succeeds."""
    clock = fit_reference_clock(starts, pts, exclude=exclude)
    assert clock is not None
    return clock


def residual(starts, pts, index, exclude=None):
    value = reference_residual(fitted(starts, pts, exclude=exclude), index, starts, pts)
    assert value is not None
    return value


def timeline(n=20):
    """Frame pts and the board times an exact clock would put them at."""
    pts = {i: i * PERIOD for i in range(n)}
    starts = {i: CLOCK_RATE * p + CLOCK_OFFSET_MS for i, p in pts.items()}
    return starts, pts


def test_fit_recovers_the_clock_it_was_generated_from():
    starts, pts = timeline()
    clock = fitted(starts, pts)

    assert clock.clock_rate == pytest.approx(CLOCK_RATE, abs=1e-9)
    assert clock.clock_offset_ms == pytest.approx(CLOCK_OFFSET_MS, abs=1e-6)
    assert clock.n_frames_fitted == 20
    assert clock.rmse_ms == pytest.approx(0.0, abs=1e-6)
    assert (clock.pts_min_ms, clock.pts_max_ms) == (0.0, 19 * PERIOD)
    assert reference_outliers(clock, starts, pts, THRESHOLD_MS) == []


def test_fit_needs_more_than_a_handful_of_frames():
    starts, pts = timeline(MIN_REFERENCE_FRAMES - 1)
    assert fit_reference_clock(starts, pts) is None


def test_fit_needs_two_distinct_presentation_timestamps():
    starts, pts = timeline()
    assert fit_reference_clock(starts, dict.fromkeys(pts, 0.0)) is None


def test_a_shifted_annotation_is_the_only_outlier():
    starts, pts = timeline()
    starts[7] += 15.0

    outliers = reference_outliers(fitted(starts, pts), starts, pts, THRESHOLD_MS)

    assert [index for index, _ in outliers] == [7]


def test_leave_one_out_shows_the_full_error_of_a_bad_frame():
    starts, pts = timeline()
    starts[7] += 15.0

    # Included, the frame drags the line towards itself and under-reports its own error
    assert abs(residual(starts, pts, 7)) < 15.0
    assert residual(starts, pts, 7, exclude=7) == pytest.approx(-15.0, abs=1e-6)
    # and it is the only frame the leave-one-out fit disagrees with
    for index in starts:
        if index != 7:
            assert abs(residual(starts, pts, index, exclude=index)) < THRESHOLD_MS


def test_round_trips_through_a_dict():
    clock = fitted(*timeline())
    restored = ReferenceClock.from_dict(clock.to_dict(THRESHOLD_MS))

    assert restored == clock
    assert clock.to_dict(THRESHOLD_MS)["residual_threshold_ms"] == THRESHOLD_MS


# ── Per-video tolerance ─────────────────────────────────────────────────────


def test_the_measured_tolerance_is_a_third_of_a_source_frame():
    assert measured_residual_threshold_ms(33.333) == pytest.approx(11.111, abs=1e-3)


def test_the_measured_tolerance_is_clamped_at_both_ends():
    # A 240 fps camera would ask for less than the board itself resolves
    assert measured_residual_threshold_ms(1000 / 240) == MEASURED_RESIDUAL_MIN_MS
    # and a 2 fps one for more than a ring period, which would hide a counter step
    assert measured_residual_threshold_ms(500.0) == MEASURED_RESIDUAL_MAX_MS
    # An unreadable frame rate falls back to the loosest tolerance rather than the tightest
    assert measured_residual_threshold_ms(None) == MEASURED_RESIDUAL_MAX_MS


def test_a_synthesized_timeline_is_held_to_one_led_at_each_end():
    entry = {"timeline": "synthesized", "source_frame_period_ms": 33.333}
    assert residual_threshold_ms(entry) == SYNTHESIZED_RESIDUAL_THRESHOLD_MS


def test_a_measured_timeline_is_held_to_its_own_frame_rate():
    entry = {"timeline": "measured", "source_frame_period_ms": 33.333}
    assert residual_threshold_ms(entry) == pytest.approx(11.111, abs=1e-3)


def test_a_stored_tolerance_outranks_the_rule_of_the_day():
    """A frozen reference stays valid under the rule it was frozen by."""
    entry = {"timeline": "measured", "source_frame_period_ms": 33.333}
    assert residual_threshold_ms({**entry, "residual_threshold_ms": 7.5}) == 7.5


@pytest.fixture(scope="module")
def decimated_clip(tmp_path_factory):
    """Every 16th frame of a 30 fps recording, the shape the dataset's clips have."""
    directory = tmp_path_factory.mktemp("decimated")
    source, decimated = directory / "src.mp4", directory / "decimated.mp4"
    common = ["ffmpeg", "-y", "-loglevel", "error"]
    subprocess.run(
        [*common, "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30:duration=4",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )  # fmt: skip
    subprocess.run(
        [*common, "-i", str(source), "-vf", "select='not(mod(n,16))'",
         "-fps_mode", "passthrough", "-c:v", "libx264", str(decimated)],
        check=True,
    )  # fmt: skip
    return decimated


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg to synthesize the clip")
def test_the_source_frame_period_survives_decimation(decimated_clip):
    """The clip's timestamps sit 533 ms apart, but its packets still say 33.3 ms."""
    assert median_frame_period(frame_pts(decimated_clip)) == pytest.approx(533, abs=1)
    assert source_frame_period_ms(decimated_clip) == pytest.approx(1000 / 30, abs=0.1)
