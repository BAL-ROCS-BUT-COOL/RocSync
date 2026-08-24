"""The reference clock the benchmark scores against.

It is fitted by plain least squares rather than the production RANSAC, so the ground
truth cannot move when the production fitter changes, and no outlier is tolerated. The
annotator leans on the leave-one-out property pinned here: a frame's residual is
measured against a fit that does not contain it, so a single bad annotation shows its
full error instead of dragging the line towards itself and hiding half of it.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

import rocsync.benchmark
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


def test_the_measured_tolerance_is_clamped_at_both_ends():
    # A camera fast enough that a whole frame is below what the board itself resolves
    assert measured_residual_threshold_ms(MEASURED_RESIDUAL_MIN_MS) == MEASURED_RESIDUAL_MIN_MS
    # and a 2 fps one for more than a ring period, which would hide a counter step
    assert measured_residual_threshold_ms(500.0) == MEASURED_RESIDUAL_MAX_MS
    # An unreadable frame rate falls back to the loosest tolerance rather than the tightest
    assert measured_residual_threshold_ms(None) == MEASURED_RESIDUAL_MAX_MS


def test_a_synthesized_timeline_is_held_to_one_led_at_each_end():
    entry = {"timeline": "synthesized", "source_frame_period_ms": 33.333}
    assert residual_threshold_ms(entry) == SYNTHESIZED_RESIDUAL_THRESHOLD_MS


def test_a_measured_timeline_is_held_to_its_own_frame_rate():
    entry = {"timeline": "measured", "source_frame_period_ms": 33.333}
    assert residual_threshold_ms(entry) == measured_residual_threshold_ms(33.333)


def test_a_stored_tolerance_outranks_the_rule_of_the_day():
    """A frozen reference stays valid under the rule it was frozen by."""
    entry = {"timeline": "measured", "source_frame_period_ms": 33.333}
    assert residual_threshold_ms({**entry, "residual_threshold_ms": 7.5}) == 7.5


PREPARE_CLIP = Path(rocsync.benchmark.__file__).parent / "prepare_clip.sh"


@pytest.fixture(scope="module")
def decimated_clip(tmp_path_factory):
    """A 30 fps recording cut down to ~2 fps, the shape the dataset's clips have."""
    directory = tmp_path_factory.mktemp("decimated")
    source, decimated = directory / "src.mp4", directory / "decimated.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30:duration=4",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )  # fmt: skip
    subprocess.run([str(PREPARE_CLIP), str(source), "1.9", str(decimated)], check=True)
    return source, decimated


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg to synthesize the clip")
def test_the_source_frame_period_survives_decimation(decimated_clip):
    """The clip's own frames sit 533 ms apart, and it still reports the 30 fps it was cut from.

    The tolerance scales from the recording's frame rate, so a clip that could only report
    its own would be held to a bound 16 times too loose to catch anything.
    """
    _, clip = decimated_clip

    assert median_frame_period(frame_pts(clip)) == pytest.approx(533, abs=1)
    assert source_frame_period_ms(clip) == pytest.approx(1000 / 30, abs=0.1)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg to synthesize the clip")
def test_a_prepared_clip_keeps_the_timestamps_it_was_cut_from(decimated_clip):
    """The frames it kept sit where they sat in the recording.

    An encoder left to itself rounds every timestamp onto the grid of the frame rate it
    guesses, which on a real recording moves a frame by up to half a source frame -- more
    than the tolerance a clock fitted to those frames is held to.
    """
    source, clip = decimated_clip
    recorded, kept = frame_pts(source), frame_pts(clip)

    assert len(kept) > 5
    for pts in kept:
        assert min(abs(pts - t) for t in recorded) == pytest.approx(0.0, abs=0.01)
