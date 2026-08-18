"""The reference clock the benchmark scores against.

It is fitted by plain least squares rather than the production RANSAC, so the ground
truth cannot move when the production fitter changes, and no outlier is tolerated. The
annotator leans on the leave-one-out property pinned here: a frame's residual is
measured against a fit that does not contain it, so a single bad annotation shows its
full error instead of dragging the line towards itself and hiding half of it.
"""

import pytest

from rocsync.benchmark.common import (
    MIN_REFERENCE_FRAMES,
    REFERENCE_RESIDUAL_THRESHOLD_MS,
    ReferenceClock,
    fit_reference_clock,
    reference_outliers,
    reference_residual,
)

PERIOD = 500.0  # ms between frames
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
    assert reference_outliers(clock, starts, pts) == []


def test_fit_needs_more_than_a_handful_of_frames():
    starts, pts = timeline(MIN_REFERENCE_FRAMES - 1)
    assert fit_reference_clock(starts, pts) is None


def test_fit_needs_two_distinct_presentation_timestamps():
    starts, pts = timeline()
    assert fit_reference_clock(starts, dict.fromkeys(pts, 0.0)) is None


def test_a_shifted_annotation_is_the_only_outlier():
    starts, pts = timeline()
    starts[7] += 15.0

    outliers = reference_outliers(fitted(starts, pts), starts, pts)

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
            assert (
                abs(residual(starts, pts, index, exclude=index)) < REFERENCE_RESIDUAL_THRESHOLD_MS
            )


def test_round_trips_through_a_dict():
    clock = fitted(*timeline())
    restored = ReferenceClock.from_dict(clock.to_dict())

    assert restored == clock
    assert clock.to_dict()["residual_threshold_ms"] == REFERENCE_RESIDUAL_THRESHOLD_MS
