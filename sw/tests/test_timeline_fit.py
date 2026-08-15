"""The timeline fit must survive dropped frames.

Regressing board time on the frame index cannot: a dropped span shifts every
later index by a constant, and the fit can only absorb that by trading clock_rate
against offset, which smears the gap over the whole video. The numbers asserted
here come from a real ZED recording (5046 frames, one 27.8 s dropout at ~11.2 s)
where the index fit reported 0.858x and a first frame of -5.593 s.
"""

import pytest

from rocsync.timeline import (
    affine_from_statistics,
    detect_dropouts,
    fit_timeline,
    median_frame_period,
)

PERIOD = 1000 / 29.97  # 33.3667 ms


def board_timestamps(frame_times, offset=1234.5, clock_rate=1.0, exposure=9.0, stride=1):
    """Board readings for a set of frames, exactly on the given affine map."""
    return {
        k: (clock_rate * pts + offset, clock_rate * pts + offset + exposure)
        for i, (k, pts) in enumerate(sorted(frame_times.items()))
        if i % stride == 0
    }


def test_fit_recovers_clock_on_a_clean_recording():
    frame_times = {i: i * PERIOD for i in range(3000)}
    fit = fit_timeline(frame_times, board_timestamps(frame_times, stride=30))

    assert fit.clock_rate == pytest.approx(1.0, abs=1e-9)
    assert fit.clock_offset_ms == pytest.approx(1234.5, abs=1e-6)
    assert fit.inlier_mask.all()
    assert fit.rmse_after == pytest.approx(0.0, abs=1e-6)


def test_fit_ignores_a_dropout_instead_of_absorbing_it():
    """The regression this module exists for."""
    # 27.797 s missing after frame 326, as observed on the ZED recording
    dropout_ms = 27797.155
    pts = [i * PERIOD for i in range(327)]
    pts += [pts[-1] + dropout_ms + (j + 1) * PERIOD for j in range(4719)]
    frame_times = dict(enumerate(pts))
    timestamps = board_timestamps(frame_times, stride=30)

    fit = fit_timeline(frame_times, timestamps)

    # The container clock needs no rescaling, so the clock_rate is 1 and the offset
    # is the true one -- not the 0.858x / -5.593 s the index fit produced.
    assert fit.clock_rate == pytest.approx(1.0, abs=1e-4)
    assert fit.clock_offset_ms == pytest.approx(1234.5, abs=1.0)
    assert fit.rmse_after < 1.0
    assert fit.inlier_mask.all()

    # The gap is reported rather than spread over the recording
    period = median_frame_period(frame_times.values())
    n_gaps, n_dropped, largest_gap_ms, gaps = detect_dropouts(pts, period)
    assert n_gaps == 1
    assert n_dropped == round(dropout_ms / PERIOD)  # 833 frames
    assert largest_gap_ms == pytest.approx(dropout_ms + PERIOD, abs=1e-6)
    assert gaps[0][0] == pytest.approx(326 * PERIOD)


def test_fit_survives_many_dropouts():
    """The FusionTrack case: 4 gaps, 50.7 s of excess, 52.94% inliers before."""
    pts, t = [], 0.0
    period = 1000 / 25.77
    for segment, gap in enumerate([12_000.0, 20_300.0, 8_400.0, 10_000.0, 0.0]):
        for _ in range(100):
            pts.append(t)
            t += period
        t += gap
    frame_times = dict(enumerate(pts))
    fit = fit_timeline(frame_times, board_timestamps(frame_times, stride=2))

    assert fit.clock_rate == pytest.approx(1.0, abs=1e-4)
    assert fit.inlier_mask.mean() > 0.99

    n_gaps, _, _, _ = detect_dropouts(pts, median_frame_period(pts))
    assert n_gaps == 4


def test_fit_rejects_misread_board_timestamps():
    frame_times = {i: i * PERIOD for i in range(1500)}
    timestamps = board_timestamps(frame_times, stride=10)

    corrupted = sorted(timestamps)[::10]
    for i, k in enumerate(corrupted):
        start, end = timestamps[k]
        shift = 5000.0 if i % 2 else -5000.0
        timestamps[k] = (start + shift, end + shift)

    fit = fit_timeline(frame_times, timestamps)
    rejected = {k for k, ok in zip(fit.order, fit.inlier_mask) if not ok}

    assert rejected == set(corrupted)
    assert fit.clock_rate == pytest.approx(1.0, abs=1e-9)


def test_fit_measures_a_drifting_clock():
    """+8.6 ppm, the drift measured on the ZED recording."""
    frame_times = {i: i * PERIOD for i in range(4000)}
    fit = fit_timeline(
        frame_times, board_timestamps(frame_times, clock_rate=1.0000086, stride=30)
    )
    assert fit.clock_rate == pytest.approx(1.0000086, abs=1e-9)


def test_median_frame_period_never_returns_zero():
    """A zero residual threshold would make RANSAC reject every sample."""
    assert median_frame_period([], fallback=33.0) == 33.0
    assert median_frame_period([5.0], fallback=33.0) == 33.0
    assert median_frame_period([7.0, 7.0, 7.0], fallback=33.0) == 33.0
    assert median_frame_period([0.0, 10.0, 20.0], fallback=33.0) == 10.0
    # A gap must not drag the period away from the true frame interval
    assert median_frame_period([0.0, 10.0, 20.0, 5000.0, 5010.0]) == 10.0


def test_fit_needs_two_usable_frames():
    with pytest.raises(ValueError):
        fit_timeline({0: 0.0}, {0: (1.0, 2.0)})
    # A timestamped frame whose presentation timestamp is unknown is unusable
    with pytest.raises(ValueError):
        fit_timeline({0: 0.0}, {0: (1.0, 2.0), 5: (3.0, 4.0)}, fallback_period=33.0)


def test_fit_ignores_timestamps_without_a_presentation_time():
    frame_times = {i: i * PERIOD for i in range(100)}
    timestamps = board_timestamps(frame_times, stride=10)
    timestamps[500] = (99999.0, 99999.0)  # never read, so it has no pts

    fit = fit_timeline(frame_times, timestamps)
    assert 500 not in fit.order
    assert fit.clock_rate == pytest.approx(1.0, abs=1e-9)


def test_detect_dropouts_on_a_gapless_timeline():
    pts = [i * PERIOD for i in range(500)]
    assert detect_dropouts(pts, PERIOD) == (0, 0, 0.0, [])
    assert detect_dropouts([], PERIOD) == (0, 0, 0.0, [])
    assert detect_dropouts(pts, 0) == (0, 0, 0.0, [])


def test_affine_reads_clock_rate_and_clock_offset_ms():
    assert affine_from_statistics({"clock_rate": 1.5, "clock_offset_ms": 2.0}) == (1.5, 2.0)


def test_affine_rejects_data_from_before_the_pts_fit():
    with pytest.raises(KeyError, match="Re-run rocsync"):
        affine_from_statistics({"clock_rate": 0.858, "first_frame": -5593.0})
