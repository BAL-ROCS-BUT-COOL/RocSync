"""Parsing and resolution of the search windows given on the command line."""

import math

import pytest

from rocsync.main import parse_time
from rocsync.video import resolve_windows


@pytest.mark.parametrize(
    "time_str, expected",
    [
        ("0:00:10", 10.0),
        ("00:01:30", 90.0),
        ("1:00:10", 3610.0),
        ("end", math.inf),
        ("END", math.inf),
        ("end-0:00:30", -30.0),  # counting back from the last frame
        ("end-1:00:00", -3600.0),
        ("-0:00:10", -10.0),  # the sign applies to the whole time, not just the hours
        ("-0:01:30", -90.0),
        ("-1:00:10", -3610.0),
    ],
)
def test_parse_time(time_str, expected):
    assert parse_time(time_str) == expected


@pytest.mark.parametrize(
    "time_str", ["", "10", "0:10", "0:00:00:10", "ten", "0:-1:00", "--1:00:10", "end-"]
)
def test_parse_time_rejects_malformed_times(time_str):
    with pytest.raises(ValueError):
        parse_time(time_str)


def test_no_window_means_the_whole_file():
    assert resolve_windows(None, "unused.mp4") == [(0.0, math.inf)]
    assert resolve_windows([], "unused.mp4") == [(0.0, math.inf)]


def test_windows_are_returned_in_order():
    windows = resolve_windows([(3.0, 4.0), (1.0, 2.0)], "unused.mp4")
    assert windows == [(1.0, 2.0), (3.0, 4.0)]


def test_disjoint_windows_are_kept_apart():
    windows = [(0.0, 1.0), (2.0, 3.0), (4.0, math.inf)]
    assert resolve_windows(windows, "unused.mp4") == windows


@pytest.mark.parametrize(
    "windows, expected",
    [
        ([(0.0, 2.0), (1.0, 3.0)], [(0.0, 3.0)]),  # overlapping
        ([(0.0, 1.0), (1.0, 2.0)], [(0.0, 2.0)]),  # touching
        ([(0.0, 3.0), (1.0, 2.0)], [(0.0, 3.0)]),  # contained
        ([(1.0, 2.0), (0.0, math.inf)], [(0.0, math.inf)]),
        ([(0.0, 1.0), (0.5, 2.0), (5.0, 6.0)], [(0.0, 2.0), (5.0, 6.0)]),
    ],
)
def test_overlapping_windows_are_merged(windows, expected):
    assert resolve_windows(windows, "unused.mp4") == expected


def test_a_window_that_ends_before_it_starts_is_rejected():
    with pytest.raises(ValueError):
        resolve_windows([(3.0, 1.0)], "unused.mp4")


def test_negative_bounds_resolve_against_the_last_frame(monkeypatch):
    monkeypatch.setattr("rocsync.video.probe_last_pts_ms", lambda _: 10_000.0)

    assert resolve_windows([(-2.0, math.inf)], "video.mp4") == [(8.0, math.inf)]
    assert resolve_windows([(-4.0, -2.0)], "video.mp4") == [(6.0, 8.0)]
    # An offset reaching past the start of the file is clamped, not negative
    assert resolve_windows([(-30.0, -2.0)], "video.mp4") == [(0.0, 8.0)]


def test_a_negative_bound_needs_a_readable_last_frame(monkeypatch):
    monkeypatch.setattr("rocsync.video.probe_last_pts_ms", lambda _: None)

    with pytest.raises(ValueError):
        resolve_windows([(-2.0, math.inf)], "video.mp4")


def test_absolute_windows_never_touch_the_video(monkeypatch):
    def fail(_):
        raise AssertionError("probed the video for an absolute window")

    monkeypatch.setattr("rocsync.video.probe_last_pts_ms", fail)
    assert resolve_windows([(1.0, 2.0)], "video.mp4") == [(1.0, 2.0)]
