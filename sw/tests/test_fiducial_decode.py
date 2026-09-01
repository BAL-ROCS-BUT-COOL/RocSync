"""decode_board_points: read a board's counter and ring off points already in its own
millimetre frame, with the orientation already resolved by the caller.
"""

import numpy as np
import pytest

from rocsync.board_profiles import BOARD_V1, BOARD_V2, COUNTER_ZERO, RING
from rocsync.camera import CameraType
from rocsync.fiducial_decode import decode_board_points


def lit_points(profile, counter, ring_window):
    """Every point that should be lit for one counter/ring reading, in board mm."""
    ir = CameraType.INFRARED
    points = [profile.always_on_leds[ir]]

    bits = profile.counter_led_coords[ir]
    n = profile.counter_bits
    lit_bits = [p for i, p in enumerate(bits) if counter >> (n - 1 - i) & 1]
    if lit_bits:
        points.append(np.asarray(lit_bits))

    start, end = ring_window
    ring = profile.ring_led_coords(ir)
    points.append(np.asarray([ring[i % profile.period] for i in range(start, end + 1)]))

    return np.concatenate(points).tolist()


@pytest.mark.parametrize("profile", [BOARD_V1, BOARD_V2], ids=["v1", "v2"])
def test_decodes_counter_and_ring(profile):
    counter = 12345 if profile.counter_bits >= 20 else 4321
    ring_window = (10, 30)

    result = decode_board_points(lit_points(profile, counter, ring_window), profile)

    assert result.reject is None
    assert result.counter == counter
    assert result.ring_start == ring_window[0] + counter * profile.period
    assert result.ring_end == ring_window[1] + counter * profile.period
    assert result.board_ms == result.ring_start
    assert result.exposure_ms == ring_window[1] - ring_window[0]


def test_rejects_zero_counter():
    # No counter bits lit -- the count has not started, or an ambiguous orientation
    # search left it looking that way. Either way the ring index is meaningless.
    result = decode_board_points(lit_points(BOARD_V2, 0, (10, 30)), BOARD_V2)
    assert result.reject == COUNTER_ZERO
    assert result.board_ms == 0.0


def test_rejects_an_arc_across_the_wrap():
    # Ends at the very last ring LED: the counter could have incremented mid-exposure,
    # so board_time_from_ring refuses rather than guess which period it belongs to.
    profile = BOARD_V2
    ring_window = (profile.period - 3, profile.period - 1)
    result = decode_board_points(lit_points(profile, 12345, ring_window), profile)
    assert result.reject == RING


def test_tolerates_an_isolated_false_ring_detection():
    """A deliberate strictness difference from the ROS package this decoder replaces.

    That package's own reader (RocSync/sw's ftk.py, pre-split) required exactly one
    contiguous lit run and refused anything else. ``board_profiles.decode_ring`` -- what
    this function is built on -- instead takes the best-scoring window
    (``_optimal_ring_window``) and tolerates an isolated stray LED outside it, rather
    than refuse the whole frame over one false detection. A single lit LED far from the
    real arc is therefore read as its own one-LED window, not rejected.
    """
    profile = BOARD_V2
    ir = CameraType.INFRARED
    ring = profile.ring_led_coords(ir)
    points = np.concatenate([profile.always_on_leds[ir], [ring[10], ring[50]]]).tolist()
    counter = 12345
    bits = profile.counter_led_coords[ir]
    n = profile.counter_bits
    lit_bits = [p for i, p in enumerate(bits) if counter >> (n - 1 - i) & 1]
    points = np.concatenate([points, lit_bits]).tolist()

    result = decode_board_points(points, profile)
    assert result.reject is None
    assert result.ring_start == result.ring_end  # a one-LED window, not the pair
