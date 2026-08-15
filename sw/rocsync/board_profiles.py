"""The RocSync board, defined in millimetres.
Coordinates are LED body centres, with the origin at the top-left of the board outline
and +y pointing down (same convention as the rectified image).
``BoardProfile.rectify`` projects a profile onto a pixel grid for the camera pipeline.
The FusionTrack pipeline consumes the millimetres directly.
"""

import math
from dataclasses import dataclass

import numpy as np

from rocsync.camera import CameraType

BOARD_SIZE_MM = 250.0
DEFAULT_BOARD_SIZE = 640  # rectified image dimension (px)

# Sampling geometry, in mm so it scales with the rectified resolution.
LED_SAMPLE_RADIUS_MM = 3.0  # radius of the disc each LED's intensity is read from
RING_BG_OFFSET_MM = -10.0  # background ring, measured inward from the LED ring
LAYOUT_TOL_MM = 4.0  # how far a blob may sit from a modelled LED and still count

RING_EDGE_MARGIN = 1  # ring LEDs of slack required at either end of the period


def _ring(radius_mm, period, centre):
    """Ring LED centres (mm), index 0 at the top, running counter-clockwise."""
    angles = -(np.arange(period) / period + 0.25) * 2 * math.pi
    return np.stack(
        [centre + radius_mm * np.cos(angles), centre + radius_mm * np.sin(angles)],
        axis=1,
    )


def _optimal_ring_window(leds):
    """The half-open window [start, end) of lit ring LEDs that best explains a reading.

    Maximises the number of lit LEDs inside the window minus the number outside,
    so isolated false detections do not break up an otherwise clean run. Windows
    may wrap the end of the period, in which case start > end; start == end means
    no window scored better than the empty one.

    Args:
        leds: A list of booleans indicating the detected led state.

    Returns:
        A tuple of [start, end) indices for the optimal window.
    """
    nums = [1 if val else -1 for val in leds]
    n = len(nums)
    total_sum = sum(nums)

    # --- Find max non-wrapping subarray and its indices (Kadane's Algorithm) ---
    max_score = -float("inf")
    max_start, max_end = 0, 0
    current_max = 0
    current_start_max = 0
    for i, x in enumerate(nums):
        if current_max <= 0:
            current_start_max = i
            current_max = x
        else:
            current_max += x

        if current_max > max_score:
            max_score = current_max
            max_start = current_start_max
            max_end = i + 1

    # --- Find min non-wrapping subarray and its indices ---
    min_score = float("inf")
    min_start, min_end = 0, 0
    current_min = 0
    current_start_min = 0
    for i, x in enumerate(nums):
        if current_min >= 0:
            current_start_min = i
            current_min = x
        else:
            current_min += x

        if current_min < min_score:
            min_score = current_min
            min_start = current_start_min
            min_end = i + 1

    # A wrapping window needs at least 2 elements
    max_wrap_sum = total_sum - min_score if n > 1 else -float("inf")

    # The best window is the best non-wrapping one, the best wrapping one, or empty
    if max_score > max_wrap_sum and max_score > 0:
        return max_start, max_end
    if max_wrap_sum > 0:
        return min_end, min_start
    return 0, 0


@dataclass(frozen=True)
class BoardProfile:
    """One revision of the physical board, in millimetres."""

    name: str
    size_mm: float
    period: int
    aruco_marker_id: int
    ftk_marker_id: int
    aruco_size_mm: float  # outer edge of the marker's black border
    counter_bits: int
    ring_radius_mm: dict  # {CameraType: float}
    # LEDs lit in every frame. The first four are the perspective-transform anchors, in
    # TL, TR, BR, BL order; any further entries are extra always-on LEDs.
    always_on_leds: dict  # {CameraType: np.ndarray Nx2}, N >= 4
    counter_led_coords: dict  # {CameraType: np.ndarray Nx2}, most significant bit first
    counter_bg_y: dict  # {CameraType: float} background sample row

    def __hash__(self):
        return hash(self.name)

    @property
    def centre_mm(self):
        return self.size_mm / 2

    def transform_corners(self, camera_type):
        """The four perspective anchors (mm), clockwise from the top-left."""
        return self.always_on_leds[camera_type][:4]

    def ring_led_coords(self, camera_type, radius_offset=0.0):
        """Ring LED centres (mm), in counter order."""
        return _ring(
            self.ring_radius_mm[camera_type] + radius_offset,
            self.period,
            self.centre_mm,
        )

    def decode_counter(self, leds):
        """Counter value from its LED states, most significant bit first."""
        bits = np.asarray(leds, dtype=bool).reshape(-1)
        weights = 2 ** np.arange(self.counter_bits - 1, -1, -1)
        return int(weights[bits].sum())

    def decode_ring(self, leds):
        """First and last lit ring LED, inclusive, or None if none are lit.

        Tolerates isolated false detections; see ``_optimal_ring_window``.
        """
        start, end = _optimal_ring_window(leds)
        if start == end:
            return None
        return start % self.period, (end - 1) % self.period

    def board_time_from_ring(self, counter, ring):
        """Board time (start_ms, end_ms) for one counter value and ring reading.

        `ring` holds the first and last lit ring LED, inclusive. Returns None when
        the counter incremented during the exposure, which makes the reading
        ambiguous: either the run wraps the end of the period, or it sits within
        RING_EDGE_MARGIN LEDs of one end and may already have wrapped.
        """
        start, end = ring
        if (
            start > end
            or start < RING_EDGE_MARGIN
            or self.period - 1 - end < RING_EDGE_MARGIN
        ):
            return None
        return start + counter * self.period, end + counter * self.period

    def aruco_corners(self):
        """The marker's black-border corners (mm), clockwise from the top-left."""
        c, half = self.centre_mm, self.aruco_size_mm / 2
        return np.array(
            [
                [c - half, c - half],
                [c + half, c - half],
                [c + half, c + half],
                [c - half, c + half],
            ],
            dtype=np.float32,
        )

    def layout_coords(self, camera_type):
        """Every LED this camera type can see the board light (mm).
        This is the model the corner search scores candidates against.
        """
        return np.concatenate(
            [
                self.always_on_leds[camera_type],
                self.ring_led_coords(camera_type),
                self.counter_led_coords[camera_type],
            ]
        )

    def rectify(self, board_size=DEFAULT_BOARD_SIZE):
        """This board projected onto a ``board_size`` square pixel grid."""
        return _rectified(self, board_size)


class RectifiedBoard:
    """A ``BoardProfile`` measured in pixels of a square rectified image.

    Mirrors the profile's accessors, converted to px and to the float32 OpenCV's
    perspective helpers expect. Instances are cached per (profile, board_size), so they
    are cheap to ask for and safe to use as dictionary keys.
    """

    def __init__(self, profile, board_size):
        self.profile = profile
        self.board_size = board_size
        self.px_per_mm = board_size / profile.size_mm

        self.period = profile.period
        self.counter_bits = profile.counter_bits
        self.aruco_marker_id = profile.aruco_marker_id
        self.led_sample_radius = max(1, round(LED_SAMPLE_RADIUS_MM * self.px_per_mm))
        self.layout_tol = max(1, round(LAYOUT_TOL_MM * self.px_per_mm))

        self.always_on_leds = {
            ct: self._px(coords) for ct, coords in profile.always_on_leds.items()
        }
        self.counter_led_coords = {
            ct: self._px(coords) for ct, coords in profile.counter_led_coords.items()
        }
        self.counter_bg_y = {
            ct: y * self.px_per_mm for ct, y in profile.counter_bg_y.items()
        }
        self.aruco_corners_coords = self._px(profile.aruco_corners())

    def __repr__(self):
        return f"RectifiedBoard({self.profile.name!r}, {self.board_size})"

    def __hash__(self):
        return hash((self.profile.name, self.board_size))

    def __eq__(self, other):
        return (
            isinstance(other, RectifiedBoard)
            and other.profile.name == self.profile.name
            and other.board_size == self.board_size
        )

    def _px(self, coords_mm):
        return np.asarray(coords_mm, dtype=np.float32) * self.px_per_mm

    def transform_corners(self, camera_type):
        return self.always_on_leds[camera_type][:4]

    def ring_led_coords(self, camera_type, radius_offset_mm=0.0):
        return self._px(self.profile.ring_led_coords(camera_type, radius_offset_mm))

    def layout_coords(self, camera_type):
        return self._px(self.profile.layout_coords(camera_type))

    def decode_counter(self, leds):
        return self.profile.decode_counter(leds)

    def decode_ring(self, leds):
        return self.profile.decode_ring(leds)

    def board_time_from_ring(self, counter, ring):
        return self.profile.board_time_from_ring(counter, ring)


_RECTIFIED_CACHE = {}


def _rectified(profile, board_size):
    key = (profile.name, board_size)
    board = _RECTIFIED_CACHE.get(key)
    if board is None:
        board = _RECTIFIED_CACHE[key] = RectifiedBoard(profile, board_size)
    return board


def _counter_row(y, x0, spacing, count):
    return [(x0 + i * spacing, y) for i in range(count)]


def _build_v1():
    # 16 counter LEDs in one row, 8 mm apart.
    counter = {
        CameraType.RGB: _counter_row(53.0, 65.0, 8.0, 16),
        CameraType.INFRARED: _counter_row(48.0, 65.0, 8.0, 16),
    }
    return BoardProfile(
        name="v1",
        size_mm=BOARD_SIZE_MM,
        period=100,
        aruco_marker_id=0,
        ftk_marker_id=240,
        aruco_size_mm=91.2,
        counter_bits=16,
        ring_radius_mm={CameraType.RGB: 115.0, CameraType.INFRARED: 110.0},
        always_on_leds={
            CameraType.RGB: np.array(
                [(20.0, 20.0), (230.0, 20.0), (230.0, 230.0), (20.0, 230.0)],
                dtype=np.float32,
            ),
            CameraType.INFRARED: np.array(
                [(5.0, 5.0), (245.0, 5.0), (245.0, 245.0), (5.0, 245.0)],
                dtype=np.float32,
            ),
        },
        counter_led_coords={
            ct: np.array(rows, dtype=np.float32) for ct, rows in counter.items()
        },
        counter_bg_y={CameraType.RGB: 43.0, CameraType.INFRARED: 38.0},
    )


def _build_v2():
    # 20 counter LEDs in two rows of 10, 12 mm apart in both directions.
    counter = {
        CameraType.RGB: _counter_row(41.0, 71.0, 12.0, 10)
        + _counter_row(53.0, 71.0, 12.0, 10),
        CameraType.INFRARED: _counter_row(47.0, 71.0, 12.0, 10)
        + _counter_row(59.0, 71.0, 12.0, 10),
    }
    return BoardProfile(
        name="v2",
        size_mm=BOARD_SIZE_MM,
        period=100,
        aruco_marker_id=21,
        ftk_marker_id=241,
        aruco_size_mm=90.0,
        counter_bits=20,
        ring_radius_mm={CameraType.RGB: 115.0, CameraType.INFRARED: 110.0},
        always_on_leds={
            # The 5th LED of each channel sits beside the top-left corner. It is the
            # only asymmetry in the layout: mirroring the board about a diagonal maps
            # every other LED onto another LED, so this one is what tells a correct fit
            # from its mirror image.
            CameraType.RGB: np.array(
                [
                    (20.0, 20.0),
                    (230.0, 20.0),
                    (230.0, 230.0),
                    (20.0, 230.0),
                    (5.0, 20.0),
                ],
                dtype=np.float32,
            ),
            CameraType.INFRARED: np.array(
                [
                    (5.0, 5.0),
                    (245.0, 5.0),
                    (245.0, 245.0),
                    (5.0, 245.0),
                    (20.0, 5.0),
                ],
                dtype=np.float32,
            ),
        },
        counter_led_coords={
            ct: np.array(rows, dtype=np.float32) for ct, rows in counter.items()
        },
        counter_bg_y={CameraType.RGB: 34.0, CameraType.INFRARED: 34.0},
    )


BOARD_V1 = _build_v1()
BOARD_V2 = _build_v2()
ALL_PROFILES = [BOARD_V1, BOARD_V2]
PROFILES_BY_NAME = {p.name: p for p in ALL_PROFILES}
PROFILES_BY_ARUCO = {p.aruco_marker_id: p for p in ALL_PROFILES}
PROFILES_BY_FTK = {p.ftk_marker_id: p for p in ALL_PROFILES}
