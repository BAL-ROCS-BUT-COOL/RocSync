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


def _ring(radius_mm, period, centre):
    """Ring LED centres (mm), index 0 at the top, running counter-clockwise."""
    angles = -(np.arange(period) / period + 0.25) * 2 * math.pi
    return np.stack(
        [centre + radius_mm * np.cos(angles), centre + radius_mm * np.sin(angles)],
        axis=1,
    )


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
