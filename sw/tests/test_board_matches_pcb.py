"""Check the board profiles against the KiCad PCBs they describe.

``board_profiles`` is hand-written millimetres, so the thing worth testing is not that
it agrees with some earlier version of itself but that it agrees with the hardware.
This walks the PCB files and asserts a bijection: every LED the software models matches
a distinct footprint on the board, and every footprint on the board is modelled.

The PCBs live outside the ``sw`` package, so the test skips when ``hw`` is absent.
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest

from rocsync.board_profiles import BOARD_V1, BOARD_V2
from rocsync.camera import CameraType

HW_DIR = Path(__file__).resolve().parents[2] / "hw"

# The two LED parts. Their footprint anchors are the LED body centre: in both, the
# courtyard and fab outline are symmetric about the origin.
FOOTPRINTS = {
    CameraType.RGB: "custom_footprints:HL-A-3528",
    CameraType.INFRARED: "custom_footprints:XYC-HIR76C-LX4",
}

# KiCad stores coordinates in nanometres, so exact values round-trip to ~1e-6 mm.
TOL_MM = 1e-4

pytestmark = pytest.mark.skipif(
    not HW_DIR.is_dir(), reason=f"KiCad sources not available at {HW_DIR}"
)


def _balanced(text, start):
    """The balanced-paren block beginning at ``text[start]``."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("unbalanced parentheses")


def _origin(pcb_text):
    """Top-left of the Edge.Cuts rectangle, and the board's size in mm."""
    for m in re.finditer(r"\(gr_rect", pcb_text):
        block = _balanced(pcb_text, m.start())
        if "Edge.Cuts" not in block:
            continue
        s = re.search(r"\(start ([-\d.]+) ([-\d.]+)\)", block)
        e = re.search(r"\(end ([-\d.]+) ([-\d.]+)\)", block)
        assert s and e, "Edge.Cuts rectangle has no start/end"
        x0, y0 = float(s.group(1)), float(s.group(2))
        x1, y1 = float(e.group(1)), float(e.group(2))
        return (min(x0, x1), min(y0, y1)), (abs(x1 - x0), abs(y1 - y0))
    raise AssertionError("no Edge.Cuts rectangle found")


def _led_positions(pcb_text, footprint, origin):
    """Anchors of every ``footprint`` instance, relative to the board origin (mm)."""
    out = []
    for m in re.finditer(rf'\(footprint "{re.escape(footprint)}"', pcb_text):
        block = _balanced(pcb_text, m.start())
        at = re.search(r"\(at ([-\d.]+) ([-\d.]+)", block)
        assert at, f"{footprint} instance has no anchor"
        out.append([float(at.group(1)) - origin[0], float(at.group(2)) - origin[1]])
    return np.array(out)


CASES = [pytest.param(BOARD_V1, "rev1", ct, id=f"v1-{ct.value}") for ct in CameraType] + [
    pytest.param(BOARD_V2, "rev2", ct, id=f"v2-{ct.value}") for ct in CameraType
]


@pytest.mark.parametrize("profile, rev, camera_type", CASES)
def test_layout_matches_pcb(profile, rev, camera_type):
    text = (HW_DIR / rev / "RocSync_testing.kicad_pcb").read_text()
    origin, (width, height) = _origin(text)

    assert (width, height) == (profile.size_mm, profile.size_mm)

    pcb = _led_positions(text, FOOTPRINTS[camera_type], origin)
    model = profile.layout_coords(camera_type)

    assert len(model) == len(pcb), (
        f"{profile.name} {camera_type.value}: model has {len(model)} LEDs, PCB has {len(pcb)}"
    )

    distance = np.linalg.norm(model[:, None, :] - pcb[None, :, :], axis=2)
    nearest = distance.argmin(axis=1)

    worst = distance.min(axis=1).max()
    assert worst < TOL_MM, f"worst model-to-PCB distance {worst:.2e} mm"
    assert len(set(nearest)) == len(pcb), "two modelled LEDs claim the same footprint"


@pytest.mark.parametrize("profile, rev", [(BOARD_V1, "rev1"), (BOARD_V2, "rev2")])
def test_aruco_marker_matches_pcb(profile, rev):
    """The marker's black border, derived from the silkscreen module grid."""
    text = (HW_DIR / rev / "RocSync_testing.kicad_pcb").read_text()
    origin, _ = _origin(text)

    # rev2 places a dedicated aruco footprint; rev1 carries the marker as a traced
    # "LOGO" footprint. Either way the silk polygons cover the 4x4 data area, and the
    # black border is one module of bare soldermask around it.
    pattern = r'\(footprint "(?:custom_footprints:aruco[^"]*|LOGO)"'
    m = re.search(pattern, text)
    assert m, f"{rev}: no marker footprint found"
    block = _balanced(text, m.start())
    at = re.search(r"\(at ([-\d.]+) ([-\d.]+)", block)
    assert at, f"{rev}: marker footprint has no anchor"
    centre = (float(at.group(1)) - origin[0], float(at.group(2)) - origin[1])

    polygons = [
        np.array([[float(x), float(y)] for x, y in re.findall(r"\(xy ([-\d.]+) ([-\d.]+)\)", pts)])
        for pts in re.findall(r"\(fp_poly\s*\(pts(.*?)\)\s*\(stroke", block, re.DOTALL)
    ]
    assert polygons, f"{rev}: marker footprint has no polygons"

    # The silk polygons are the *white* modules of the 4x4 data area. rev2 also draws a
    # quiet-zone frame around the marker; it is recognisable as the one polygon whose
    # bounding box strictly encloses every other polygon, so drop it and keep the data
    # area. rev1 draws no frame and nothing is dropped.
    def bbox(points):
        return points.min(axis=0), points.max(axis=0)

    data = polygons
    if len(polygons) > 1:
        for i, poly in enumerate(polygons):
            rest = np.concatenate(polygons[:i] + polygons[i + 1 :])
            (lo, hi), (rlo, rhi) = bbox(poly), bbox(rest)
            if (lo < rlo).all() and (hi > rhi).all():
                data = polygons[:i] + polygons[i + 1 :]
                break

    lo, hi = bbox(np.concatenate(data))
    module = float((hi - lo).mean()) / 4  # the data area is 4 modules across
    marker = 6 * module  # plus a one-module black border on each side
    data_centre = (lo + hi) / 2

    assert math.isclose(marker, profile.aruco_size_mm, abs_tol=0.15), (
        f"{rev}: PCB implies a {marker:.3f} mm marker, profile says {profile.aruco_size_mm}"
    )
    # rev1's marker is a traced bitmap, so it lands a fraction of a millimetre off the
    # nominal centre; rev2's is exact geometry.
    assert math.isclose(centre[0] + data_centre[0], profile.centre_mm, abs_tol=0.25)
    assert math.isclose(centre[1] + data_centre[1], profile.centre_mm, abs_tol=0.25)
