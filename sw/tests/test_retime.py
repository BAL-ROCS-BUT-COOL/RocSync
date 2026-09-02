"""Retimed benchmark clips: the timeline is rewritten, the pixels are not.

A recording's container timestamps are a nominal grid the sensor only approximates, which
puts a floor of several milliseconds under any clock fitted against them. `rocsync-retime`
removes that floor by making the annotated board times the timeline, so the clock a
benchmark run has to recover is known exactly. Two properties carry the whole idea and are
pinned here: every anchor lands on the recorded clock to well under the 2 ms tolerance, and
every frame is bit-identical to the one it was copied from -- without which the corner,
homography and LED annotations the clip is built from would no longer describe it.
"""

import itertools
import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from rocsync.benchmark.common import (
    SYNTHESIZED_RESIDUAL_THRESHOLD_MS,
    annotated_board_time,
    frame_key,
    retimed_videos,
    source_key,
)
from rocsync.benchmark.retime import RetimeError, retime_video
from rocsync.timeline import frame_pts

av = pytest.importorskip("av", reason="rocsync-retime needs PyAV")

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="needs ffmpeg to synthesize the fixture"
)

VIDEO = "clips/take.mp4"
RETIMED = "clips/take.retimed.mp4"
N_FRAMES = 40
FPS = 20
ARUCO_ID = 21
BOARD_PERIOD = 100  # ring LEDs, so board time is counter * 100 + ring start
EXPOSURE_MS = 8
FIRST_ANCHOR, LAST_ANCHOR, ANCHOR_STRIDE = 4, 36, 4
BOARD_START_MS = 61_000.0  # a plausible board time, far from zero


def as_annotation(board_ms):
    """A ground truth entry whose counter and ring reconstruct to `board_ms`."""
    counter, ring_start = divmod(round(board_ms), BOARD_PERIOD)
    return {
        "aruco": {"visible": True, "id": ARUCO_ID},
        "counter": {"visible": True, "value": counter},
        "ring": {"start": ring_start, "end": (ring_start + EXPOSURE_MS) % BOARD_PERIOD},
    }


@pytest.fixture
def dataset(tmp_path):
    """A clip whose every frame differs, annotated on every fourth frame.

    The board advances a little faster than the container thinks it does, so a retimer
    that ignored the annotations and merely copied the timeline would be caught.
    """
    (tmp_path / "clips").mkdir()
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size=64x48:rate={FPS}:duration={N_FRAMES / FPS:g}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp_path / VIDEO),
        ],
        check=True,
    )  # fmt: skip

    pts = frame_pts(tmp_path / VIDEO)
    images = {
        frame_key(VIDEO, index): as_annotation(BOARD_START_MS + 1.004 * pts[index])
        for index in range(FIRST_ANCHOR, LAST_ANCHOR + 1, ANCHOR_STRIDE)
    }
    return tmp_path, {"images": images, "videos": {}}


def read_all(path):
    """Every frame of a clip, decoded front to back."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)
    cap.release()
    return frames


def anchor_board_times(ground_truth):
    """{source frame index: board time} for the annotations that anchor a timeline.

    Read through `annotated_board_time` rather than recomputed, so the test anchors on
    exactly the frames the retimer does -- an arc too close to the edge of the period
    names no single time and is no anchor, however the entry was written.
    """
    anchors = {}
    for index in range(FIRST_ANCHOR, LAST_ANCHOR + 1, ANCHOR_STRIDE):
        board_ms = annotated_board_time(ground_truth["images"][frame_key(VIDEO, index)])
        if board_ms is not None:
            anchors[index] = board_ms
    return anchors


def test_every_anchor_lands_on_the_recorded_clock(dataset):
    """The property the whole exercise exists for, read back the way the pipeline reads."""
    data_dir, ground_truth = dataset
    retime_video(data_dir, VIDEO, ground_truth)
    entry = ground_truth["videos"][RETIMED]

    pts = frame_pts(data_dir / RETIMED)
    offset = entry["source_frame_offset"]
    residuals = [
        entry["clock_rate"] * pts[index - offset] + entry["clock_offset_ms"] - board_ms
        for index, board_ms in anchor_board_times(ground_truth).items()
    ]

    assert max(abs(r) for r in residuals) < 0.1
    assert entry["max_residual_ms"] < SYNTHESIZED_RESIDUAL_THRESHOLD_MS
    assert entry["clock_rate"] != 1.0  # drawn, so a bug that ignores it cannot pass


def test_the_pixels_are_copied_rather_than_re_encoded(dataset):
    """The annotations describe the source's pixels, so they have to survive unchanged."""
    data_dir, ground_truth = dataset
    retime_video(data_dir, VIDEO, ground_truth)
    offset = ground_truth["videos"][RETIMED]["source_frame_offset"]

    source, retimed = read_all(data_dir / VIDEO), read_all(data_dir / RETIMED)

    assert retimed  # a clip that decoded to nothing would pass the loop below
    for index, frame in enumerate(retimed):
        assert np.array_equal(frame, source[index + offset]), f"frame {index} differs"


def test_the_clip_spans_the_annotated_window(dataset):
    """Trimming to the anchors is what keeps every timestamp interpolated, never guessed."""
    data_dir, ground_truth = dataset
    retime_video(data_dir, VIDEO, ground_truth)
    entry = ground_truth["videos"][RETIMED]

    offset, n_frames = entry["source_frame_offset"], entry["n_frames"]
    assert offset <= FIRST_ANCHOR
    assert offset + n_frames - 1 >= LAST_ANCHOR
    # Unannotated frames inside the window stay: wraps and partial boards are the point
    assert n_frames > len(anchor_board_times(ground_truth))

    pts = frame_pts(data_dir / RETIMED)
    assert len(pts) == n_frames
    assert pts[0] >= 0.0
    assert all(b > a for a, b in itertools.pairwise(pts))


def test_the_annotations_are_left_where_they_are(dataset):
    """A retimed clip is a derived artifact; regenerating it must cost no annotation work."""
    data_dir, ground_truth = dataset
    before = json.dumps(ground_truth["images"], sort_keys=True)

    retime_video(data_dir, VIDEO, ground_truth)

    assert json.dumps(ground_truth["images"], sort_keys=True) == before
    retimed = retimed_videos(ground_truth)
    offset = ground_truth["videos"][RETIMED]["source_frame_offset"]
    assert source_key(frame_key(RETIMED, 0), retimed) == frame_key(VIDEO, offset)


def test_a_clip_is_rebuilt_only_when_its_annotations_move(dataset):
    data_dir, ground_truth = dataset
    retime_video(data_dir, VIDEO, ground_truth)
    first = dict(ground_truth["videos"][RETIMED])

    assert retime_video(data_dir, VIDEO, ground_truth).endswith("up to date")
    assert ground_truth["videos"][RETIMED] == first

    edited = frame_key(VIDEO, LAST_ANCHOR)
    ground_truth["images"][edited] = as_annotation(
        anchor_board_times(ground_truth)[LAST_ANCHOR] + 3
    )
    retime_video(data_dir, VIDEO, ground_truth)

    assert ground_truth["videos"][RETIMED]["anchors_digest"] != first["anchors_digest"]


def test_annotations_that_disagree_with_the_recording_are_refused(dataset):
    """Board time running backwards is a bad annotation, not a timeline to build on."""
    data_dir, ground_truth = dataset
    ground_truth["images"][frame_key(VIDEO, LAST_ANCHOR)] = as_annotation(BOARD_START_MS - 450)

    with pytest.raises(RetimeError, match="do not increase"):
        retime_video(data_dir, VIDEO, ground_truth)
