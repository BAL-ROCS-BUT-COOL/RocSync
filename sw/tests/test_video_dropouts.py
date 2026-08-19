"""End-to-end timing over a video that really is missing frames.

These go through `process_video` / `process_video_window` rather than the fit in
isolation, so they fail if the frames' presentation timestamps stop being the
quantity the pipeline times against. The board detection is stubbed out -- what
is under test is the timing, not the vision.
"""

import math
import shutil
import subprocess

import pytest

from rocsync import video
from rocsync.camera import CameraType
from rocsync.timeline import frame_pts

FPS = 30.0
PERIOD = 1000 / FPS
DROP_FIRST, DROP_LAST = 30, 74  # 45 frames removed, 1.5 s, starting at 1 s
N_DROPPED = DROP_LAST - DROP_FIRST + 1
OFFSET = 1234.5  # board time at pts 0
EXPOSURE = 9.0

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="needs ffmpeg to synthesize the fixture"
)


@pytest.fixture(scope="module")
def gap_video(tmp_path_factory):
    """A 4 s clip with a 1.5 s hole punched out, keeping the original timestamps."""
    directory = tmp_path_factory.mktemp("dropouts")
    source = directory / "src.mp4"
    gapped = directory / "gap.mp4"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate={FPS:g}:duration=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    # `select` without a following `setpts` leaves the surviving frames on their
    # original timeline, so the removed span becomes a real gap in the PTS.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            f"select='not(between(n,{DROP_FIRST},{DROP_LAST}))'",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "libx264",
            str(gapped),
        ],
        check=True,
    )
    return str(gapped)


@pytest.fixture
def board_at_pts(gap_video, monkeypatch):
    """Stub the board reader so each frame reports board time = pts + OFFSET.

    Keyed on the frame number the pipeline hands to `process_frame`, so if the
    pipeline mislabels a frame the timing comes out wrong.
    """
    pts_by_index = frame_pts(gap_video)
    seen = []

    def fake_process_frame(frame, camera_type, frame_number, board, debug_dir=None):
        pts = pts_by_index[frame_number]
        seen.append((frame_number, pts))
        start = pts + OFFSET
        return True, (start, start + EXPOSURE)

    monkeypatch.setattr(video, "process_frame", fake_process_frame)
    return seen


def test_reader_reports_the_presentation_timestamp_of_each_frame(gap_video):
    """Guards the claim that POS_MSEC needs no off-by-one correction."""
    import queue as queue_module

    import cv2

    cap = cv2.VideoCapture(gap_video, cv2.CAP_FFMPEG)
    frame_queue = queue_module.Queue(maxsize=8)
    thread = __import__("threading").Thread(target=video.read_frames_async, args=(cap, frame_queue))
    thread.daemon = True
    thread.start()

    read = []
    while True:
        frame, frame_number, pts_ms = frame_queue.get()
        if frame is None:
            break
        read.append((frame_number, pts_ms))
    thread.join(timeout=5)
    cap.release()

    expected = list(enumerate(frame_pts(gap_video)))
    assert read == expected
    assert read[0] == (0, 0.0)  # first frame at zero, not one period in


def test_process_video_recovers_the_clock_across_a_dropout(gap_video, board_at_pts):
    statistics = video.process_video(gap_video, CameraType.RGB, stride=1)
    assert statistics is not None, "process_video found no board"

    # The container clock is correct as it stands, so no rescaling is needed.
    # Fitting on the frame index instead would misplace this by the 1.5 s hole.
    assert statistics.clock_rate == pytest.approx(1.0, abs=1e-4)
    assert statistics.clock_offset_ms == pytest.approx(OFFSET, abs=1.0)
    assert statistics.first_frame == pytest.approx(OFFSET, abs=1.0)
    assert statistics.rmse_after < 1.0
    assert statistics.n_rejected_frames == 0

    # The gap is reported, not absorbed
    assert statistics.n_gaps == 1
    assert statistics.n_dropped_frames == N_DROPPED
    assert statistics.largest_gap_ms == pytest.approx((N_DROPPED + 1) * PERIOD, abs=1.0)

    # measured_fps is a real rate, not frames divided by a gap-inflated duration
    assert statistics.measured_fps == pytest.approx(FPS, abs=0.01)
    assert statistics.median_frame_period == pytest.approx(PERIOD, abs=0.01)
    assert not statistics.timeline_windowed


def test_last_frame_is_anchored_on_a_frame_that_exists(gap_video, board_at_pts):
    statistics = video.process_video(gap_video, CameraType.RGB, stride=1)
    assert statistics is not None, "process_video found no board"

    pts = frame_pts(gap_video)
    assert statistics.last_frame == pytest.approx(pts[-1] + OFFSET, abs=1.0)
    assert statistics.container_duration == pytest.approx(pts[-1] - pts[0], abs=1e-6)

    # The recording is longer than its frames account for, so walking n_frames
    # forward at the true frame period stops well short of the real last frame
    assert statistics.board_duration > statistics.n_frames * PERIOD
    assert statistics.board_duration == pytest.approx(pts[-1] - pts[0], abs=1e-6)


def test_window_selects_frames_by_timestamp_not_by_index(gap_video, board_at_pts):
    """A window after the gap must not be shifted by the missing frames."""
    window_start, window_end = 2.0, 3.0
    timestamps, frame_times = video.process_video_window(
        gap_video, CameraType.RGB, window_start, window_end, stride=1
    )

    analyzed_pts = [pts for _, pts in board_at_pts]
    assert analyzed_pts, "nothing was analyzed"
    assert min(analyzed_pts) >= window_start * 1000
    assert max(analyzed_pts) <= window_end * 1000

    # Every frame the file actually has in that span was analyzed
    expected = [p for p in frame_pts(gap_video) if window_start * 1000 <= p <= window_end * 1000]
    assert sorted(analyzed_pts) == pytest.approx(sorted(expected))

    # Indices are shifted by the 45 removed frames, so an index-derived window
    # would have started 1.5 s late; the timestamps prove which frames were used
    analyzed_indices = [index for index, _ in board_at_pts]
    assert min(analyzed_indices) == len(
        [p for p in frame_pts(gap_video) if p < window_start * 1000]
    )
    assert set(timestamps) == set(analyzed_indices)
    assert all(frame_times[i] == pytest.approx(p) for i, p in board_at_pts)


def test_a_single_window_is_the_only_span_analyzed(gap_video, board_at_pts):
    """A lone window must not be widened to the whole file."""
    video.process_video(gap_video, CameraType.RGB, stride=1, windows=[(2.5, 3.5)])

    analyzed_pts = [pts for _, pts in board_at_pts]
    assert analyzed_pts, "nothing was analyzed"
    assert min(analyzed_pts) >= 2500
    assert max(analyzed_pts) <= 3500


def test_disjoint_windows_are_analyzed_and_the_span_between_them_is_not(gap_video, board_at_pts):
    statistics = video.process_video(
        gap_video, CameraType.RGB, stride=1, windows=[(0.0, 0.5), (3.0, 3.5)]
    )
    assert statistics is not None, "process_video found no board"

    analyzed_pts = [pts for _, pts in board_at_pts]
    assert any(pts <= 500 for pts in analyzed_pts)
    assert any(pts >= 3000 for pts in analyzed_pts)
    assert not any(500 < pts < 3000 for pts in analyzed_pts)

    # The 1.5 s hole lies between the windows, so it is nobody's dropout
    assert statistics.n_dropped_frames == 0
    assert statistics.timeline_windowed


def test_overlapping_windows_are_merged_into_one_scan(gap_video, board_at_pts):
    statistics = video.process_video(
        gap_video, CameraType.RGB, stride=1, windows=[(0.0, 2.0), (1.0, 3.0)]
    )
    assert statistics is not None, "process_video found no board"

    analyzed_indices = [index for index, _ in board_at_pts]
    assert analyzed_indices, "nothing was analyzed"
    assert len(analyzed_indices) == len(set(analyzed_indices)), "frames scanned twice"
    assert max(pts for _, pts in board_at_pts) <= 3000

    # Counting the hole once, not once per overlapping window
    assert statistics.n_gaps == 1
    assert statistics.n_dropped_frames == N_DROPPED


def test_negative_window_bounds_count_back_from_the_last_frame(gap_video, board_at_pts):
    pts = frame_pts(gap_video)
    video.process_video(gap_video, CameraType.RGB, stride=1, windows=[(-1.0, math.inf)])

    analyzed_pts = [p for _, p in board_at_pts]
    assert analyzed_pts, "nothing was analyzed"
    assert min(analyzed_pts) >= pts[-1] - 1000
    assert max(analyzed_pts) == pytest.approx(pts[-1])


def test_probe_last_pts_ms_finds_the_real_last_frame(gap_video):
    assert video.probe_last_pts_ms(gap_video) == pytest.approx(frame_pts(gap_video)[-1])
