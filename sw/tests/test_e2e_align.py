"""`rocsync-align` end to end, on synthetic cameras whose clock the test chose.

Every frame shows its own index as a row of black and white blocks, so the board time a
player displays at any moment of an aligned output follows from the pixels alone:
`board(k) = clock_rate * pts_k + clock_offset_ms`. The sync file is built from
`RecordingStatistics`, so align reads the schema `rocsync` actually writes.

`--compensate-drift` re-encodes with whichever encoder align picks on this machine.
"""

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest

from rocsync.recording_statistics import (
    CLOCK_DRIFT_BAD_PPM,
    CLOCK_DRIFT_WARN_PPM,
    RecordingStatistics,
    rate_uncertainty_limit_ms,
)
from rocsync.timeline import frame_pts
from rocsync.video_reader import VideoReader
from tests.e2e import (
    assert_succeeded,
    cli,
    decode_gray,
    displayed_frames,
    output_of,
    requires_ffmpeg,
)

pytestmark = [pytest.mark.e2e, requires_ffmpeg]

WIDTH, HEIGHT = 160, 120
INDEX_BITS = 16
BLOCK_PX = 10  # blocks sit on a 20 px grid, two rows of eight
TARGET_FPS = 30
SLACK_MS = 1e-3  # ffprobe prints times to the microsecond


def index_image(k):
    image = np.zeros((HEIGHT, WIDTH), np.uint8)
    for bit in range(INDEX_BITS):
        if k >> bit & 1:
            x, y = (bit % 8) * 20 + 5, 40 + (bit // 8) * 20
            image[y : y + BLOCK_PX, x : x + BLOCK_PX] = 255
    return image


def read_index(image):
    k = 0
    for bit in range(INDEX_BITS):
        x, y = (bit % 8) * 20 + 5 + BLOCK_PX // 2, 40 + (bit // 8) * 20 + BLOCK_PX // 2
        if image[y, x] > 128:
            k |= 1 << bit
    return k


@dataclass(frozen=True)
class Camera:
    name: str
    fps: str  # as ffmpeg takes it, so 29.97 stays exact
    n_frames: int
    clock_rate: float
    clock_offset_ms: float

    @property
    def period_ms(self):
        return 1000.0 / float(Fraction(self.fps))

    def board(self, k):
        """Board time in ms of source frame `k`, which ffmpeg stamps at exactly k periods."""
        return self.clock_rate * k * self.period_ms + self.clock_offset_ms

    def board_period_ms(self):
        return self.clock_rate * self.period_ms

    def nearest_ms(self):
        """How far the frame nearest any board time can be from it: half a frame."""
        return self.board_period_ms() / 2 + SLACK_MS


def write_camera(path, camera, gop):
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{WIDTH}x{HEIGHT}",
        "-framerate", camera.fps, "-i", "-",
        "-c:v", "libx264", "-g", str(gop), "-pix_fmt", "yuv420p", str(path),
    ]  # fmt: skip
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for k in range(camera.n_frames):
        process.stdin.write(index_image(k).tobytes())
    process.stdin.close()
    assert process.wait() == 0


DROP = object()  # an override that removes the field altogether


def sync_entry(camera, **overrides):
    """One `output.json` entry for `camera`, shaped by the dataclass `rocsync` serializes."""
    period = camera.period_ms
    statistics = RecordingStatistics(
        n_frames=camera.n_frames,
        n_considered_frames=camera.n_frames,
        n_rejected_frames=0,
        r2_before=1.0,
        rmse_before=0.0,
        r2_after=1.0,
        rmse_after=0.0,
        source_duration=(camera.n_frames - 1) * period,
        board_duration=camera.board(camera.n_frames - 1) - camera.board(0),
        nominal_fps=1000.0 / period,
        measured_fps=1000.0 / period,
        clock_rate=camera.clock_rate,
        clock_offset_ms=camera.clock_offset_ms,
        clock_rate_stderr=0.0,
        extrapolation_stderr_ms=1.0,
        source_tick_ms=1.0,
        first_frame=camera.board(0),
        last_frame=camera.board(camera.n_frames - 1),
        median_frame_period=period,
        n_gaps=0,
        n_dropped_frames=0,
        largest_gap_ms=0.0,
        timeline_windowed=False,
        inlier_span_ms=(camera.n_frames - 1) * period,
        mean_exposure_time=0.0,
        min_exposure_time=0.0,
        max_exposure_time=0.0,
        std_exposure_time=0.0,
        considered_timestamps={},
        rejected_timestamps={},
    )
    entry = {"type": "video", **statistics.to_dict(), **overrides}
    return {k: v for k, v in entry.items() if v is not DROP}


def write_sync(directory, cameras, entries=None):
    entries = entries or {}
    sync = {
        str(directory / f"{c.name}.mp4"): sync_entry(c, **entries.get(c.name, {})) for c in cameras
    }
    path = directory / "sync.json"
    path.write_text(json.dumps(sync))
    return path


def common_span(cameras):
    """(origin, end): the board-time span every camera covers, from the chosen clocks."""
    origin = max(c.board(0) for c in cameras)
    end = min(c.board(c.n_frames - 1) for c in cameras)
    return origin, end


def run_align(sync_path, output_dir, *args):
    result = cli("rocsync-align", sync_path, "--output_dir", output_dir, *args, timeout=600)
    assert_succeeded(result)
    return result


def shown(camera, output):
    """(player timestamps in ms, board time of each displayed frame) of one output."""
    frames = decode_gray(output)
    times = np.array(displayed_frames(output))
    assert len(times) == len(frames)
    indices = np.array([read_index(frame) for frame in frames])
    return times, indices, np.array([camera.board(k) for k in indices])


# Staggered starts and ends, three frame rates, drift of a few hundred ppm
STREAM_COPY_CAMERAS = (
    Camera("cam25", "25", 500, 1.0, 3000.0),
    Camera("cam30", "30000/1001", 540, 1.0002, 5123.4),
    Camera("cam60", "60", 960, 0.9997, 4000.0),
)

# Drift of 1 %, so an output that ignored it would be off by several frames at the end
DRIFTING_CAMERAS = (
    Camera("cam25", "25", 300, 1.01, 3000.0),
    Camera("cam30", "30000/1001", 330, 0.99, 3800.0),
    Camera("cam60", "60", 600, 1.0, 3400.0),
)


GOPS = [pytest.param(1, id="all-intra"), pytest.param(30, id="gop30")]


@pytest.fixture(scope="module", params=GOPS)
def stream_copy(request, tmp_path_factory):
    """Stream-copy outputs of the three cameras, with keyframes every `gop` frames."""
    gop = request.param
    directory = tmp_path_factory.mktemp(f"align-copy-gop{gop}")
    for camera in STREAM_COPY_CAMERAS:
        write_camera(directory / f"{camera.name}.mp4", camera, gop)
    output_dir = directory / "synced"
    run_align(write_sync(directory, STREAM_COPY_CAMERAS), output_dir)
    return {c.name: output_dir / f"{c.name}.mp4" for c in STREAM_COPY_CAMERAS}


@pytest.mark.parametrize("camera", STREAM_COPY_CAMERAS, ids=lambda c: c.name)
def test_stream_copy_starts_on_the_common_span_and_keeps_time(stream_copy, camera):
    origin, _ = common_span(STREAM_COPY_CAMERAS)
    times, indices, board = shown(camera, stream_copy[camera.name])
    tolerance = camera.nearest_ms()

    assert np.all(np.diff(indices) > 0), "a stream copy must not repeat or reorder frames"
    assert abs(board[0] - origin) <= tolerance, f"starts at board {board[0]:.1f}, not {origin:.1f}"
    # A stream copy keeps the camera's own clock, so board time runs at clock_rate
    error = board - (origin + camera.clock_rate * times)
    assert np.max(np.abs(error)) <= tolerance, f"worst frame off by {np.max(np.abs(error)):.1f} ms"


@pytest.mark.parametrize(
    "stream_copy",
    [
        GOPS[0],
        pytest.param(
            30,
            id="gop30",
            # Whether it overruns depends on where the GOP structure falls at the end
            marks=pytest.mark.xfail(
                strict=False,
                reason="-t cuts a stream copy in decode order, so the B-frames before the "
                "last kept reference frame are lost and the output overruns the span",
            ),
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("camera", STREAM_COPY_CAMERAS, ids=lambda c: c.name)
def test_stream_copy_ends_on_the_common_span(stream_copy, camera):
    _, end = common_span(STREAM_COPY_CAMERAS)
    _, indices, board = shown(camera, stream_copy[camera.name])

    assert np.all(np.diff(indices) == 1), "a stream copy must not drop frames"
    assert abs(board[-1] - end) <= camera.nearest_ms(), (
        f"ends at board {board[-1]:.1f}, not {end:.1f}"
    )


@pytest.mark.parametrize("camera", STREAM_COPY_CAMERAS, ids=lambda c: c.name)
def test_rocsync_reads_an_aligned_output_like_a_player(stream_copy, camera):
    output = stream_copy[camera.name]
    player = displayed_frames(output)

    assert frame_pts(output) == pytest.approx(player, abs=1e-3)
    with VideoReader(output) as reader:
        assert len(reader) == len(player)
        first = reader.read(0)
        assert first is not None
        assert read_index(first[:, :, 0]) == read_index(decode_gray(output)[0])


@pytest.fixture(scope="module")
def compensated(tmp_path_factory):
    directory = tmp_path_factory.mktemp("align-drift")
    for camera in DRIFTING_CAMERAS:
        write_camera(directory / f"{camera.name}.mp4", camera, gop=30)
    output_dir = directory / "synced"
    run_align(
        write_sync(directory, DRIFTING_CAMERAS),
        output_dir,
        "--compensate-drift",
        "--fps",
        TARGET_FPS,
    )
    return {c.name: shown(c, output_dir / f"{c.name}.mp4") for c in DRIFTING_CAMERAS}


@pytest.mark.parametrize("camera", DRIFTING_CAMERAS, ids=lambda c: c.name)
def test_compensated_output_runs_on_board_time(compensated, camera):
    origin, end = common_span(DRIFTING_CAMERAS)
    times, _, board = compensated[camera.name]
    tolerance = camera.nearest_ms()

    error = board - (origin + times)
    assert np.max(np.abs(error)) <= tolerance, f"worst frame off by {np.max(np.abs(error)):.1f} ms"
    # The last output frame is the last one due within the span
    assert -tolerance - 1000 / TARGET_FPS < board[-1] - end <= tolerance, (
        f"ends at board {board[-1]:.1f}, not {end:.1f}"
    )


def test_compensated_outputs_agree_frame_by_frame(compensated):
    origin, end = common_span(DRIFTING_CAMERAS)
    counts = [len(board) for _, _, board in compensated.values()]
    due = int((end - origin) * TARGET_FPS / 1000) + 1
    assert counts == [due] * len(counts), f"frame counts {counts}, {due} frames are due"

    boards = np.array([board for _, _, board in compensated.values()])
    spread = boards.max(axis=0) - boards.min(axis=0)
    # Each camera within half its own frame, so two within the sum of their halves
    halves = sorted(c.nearest_ms() for c in DRIFTING_CAMERAS)
    tolerance = halves[-1] + halves[-2]
    assert np.max(spread) <= tolerance, f"cameras disagree by up to {np.max(spread):.1f} ms"


WARNING_CAMERAS = (
    Camera("steady", "25", 50, 1.0, 1000.0),
    Camera("odd", "25", 50, 1.0, 1100.0),
)

PERIOD_25 = 40.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"clock_rate": 1 + 2 * CLOCK_DRIFT_WARN_PPM * 1e-6}, "ppm off board time"),
        ({"clock_rate": 1 + 2 * CLOCK_DRIFT_BAD_PPM * 1e-6}, "something is wrong with the fit"),
        ({"extrapolation_stderr_ms": DROP}, "has no clock-rate uncertainty"),
        ({"extrapolation_stderr_ms": None}, "could not be measured"),
        (
            {"extrapolation_stderr_ms": 2 * rate_uncertainty_limit_ms(PERIOD_25)},
            "clock rate uncertain",
        ),
    ],
    ids=["drift", "bad-drift", "no-uncertainty", "unmeasured-uncertainty", "uncertain-rate"],
)
def test_align_warns_about_a_doubtful_clock_and_proceeds(tmp_path, overrides, message):
    for camera in WARNING_CAMERAS:
        write_camera(tmp_path / f"{camera.name}.mp4", camera, gop=1)
    sync = write_sync(tmp_path, WARNING_CAMERAS, {"odd": overrides})

    result = run_align(sync, tmp_path / "synced")

    lines = [line for line in output_of(result).splitlines() if message in line]
    assert lines, f"no warning containing {message!r}:\n{output_of(result)}"
    assert all("odd.mp4" in line for line in lines)
    assert all((tmp_path / "synced" / f"{c.name}.mp4").exists() for c in WARNING_CAMERAS)
