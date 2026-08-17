"""Video enumeration and the frame reads behind it.

`FrameSource` reaches a frame by grabbing forward or by seeking depending on how far
away it is, and either path can return the *wrong* frame rather than fail, so every read
is checked against a plain sequential decode of the same file. Key formatting is covered
by test_frame_keys, which needs no fixture.
"""

import shutil
import subprocess

import cv2
import numpy as np
import pytest

from rocsync.benchmark.common import (
    FORWARD_GRAB_LIMIT,
    FRAME_CACHE_SIZE,
    FrameRef,
    FrameSource,
    collect_frames,
    count_video_frames,
    frame_key,
)

N_FRAMES = 20
FPS = 10


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="needs ffmpeg to synthesize the fixture"
)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A short clip whose every frame differs, so a misread frame cannot pass."""
    directory = tmp_path_factory.mktemp("frames")
    path = directory / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=64x48:rate={FPS}:duration={N_FRAMES / FPS:g}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def expected(clip):
    """Every frame of the clip, decoded front to back -- the reference to match."""
    cap = cv2.VideoCapture(str(clip), cv2.CAP_FFMPEG)
    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) == N_FRAMES
    return frames


def _ref(clip, index):
    return FrameRef(clip, index, frame_key(clip.name, index))


def test_count_video_frames_matches_a_full_decode(clip, expected):
    assert count_video_frames(clip) == len(expected)


def test_sequential_read_returns_every_frame(clip, expected):
    with FrameSource() as source:
        for i, reference in enumerate(expected):
            assert np.array_equal(source.read(_ref(clip, i)), reference), f"frame {i}"


def test_stepping_back_returns_the_earlier_frame(clip, expected):
    with FrameSource() as source:
        source.read(_ref(clip, 5))
        assert np.array_equal(source.read(_ref(clip, 4)), expected[4])


def test_jumping_backwards_seeks_to_the_right_frame(clip, expected):
    """Past the cache, so this exercises the seek rather than a cache hit."""
    with FrameSource() as source:
        for i in range(N_FRAMES):
            source.read(_ref(clip, i))
        assert np.array_equal(source.read(_ref(clip, 1)), expected[1])


def test_jumping_far_forward_seeks_to_the_right_frame(clip, expected):
    target = FORWARD_GRAB_LIMIT + 5
    assert target < N_FRAMES, "fixture too short to pass the forward-grab limit"
    with FrameSource() as source:
        source.read(_ref(clip, 0))
        assert np.array_equal(source.read(_ref(clip, target)), expected[target])


def test_reading_the_same_frame_twice_is_stable(clip, expected):
    with FrameSource() as source:
        first = source.read(_ref(clip, 7))
        assert first is not None
        first = first.copy()
        assert np.array_equal(source.read(_ref(clip, 7)), first)
        assert np.array_equal(first, expected[7])


def test_cache_stays_bounded(clip):
    """A 4K frame is ~25 MB, so an unbounded cache would grow with the dataset."""
    with FrameSource() as source:
        for i in range(N_FRAMES):
            source.read(_ref(clip, i))
        assert len(source._cache) <= FRAME_CACHE_SIZE


def test_out_of_range_frame_reads_as_none(clip):
    with FrameSource() as source:
        assert source.read(_ref(clip, N_FRAMES + 10)) is None


# ── Enumeration ─────────────────────────────────────────────────────────────


@pytest.fixture
def dataset(tmp_path, clip):
    """A directory mixing a still image with a video, as a real dataset does."""
    shutil.copy(clip, tmp_path / "clip.mp4")
    cv2.imwrite(str(tmp_path / "still.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    return tmp_path


def test_collect_frames_enumerates_images_and_every_video_frame(dataset):
    frames = collect_frames(dataset)

    expected_keys = [f"clip.mp4#{i:06d}" for i in range(N_FRAMES)] + ["still.png"]
    assert [ref.key for ref in frames] == expected_keys
    assert [ref.index for ref in frames[:N_FRAMES]] == list(range(N_FRAMES))
    assert frames[-1].index is None


def test_collect_frames_ignores_a_directory_named_like_a_video(dataset):
    (dataset / "notavideo.mp4").mkdir()
    assert len(collect_frames(dataset)) == N_FRAMES + 1


def test_collect_frames_skips_an_unreadable_video(dataset):
    (dataset / "broken.mp4").write_bytes(b"not a video")
    frames = collect_frames(dataset)

    # No key at all, so the annotator and the validator cannot disagree about it
    assert not any("broken" in ref.key for ref in frames)
    assert len(frames) == N_FRAMES + 1
