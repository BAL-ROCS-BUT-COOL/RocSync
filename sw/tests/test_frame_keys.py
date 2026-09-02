"""The identity that ground truth and benchmark results associate through.

A key that formats or sorts differently silently splits one frame into two: the
annotation lands under one string and the prediction under another, and evaluation
scores the frame as a miss on both sides rather than reporting an error.
"""

from rocsync.benchmark.common import frame_key, parse_frame_key, retimed_videos, source_key

RETIMED = retimed_videos(
    {
        "videos": {
            "clip.retimed.mp4": {"source": "clip.mp4", "source_frame_offset": 12},
            "clip.mp4": {"timeline": "measured"},
        }
    }
)


def test_still_image_key_is_its_relative_path():
    assert frame_key("sub/img.png") == "sub/img.png"
    assert frame_key("sub/img.png", None) == "sub/img.png"


def test_video_frame_key_carries_a_padded_index():
    assert frame_key("sub/clip.mp4", 0) == "sub/clip.mp4#000000"
    assert frame_key("sub/clip.mp4", 42) == "sub/clip.mp4#000042"


def test_keys_sort_in_frame_order():
    """Padding exists for this; without it frame 100 would sort before frame 9."""
    keys = [frame_key("clip.mp4", i) for i in range(151)]
    assert sorted(keys) == keys


def test_a_video_frame_sorts_next_to_its_own_file():
    """`#` sorts below `.` and `/`, so frames never interleave with sibling names."""
    keys = sorted(["clip.mp4#000001", "clip.mp4#000000", "clip.txt", "clip-other.mp4"])
    assert keys == ["clip-other.mp4", "clip.mp4#000000", "clip.mp4#000001", "clip.txt"]


def test_a_key_round_trips_back_to_its_parts():
    for rel_path, index in [("sub/img.png", None), ("sub/clip.mp4", 0), ("sub/clip.mp4", 42)]:
        assert parse_frame_key(frame_key(rel_path, index)) == (rel_path, index)


def test_only_the_last_separator_splits_the_key():
    """A path may contain a `#` of its own; the index is always what follows the last."""
    assert parse_frame_key("take #2/clip.mp4#000007") == ("take #2/clip.mp4", 7)
    assert parse_frame_key("take #2/still.png") == ("take #2/still.png", None)


def test_a_trailing_separator_without_an_index_is_part_of_the_path():
    assert parse_frame_key("clip.mp4#") == ("clip.mp4#", None)
    assert parse_frame_key("clip.mp4#abc") == ("clip.mp4#abc", None)


def test_a_retimed_frame_resolves_to_the_annotation_it_was_cut_from():
    """Annotations stay with the recording, so a retimed prediction has to reach back."""
    assert source_key("clip.retimed.mp4#000000", RETIMED) == "clip.mp4#000012"
    assert source_key("clip.retimed.mp4#000005", RETIMED) == "clip.mp4#000017"


def test_source_key_round_trips_against_frame_key():
    video = RETIMED["clip.retimed.mp4"]
    for index in range(5):
        key = source_key(frame_key(video.path, index), RETIMED)
        assert parse_frame_key(key) == (video.source, index + video.frame_offset)


def test_source_key_leaves_a_recording_and_a_still_alone():
    assert source_key("clip.mp4#000003", RETIMED) == "clip.mp4#000003"
    assert source_key("still.png", RETIMED) == "still.png"
    assert source_key("clip.mp4#000003", {}) == "clip.mp4#000003"
