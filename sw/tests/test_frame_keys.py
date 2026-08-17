"""The identity that ground truth and benchmark results associate through.

A key that formats or sorts differently silently splits one frame into two: the
annotation lands under one string and the prediction under another, and evaluation
scores the frame as a miss on both sides rather than reporting an error.
"""

from rocsync.benchmark.common import frame_key


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
