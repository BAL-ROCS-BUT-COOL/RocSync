"""Clip selection and reading, shared by the evaluation scripts.

A clip is a span of board time. Extracting one means picking a source frame for
every output frame, then reading those frames back out of the file.
"""

import json
from collections.abc import Iterator

import cv2
import numpy as np
from tqdm import tqdm

from rocsync.timecode import ms_to_timecode, timecode_to_ms, timecode_to_path_part


class Clip:
    """A requested clip, as board-time milliseconds and as path-safe strings.

    Timecodes in the clips config are board time, unless `from_camera_time` is
    set: they are then read as positions in one camera's own container timeline
    and mapped through that camera's fitted clock.
    """

    def __init__(
        self,
        start_string: str,
        end_string: str,
        from_camera_time: bool = False,
        clock_offset_ms: float | None = None,
        clock_rate: float | None = None,
    ) -> None:
        if from_camera_time:
            if clock_offset_ms is None or clock_rate is None:
                raise ValueError(
                    "from_camera_time needs both clock_offset_ms and clock_rate to map "
                    "the camera's timeline onto board time."
                )
            start_string = ms_to_timecode(
                clock_offset_ms + clock_rate * timecode_to_ms(start_string)
            )
            end_string = ms_to_timecode(clock_offset_ms + clock_rate * timecode_to_ms(end_string))

        self.start = timecode_to_ms(start_string)
        self.end = timecode_to_ms(end_string)
        self.start_string_formatted = timecode_to_path_part(start_string)
        self.end_string_formatted = timecode_to_path_part(end_string)


def parse_clips_json(
    path: str,
    from_camera_time: bool = False,
    clock_offset_ms: float | None = None,
    clock_rate: float | None = None,
) -> list[Clip]:
    """Read a clips config: a JSON list of {"start": ..., "end": ...} timecodes."""
    with open(path, encoding="utf-8") as f:
        clips_raw = json.load(f)

    return [
        Clip(clip["start"], clip["end"], from_camera_time, clock_offset_ms, clock_rate)
        for clip in clips_raw
    ]


def select_frame_indices(
    target_fps: float,
    frame_times: list[float],
    start_ms: float,
    end_ms: float,
) -> list[int]:
    """Source frame indices sampling [`start_ms`, `end_ms`) onto a uniform `target_fps` grid.

    `frame_times` holds the board time of every source frame, so the clip is
    picked by time rather than by index and a dropped span does not shift it.
    One index is returned per grid point strictly inside the clip: the frame
    whose timestamp is closest to that point, preferring the earlier frame when
    two are equally close.
    """
    if target_fps <= 0:
        return []

    actual_timestamps = np.asarray(frame_times, dtype=np.float64)
    if actual_timestamps.size == 0:
        return []

    dt = 1000.0 / target_fps
    target_timestamps = np.arange(start_ms, end_ms, dt)

    frames_to_extract: list[int] = []
    i = 0
    for t in tqdm(target_timestamps, desc="Matching timestamps"):
        # The grid only increases, so the search never has to walk back
        while i + 1 < len(actual_timestamps) and abs(actual_timestamps[i + 1] - t) < abs(
            actual_timestamps[i] - t
        ):
            i += 1
        frames_to_extract.append(i)

    return frames_to_extract


# Caps decoded frames handed to a writer pool but not yet encoded, so a consumer
# slower than the decoder queues frames instead of the whole file. At 4K a frame
# is ~25 MB, so this is the difference between a fixed cost and one that grows
# with the clip.
MAX_FRAMES_IN_FLIGHT = 8


def read_frames_at_indices(
    video_path: str,
    frame_indices: list[int],
    desc: str = "Reading frames",
) -> Iterator[tuple[int, "cv2.typing.MatLike"]]:
    """Yield (output index, frame) for each entry of `frame_indices`, in order.

    The indices are known up front and are non-decreasing. Reading them with a
    seek per frame makes the decoder discard its state and start again from the
    preceding keyframe, which costs far more than simply decoding every frame in
    between once, so the file is walked linearly and each requested frame is
    handed straight to its consumer. Neither the decode cost nor the memory
    scales with how the clip was sampled.

    A repeated index yields the same
    decoded frame again, so an output faster than the source repeats frames
    rather than re-reading them. Frames are not copied -- consumers that keep a
    frame beyond its iteration must copy it themselves.

    Stops early and warns if the source runs out of frames.
    """
    if not frame_indices:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Could not open video file: {video_path}")

    try:
        # Skipping to the first requested frame is the one seek worth doing.
        first = max(0, frame_indices[0])
        if first > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)

        pbar = tqdm(total=len(frame_indices), desc=desc)
        try:
            out_index = 0
            source_index = first
            while out_index < len(frame_indices):
                if not cap.grab():
                    print(
                        f"Warning: {video_path} ended after "
                        f"{out_index}/{len(frame_indices)} requested frames"
                    )
                    return

                # Frames between two wanted ones are grabbed but never retrieved,
                # which skips converting and copying them into an image. Sampling
                # below the source frame rate skips most of the file this way.
                if frame_indices[out_index] > source_index:
                    source_index += 1
                    continue

                success, frame = cap.retrieve()
                if not success:
                    print(
                        f"Warning: {video_path} ended after "
                        f"{out_index}/{len(frame_indices)} requested frames"
                    )
                    return

                while out_index < len(frame_indices) and frame_indices[out_index] <= source_index:
                    yield out_index, frame
                    out_index += 1
                    pbar.update(1)
                source_index += 1
        finally:
            pbar.close()
    finally:
        cap.release()
