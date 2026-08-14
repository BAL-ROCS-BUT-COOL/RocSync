"""Shared clip-extraction helpers for the evaluation scripts.

Sampling a clip means picking one source frame per output frame, so the frame
indices to read are known up front and are non-decreasing. Reading them with a
seek per frame makes the decoder discard its state and start again from the
preceding keyframe, which costs far more than simply decoding every frame in
between once. `read_frames_at_indices` therefore walks the file linearly and
hands each requested frame straight to its consumer, so neither the decode cost
nor the memory scales with how the clip was sampled.
"""

from collections.abc import Iterator

import cv2
from tqdm import tqdm

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

    `frame_indices` must be non-decreasing; a repeated index yields the same
    decoded frame again, so an output faster than the source repeats frames
    rather than re-reading them. Frames are not copied -- consumers that keep a
    frame beyond its iteration must copy it themselves.

    Stops early and warns if the source runs out of frames.
    """
    if not frame_indices:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

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

                while (
                    out_index < len(frame_indices)
                    and frame_indices[out_index] <= source_index
                ):
                    yield out_index, frame
                    out_index += 1
                    pbar.update(1)
                source_index += 1
        finally:
            pbar.close()
    finally:
        cap.release()
