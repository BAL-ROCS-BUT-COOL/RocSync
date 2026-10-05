"""One open capture per video, with pts as the ground truth for which frame is which.

`CAP_PROP_POS_FRAMES` converts a frame index into a timestamp through the container's
*average* frame rate, so a recording that is not spaced at that rate -- a dropped span, or
anything variable-rate -- lands a seek on a neighbour while still reporting back the index
that was asked for. `VideoReader` seeks the way this codebase's other frame readers already
do: as a starting point only, confirmed and corrected against each frame's own presentation
timestamp. `rocsync.timeline.frame_pts` reads that mapping from the container's packets, with
no decoding, so it is cheap enough to build once and trust for the reader's whole life.
"""

import bisect

import cv2

from rocsync.timeline import frame_pts

SEEK_BACKOFF_FRAMES = 32  # retry margin for a seek that lands past the frame wanted
FORWARD_GRAB_LIMIT = 12  # a seek re-decodes from the preceding keyframe anyway, so nearby wins


class VideoReader:
    """Presents one video as pts-indexed frames.

    `seek`, `read` and `frames` all land on the frame a timestamp names, never the
    neighbour the container's average frame rate would guess. `pts` is read once, lazily,
    the first time position information beyond "the next frame" is needed -- a caller that
    only ever reads forward from the start never pays for it.
    """

    def __init__(self, path):
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise OSError(f"Could not open video: {self.path}")
        self._pts = None
        self._next_index = 0  # index the capture would deliver on its next grab, or -1 if unknown

    @property
    def pts(self):
        """Presentation timestamp in ms of every frame, indexed by frame number."""
        if self._pts is None:
            self._pts = frame_pts(self.path)
        return self._pts

    @property
    def fps(self):
        """The container's nominal frame rate -- a default for a caller's own sampling
        stride, never a mapping from time to index."""
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def reported_frame_count(self):
        """The container's own frame count: free, but occasionally wrong.

        Only ever a display default -- `len(self)` is the exact count, and costs building
        the pts index to get it.
        """
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return len(self.pts)

    def index_at(self, pts_ms, side="left"):
        """Index of the first frame at/after `pts_ms` (side="left"), or strictly after it
        (side="right"). A `pts_ms` beyond the last frame returns `len(self)`."""
        bisect_fn = bisect.bisect_left if side == "left" else bisect.bisect_right
        return bisect_fn(self.pts, pts_ms)

    def _window(self, index):
        """The pts span that belongs to `index` and to no other frame: the midpoints to
        its neighbours, or one ms past an end where there is no neighbour to split with."""
        pts = self.pts
        low = (pts[index - 1] + pts[index]) / 2 if index > 0 else pts[index] - 1.0
        high = (pts[index] + pts[index + 1]) / 2 if index + 1 < len(pts) else pts[index] + 1.0
        return low, high

    def _is_frame(self, index):
        """Whether the frame just grabbed off the capture is the one `index` names."""
        low, high = self._window(index)
        return low < self._cap.get(cv2.CAP_PROP_POS_MSEC) < high

    def seek(self, index):
        """Grabs forward until frame `index` is the one just grabbed but not yet decoded;
        `retrieve()` (or the next step of `read`/`frames`) reads it from there.

        The seek is only a starting point: landing is confirmed against the grabbed
        frame's own timestamp, and a seek that overshot is retried from further back, down
        to the start of the file. Returns `index`, or None if the file has nothing there.
        """
        if index >= len(self):
            self._next_index = -1
            return None

        low, high = self._window(index)
        for start in (index, max(index - SEEK_BACKOFF_FRAMES, 0), 0):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            while True:
                if not self._cap.grab():
                    self._next_index = -1
                    return None
                now = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                if now > low:
                    break
            if now < high:
                self._next_index = index + 1
                return index
        self._next_index = -1
        return None

    def read(self, index):
        """Decoded frame `index`, or None if it could not be read.

        A nearby later frame is reached by grabbing on from wherever the capture already
        stands -- a seek re-decodes from the preceding keyframe anyway, so that is cheaper
        only once the gap is worth it -- otherwise by `seek`.
        """
        ahead = index - self._next_index
        frame = None
        if 0 <= ahead <= FORWARD_GRAB_LIMIT:
            landed = all(self._cap.grab() for _ in range(ahead))
            if landed and self._cap.grab():
                success, candidate = self._cap.retrieve()
                if success and self._is_frame(index):
                    frame = candidate
        if frame is None:
            if self.seek(index) is None:
                return None
            success, frame = self._cap.retrieve()
            frame = frame if success else None
        self._next_index = index + 1 if frame is not None else -1
        return frame

    def frames(self, start=0, stop=None, decode_if=None):
        """Yields (index, pts_ms, frame) for `start <= index < stop`, in file order.

        `decode_if(index)` gates whether a frame is decoded at all -- grabbing without
        retrieving skips converting and copying a frame nobody wants, which is how a caller
        samples a file cheaply. `None` decodes every frame. `stop=None` keeps going until
        the file runs out, without ever needing the pts index -- so a plain linear read
        from the start costs nothing beyond today's frame count.
        """
        index = start
        just_grabbed = False
        if start > 0:
            if self.seek(start) is None:
                return
            just_grabbed = True

        while stop is None or index < stop:
            if not just_grabbed and not self._cap.grab():
                self._next_index = -1
                return
            just_grabbed = False

            pts_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            frame = None
            if decode_if is None or decode_if(index):
                success, frame = self._cap.retrieve()
                if not success:
                    self._next_index = -1
                    return
            self._next_index = index + 1
            yield index, pts_ms, frame
            index += 1

    def close(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
