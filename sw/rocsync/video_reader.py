"""One open container per video, with pts as the ground truth for which frame is which.

A frame is named by its presentation timestamp, never by counting frames from wherever a
seek landed. `rocsync.timeline.frame_index` reads every frame's timestamp, and which frames
are keyframes, off the container's packets with no decoding, so it is cheap enough to build
once and trust for the reader's whole life. Decoding can only start at a keyframe, so a
seek lands on the keyframe at or before the frame wanted and decodes forward to it.
"""

import bisect

import av
from av.error import FFmpegError

from rocsync.timeline import frame_index, stream_time_base

FORWARD_GRAB_LIMIT = 12  # frames decoded forward rather than seeking, even past a keyframe


class VideoReader:
    """Presents one video as pts-indexed frames.

    `read` and `frames` land on the frame a timestamp names. `pts` and `keyframes` are read
    once, lazily, the first time position information beyond "the next frame" is needed --
    a caller that only ever reads forward from the start never pays for them.
    """

    def __init__(self, path):
        self.path = str(path)
        try:
            self._container = av.open(self.path)
        except FFmpegError as e:
            raise OSError(f"Could not open video: {self.path}") from e
        if not self._container.streams.video:
            self._container.close()
            raise OSError(f"No video stream in: {self.path}")
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        try:
            self._time_base = stream_time_base(self._stream, self.path)
        except OSError:
            self._container.close()
            raise
        start = self._stream.start_time
        self._start_tick = float(start) if start is not None else 0.0
        self._index = None
        self._decoded = None  # the decoder's frame iterator, positioned at `_next_index`
        self._next_index = 0  # index the decoder delivers next, or -1 if unknown

    @property
    def pts(self):
        """Presentation timestamp in ms of every frame, indexed by frame number."""
        return self._frame_index()[0]

    @property
    def keyframes(self):
        """Indices of the frames decoding can start at, ascending."""
        return self._frame_index()[1]

    def _frame_index(self):
        if self._index is None:
            self._index = frame_index(self.path)
        return self._index

    @property
    def fps(self):
        """The container's nominal frame rate -- never a mapping from time to index."""
        rate = self._stream.average_rate or self._stream.guessed_rate
        return float(rate) if rate else 0.0

    @property
    def reported_frame_count(self):
        """The container's own frame count: free, but occasionally wrong.

        Only ever a display default -- `len(self)` is the exact count, and costs building
        the pts index to get it.
        """
        if self._stream.frames:
            return self._stream.frames
        duration = self._container.duration  # in microseconds
        return int(duration / 1e6 * self.fps) if duration else 0

    def __len__(self):
        return len(self.pts)

    def index_at(self, pts_ms, side="left"):
        """Index of the first frame at/after `pts_ms` (side="left"), or strictly after it
        (side="right"). A `pts_ms` beyond the last frame returns `len(self)`."""
        bisect_fn = bisect.bisect_left if side == "left" else bisect.bisect_right
        return bisect_fn(self.pts, pts_ms)

    def _next(self):
        """The next decoded frame and its pts in ms, or (None, None) at the end of the file."""
        if self._decoded is None:
            self._decoded = self._container.decode(self._stream)
        for frame in self._decoded:
            if frame.pts is not None:
                # The same float arithmetic as `frame_index`, so the two agree bit for bit
                return frame, (frame.pts - self._start_tick) * self._time_base * 1000
        return None, None

    def _seek(self, index):
        """Decoded frame `index`, reached from the keyframe at or before it, or None."""
        self._next_index = -1
        if not 0 <= index < len(self):
            return None
        keyframes = self.keyframes
        position = bisect.bisect_right(keyframes, index) - 1
        key = keyframes[position] if position >= 0 else 0
        tick = round(self.pts[key] / 1000 / self._time_base + self._start_tick)
        self._container.seek(tick, stream=self._stream, backward=True)
        self._decoded = self._container.decode(self._stream)
        target = self.pts[index]
        while True:
            frame, pts_ms = self._next()
            if frame is None or pts_ms > target:
                return None
            if pts_ms == target:
                self._next_index = index + 1
                return frame

    def _forward(self, index):
        """Whether reaching `index` by decoding on beats seeking to it."""
        ahead = index - self._next_index
        if self._next_index < 0 or ahead < 0:
            return False
        if ahead <= FORWARD_GRAB_LIMIT:
            return True
        keyframes = self.keyframes
        return bisect.bisect_right(keyframes, index) == bisect.bisect_right(
            keyframes, self._next_index
        )

    def read(self, index):
        """Decoded BGR frame `index`, or None if it could not be read."""
        if self._forward(index):
            frame, pts_ms = None, None
            for _ in range(index - self._next_index + 1):
                frame, pts_ms = self._next()
                if frame is None:
                    break
            if frame is not None and index < len(self) and pts_ms == self.pts[index]:
                self._next_index = index + 1
                return to_bgr(frame)
        frame = self._seek(index)
        return None if frame is None else to_bgr(frame)

    def decoded(self, start=0, stop=None):
        """Yields (index, pts_ms, decoded frame) for `start <= index < stop`, in file order.

        The frames are still in the stream's own pixel format; `to_bgr` converts one.
        `stop=None` keeps going until the file runs out. A call that starts where the
        previous one stopped decodes on without seeking, and a plain linear read from the
        start never needs the pts index.
        """
        index = start
        pending = None
        if start != self._next_index:
            pending = self._seek(start)
            if pending is None:
                return
        while stop is None or index < stop:
            if pending is not None:
                frame, pts_ms, pending = pending, self.pts[index], None
            else:
                frame, pts_ms = self._next()
                if frame is None:
                    self._next_index = -1
                    return
            self._next_index = index + 1
            yield index, pts_ms, frame
            index += 1

    def frames(self, start=0, stop=None, decode_if=None):
        """`decoded`, converted to BGR where `decode_if(index)` holds and None elsewhere;
        `decode_if=None` converts every frame."""
        for index, pts_ms, frame in self.decoded(start, stop):
            yield index, pts_ms, to_bgr(frame) if decode_if is None or decode_if(index) else None

    def close(self):
        self._container.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def to_bgr(frame):
    """A decoded frame as a BGR image, converted on as many threads as swscale picks."""
    return frame.reformat(format="bgr24", threads=0).to_ndarray()
