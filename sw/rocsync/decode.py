"""The result of one decode attempt, shared by every detection route."""

from dataclasses import dataclass

# Why a decode attempt produced no board time
NO_BOARD = "no_board"  # the board is not in view
NO_CORNERS = "no_corners"  # the board is in view but its corners could not be fitted
COUNTER_ZERO = "counter_zero"  # the counter has not started, or the orientation is undetermined
RING = "ring"  # no single usable arc, e.g. split or across the counter's wrap
CLUTTERED = "cluttered"  # too many points to plausibly be only the board's


@dataclass
class Decode:
    """One decode attempt; ``reject`` is None exactly when the board time is a real reading."""

    reject: str | None = None
    counter: int = 0
    ring_start: int = 0  # board time in ms at the first lit ring LED
    ring_end: int = 0  # board time in ms at the last lit ring LED
    board_ms: float = 0.0
    exposure_ms: float = -1.0  # -1 on reject, since 0 is a valid exposure

    @property
    def board_seen(self) -> bool:
        return self.reject != NO_BOARD

    @property
    def board_time(self) -> tuple[int, int] | None:
        """(start_ms, end_ms), or None on reject."""
        return (self.ring_start, self.ring_end) if self.reject is None else None


def decode_reading(board, counter: int, ring: tuple[int, int] | None) -> Decode:
    """Board time from a counter and ring reading, or the reason there is none.

    A zero counter is rejected: either the count has not started or the board's
    orientation was not determined, so the ring index would be meaningless.
    """
    if counter == 0:
        return Decode(reject=COUNTER_ZERO)
    times = board.board_time_from_ring(counter, ring) if ring is not None else None
    if times is None:
        return Decode(reject=RING)
    start, end = times
    return Decode(
        counter=counter,
        ring_start=start,
        ring_end=end,
        board_ms=float(start),
        exposure_ms=float(end - start),
    )
