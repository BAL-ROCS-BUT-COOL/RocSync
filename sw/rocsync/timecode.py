"""Parsing and formatting of 'hh:mm:ss' times.

Used for the command-line search windows and for the clip timecodes in a clips
config. Fields need no zero padding, the seconds may be fractional and the hours
are not capped at 24, so a time can be written the way a player displays it.
"""

import math


def parse_hms(time_str: str, original: str | None = None, expected: str = "hh:mm:ss") -> float:
    """Seconds from an 'hh:mm:ss' or 'hh:mm:ss.mmm' time.

    `original` is the text to quote in the error when the caller has already
    stripped a prefix off it, and `expected` names the formats it accepts.
    """
    invalid = ValueError(f"invalid time {(original or time_str)!r}, expected {expected}")

    fields = time_str.split(":")
    if len(fields) != 3:
        raise invalid
    try:
        hours, minutes, seconds = int(fields[0]), int(fields[1]), float(fields[2])
    except ValueError:
        raise invalid from None
    if hours < 0 or minutes < 0 or seconds < 0:
        raise invalid

    return hours * 3600 + minutes * 60 + seconds


def timecode_to_ms(timecode: str) -> int:
    """'hh:mm:ss.mmm' -> milliseconds since 00:00:00.000."""
    return round(parse_hms(timecode) * 1000)


def ms_to_timecode(ms: float) -> str:
    """Milliseconds since 00:00:00.000 -> 'HH:MM:SS.mmm'."""
    seconds, milliseconds = divmod(round(ms), 1000)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"


def timecode_to_path_part(timecode: str) -> str:
    """'hh:mm:ss.mmm' -> 'HH_MM_SS_ffffff', usable inside a file or folder name."""
    seconds, milliseconds = divmod(timecode_to_ms(timecode), 1000)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}_{minutes:02}_{secs:02}_{milliseconds * 1000:06}"


def resolve_windows(windows, source_end_s=None):
    """Turns requested search windows into absolute [start, end] spans in seconds.

    A negative bound is an offset from the source's last time. The result is sorted,
    and overlapping spans are merged so that no frame is scanned -- and no gap between
    frames counted -- twice. `source_end_s` is a zero-argument callable returning the
    source's last time in seconds, on the same axis the windows use; it is only ever
    called when a negative bound needs resolving.
    """
    if not windows:
        return [(0.0, math.inf)]

    last_s = None
    if any(bound < 0 for window in windows for bound in window):
        last_s = source_end_s() if source_end_s else None
        if last_s is None:
            raise ValueError("no frame could be read to resolve a window bound given from the end")

    resolved = []
    for start, end in windows:
        if start < 0:
            start = max(0.0, last_s + start)
        if end < 0:
            end = max(0.0, last_s + end)
        if start >= end:
            raise ValueError(f"window [{start:.3f}s, {end:.3f}s] starts at or after it ends")
        resolved.append((start, end))

    resolved.sort()
    merged = [resolved[0]]
    for start, end in resolved[1:]:
        merged_start, merged_end = merged[-1]
        if start <= merged_end:  # overlapping or touching
            merged[-1] = (merged_start, max(merged_end, end))
        else:
            merged.append((start, end))
    return merged
