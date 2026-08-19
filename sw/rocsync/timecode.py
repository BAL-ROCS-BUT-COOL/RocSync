"""Parsing and formatting of 'hh:mm:ss' times.

Used for the command-line search windows and for the clip timecodes in a clips
config. Fields need no zero padding, the seconds may be fractional and the hours
are not capped at 24, so a time can be written the way a player displays it.
"""


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
