#!/usr/bin/env python3
"""Rewrite a benchmark video's timestamps onto a clock derived from its annotations.

A camera's container timestamps are not a record of when it exposed. The phone clips in
the dataset write a near-nominal grid while the sensor wanders +/-7 ms around it, which
puts a floor under any clock fitted against them -- fitting the undecimated original over
650 frames gives the same 3.2 ms as the decimated copy, so the information was never
written rather than lost in processing.

That floor is the honest answer to how well a phone can be synchronized, and far too
coarse to catch a regression in decoding or fitting. So this tool produces the other kind
of benchmark video: the annotated board times become the timeline, and the clock the
pipeline is supposed to recover is known exactly instead of fitted. Only the container's
timestamps change -- packets are copied through untouched, so the pixels every existing
annotation describes are still the same pixels.

Usage:
    python -m rocsync.benchmark.retime [data_dir] [-o ground_truth.json]
"""

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

from rocsync.benchmark.common import (
    MIN_REFERENCE_FRAMES,
    SYNTHESIZED_RESIDUAL_THRESHOLD_MS,
    annotated_board_time,
    collect_frames,
    is_retimed,
    parse_frame_key,
    retimed_path,
    source_frame_period_ms,
)
from rocsync.timeline import frame_pts

try:
    import av
except ImportError:  # pragma: no cover - exercised only without the dev group
    av = None

# Enough that the anchors' rounding is negligible against the 2 ms they are held to
OUTPUT_TIME_BASE = Fraction(1, 90000)

CLOCK_RATE_SPREAD = 0.05  # stays inside process_video's own 5% sanity warning
MUX_PAD_MS = 1000.0  # headroom so reordered decode timestamps stay above zero
MAX_MARGIN_FRAMES = 5  # frames the codec may force outside the annotated window
MAX_REPORTED_CONFLICTS = 5  # enough to see the pattern without burying the message


class RetimeError(Exception):
    """A video that cannot be retimed, with a reason worth printing."""


# ── The synthesized clock ───────────────────────────────────────────────────


def draw_clock(rel_path):
    """The clock rate to build one video's timeline on, drawn but reproducible.

    Seeded on the source path so regenerating a clip lands on the same rate, and drawn
    rather than fixed so a rate of exactly 1.0 -- which several kinds of bug would
    produce by accident -- never counts as a pass. The offset needs no drawing: it comes
    out as the board time of the clip's first frame, tens of seconds from zero already.
    """
    rng = random.Random(hashlib.sha256(str(rel_path).encode()).hexdigest())
    return rng.uniform(1 - CLOCK_RATE_SPREAD, 1 + CLOCK_RATE_SPREAD)


def anchors_digest(anchors):
    """Fingerprint of the annotations a retimed clip was built from."""
    payload = ";".join(f"{index}:{board_ms:.6f}" for index, board_ms in anchors)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_timeline(anchors, pts_by_index, clock_rate):
    """A monotonic map from old presentation time to new, in ms, zeroed on the first anchor.

    Interpolates between the anchors, so `board time == clock_rate * new + offset` holds
    exactly at every one of them once the caller adds its shift. Interpolating in old
    presentation time rather than frame index keeps the recording's own shape, including
    any gap where frames were dropped.

    Timestamps outside the anchored span do occur -- a stream copy cannot start mid-GOP
    or orphan a reference, and a B-frame's decode timestamp precedes the first frame it
    is shown after -- so the map extends past both ends at the overall anchor slope
    rather than flattening out, which is what `np.interp` alone would do.
    """
    old = np.array([pts_by_index[index] for index, _ in anchors], dtype=float)
    board = np.array([board_ms for _, board_ms in anchors], dtype=float)
    # Naming the frames, because the fix is to re-annotate them and a bare refusal
    # leaves that search to the reader
    conflicts = [
        f"{anchors[i][0]}->{anchors[i + 1][0]} ({board[i]:.0f} -> {board[i + 1]:.0f} ms)"
        for i in range(len(anchors) - 1)
        if board[i + 1] <= board[i] or old[i + 1] <= old[i]
    ]
    if conflicts:
        raise RetimeError(
            "annotated board times do not increase with the recording at frames "
            + ", ".join(conflicts[:MAX_REPORTED_CONFLICTS])
            + (
                f" and {len(conflicts) - MAX_REPORTED_CONFLICTS} more"
                if len(conflicts) > MAX_REPORTED_CONFLICTS
                else ""
            )
        )
    new = (board - board[0]) / clock_rate
    slope = (new[-1] - new[0]) / (old[-1] - old[0])

    def remap(old_ms):
        if old_ms < old[0]:
            return float(new[0] + (old_ms - old[0]) * slope)
        if old_ms > old[-1]:
            return float(new[-1] + (old_ms - old[-1]) * slope)
        return float(np.interp(old_ms, old, new))

    return remap


# ── Reading what is already there ───────────────────────────────────────────


def video_anchors(images, rel_path):
    """[(frame index, board start ms)] for the frames that pin down a timeline.

    Only annotations a board time follows from anchor anything: a ring arc that wraps the
    end of the period was exposed across a counter increment and names no single time.
    """
    anchors = []
    for key, entry in images.items():
        path, index = parse_frame_key(key)
        if index is None or path != rel_path:
            continue
        board_ms = annotated_board_time(entry)
        if board_ms is not None:
            anchors.append((index, board_ms))
    return sorted(anchors)


def is_current(entry, output_path, digest):
    """Whether a retimed clip already reflects the annotations it was cut from."""
    return (
        entry is not None and entry.get("anchors_digest") == digest and Path(output_path).is_file()
    )


# ── Writing the clip ────────────────────────────────────────────────────────


def remux(
    source, destination, first_index, last_index, remap, anchor_indices, jitter_ms, rng, pad_ms
):
    """Copy the frames covering [first_index, last_index] across, with new timestamps.

    Returns (frames written, display index of the first frame kept, shift applied in ms).
    Packets are copied, never re-encoded, so every pixel an annotation describes survives
    untouched.

    Frames are selected by display index but written in decode order, which is not the
    same order once B-frames are involved. A copy also cannot start mid-GOP or orphan a
    reference that a retained frame needs, so the span is widened to the enclosing
    keyframe and to whatever decode order drags in. That can pull a few frames past the
    annotated window; they follow the overall anchor slope, which over a handful of
    frames costs far less than the tolerance they are never checked against anyway.

    `remap` is zeroed on the first anchor, so everything is shifted up by at least
    `pad_ms` before it is written -- far enough that no decode timestamp lands below
    zero, which MP4 will not mux.
    """
    assert av is not None, "main() refuses to run without PyAV"
    with av.open(str(source)) as inp, av.open(str(destination), "w") as out:
        in_stream = inp.streams.video[0]
        out_stream = out.add_stream_from_template(in_stream)
        out_stream.time_base = OUTPUT_TIME_BASE

        # Decode order, which is the order they must be written back in
        packets = [p for p in inp.demux(in_stream) if p.pts is not None and p.dts is not None]
        # Display order: the n-th smallest presentation timestamp is frame n
        display_of = {
            decode_pos: display
            for display, decode_pos in enumerate(
                sorted(range(len(packets)), key=lambda i: packets[i].pts)
            )
        }

        wanted = [i for i in range(len(packets)) if first_index <= display_of[i] <= last_index]
        if not wanted:
            raise RetimeError("no packet covers the annotated window")
        lo, hi = min(wanted), max(wanted)
        while lo > 0 and not packets[lo].is_keyframe:
            lo -= 1

        kept = sorted(display_of[i] for i in range(lo, hi + 1))
        if kept != list(range(kept[0], kept[0] + len(kept))):
            raise RetimeError("the frames a stream copy needs are not contiguous")

        # `remap` speaks the decoder's start-relative timestamps, packets the raw ones
        start_ts = in_stream.start_time or 0

        def old_ms(stamp):
            return float((stamp - start_ts) * in_stream.time_base * 1000)

        jitter = {
            i: rng.gauss(0, jitter_ms) if jitter_ms and display_of[i] not in anchor_indices else 0.0
            for i in range(lo, hi + 1)
        }
        shift = pad_ms - min(
            min(remap(old_ms(packets[i].pts)) + jitter[i], remap(old_ms(packets[i].dts)))
            for i in range(lo, hi + 1)
        )
        shift = max(shift, pad_ms)

        def stamp(ms):
            return round((ms + shift) / 1000 / OUTPUT_TIME_BASE)

        for i in range(lo, hi + 1):
            packet = packets[i]
            packet.pts = stamp(remap(old_ms(packet.pts)) + jitter[i])
            packet.dts = stamp(remap(old_ms(packet.dts)))
            packet.time_base = OUTPUT_TIME_BASE
            packet.stream = out_stream
            out.mux(packet)

    return len(kept), kept[0], shift


# ── Per-video driver ────────────────────────────────────────────────────────


def retime_video(data_dir, rel_path, ground_truth, jitter_ms=0.0, force=False):
    """Retime one source video, returning a status line for it.

    Raises RetimeError when the clip cannot be built; the caller decides how loud that is.
    """
    anchors = video_anchors(ground_truth["images"], rel_path)
    if len(anchors) < MIN_REFERENCE_FRAMES:
        return f"{rel_path}: {len(anchors)} anchors, needs {MIN_REFERENCE_FRAMES}, skipped"

    out_rel = str(retimed_path(rel_path))
    digest = anchors_digest(anchors)
    entry = (ground_truth.get("videos") or {}).get(out_rel)
    if not force and is_current(entry, data_dir / out_rel, digest):
        return f"{rel_path}: up to date"

    pts_by_index = dict(enumerate(frame_pts(data_dir / rel_path)))
    missing = [i for i, _ in anchors if i not in pts_by_index]
    if missing:
        raise RetimeError(f"{rel_path}: annotated frames {missing[:5]} are not in the file")

    clock_rate = draw_clock(rel_path)
    remap = build_timeline(anchors, pts_by_index, clock_rate)

    first_index, last_index = anchors[0][0], anchors[-1][0]
    rng = random.Random(hashlib.sha256(f"jitter:{rel_path}".encode()).hexdigest())
    n_frames, start_index, _shift_ms = remux(
        data_dir / rel_path,
        data_dir / out_rel,
        first_index,
        last_index,
        remap,
        {i for i, _ in anchors},
        jitter_ms,
        rng,
        MUX_PAD_MS,
    )

    margin = (first_index - start_index) + (start_index + n_frames - 1 - last_index)
    if margin > MAX_MARGIN_FRAMES:
        Path(data_dir / out_rel).unlink(missing_ok=True)
        raise RetimeError(
            f"{rel_path}: the codec forces {margin} frames outside the annotated window, "
            f"more than the {MAX_MARGIN_FRAMES} that stay within tolerance"
        )

    # Read the clip back the way the pipeline will: OpenCV reports timestamps relative
    # to the stream's start time, so the offset that actually holds is only measurable
    # from the written file, not from what was handed to the muxer.
    written_pts = frame_pts(data_dir / out_rel)
    if len(written_pts) != n_frames:
        raise RetimeError(f"{rel_path}: wrote {n_frames} frames but reads back {len(written_pts)}")
    anchor_pts = [written_pts[index - start_index] for index, _ in anchors]
    boards = [board_ms for _, board_ms in anchors]
    clock_offset_ms = float(
        np.mean([b - clock_rate * p for p, b in zip(anchor_pts, boards, strict=True)])
    )
    residuals = np.array(
        [clock_rate * p + clock_offset_ms - b for p, b in zip(anchor_pts, boards, strict=True)]
    )
    if np.abs(residuals).max() > SYNTHESIZED_RESIDUAL_THRESHOLD_MS:
        Path(data_dir / out_rel).unlink(missing_ok=True)
        raise RetimeError(
            f"{rel_path}: the muxed timeline misses its anchors by up to "
            f"{np.abs(residuals).max():.3f} ms"
        )
    # The recording keeps its own measured reference: the two describe different
    # questions, and the report picks between them rather than needing one deleted.
    ground_truth.setdefault("videos", {})[out_rel] = {
        "timeline": "synthesized",
        "source": rel_path,
        "source_frame_offset": start_index,
        "n_frames": n_frames,
        "anchors_digest": digest,
        "clock_rate": clock_rate,
        "clock_offset_ms": clock_offset_ms,
        "pts_min_ms": min(anchor_pts),
        "pts_max_ms": max(anchor_pts),
        "n_frames_fitted": len(anchors),
        "rmse_ms": float(np.sqrt(np.mean(residuals**2))),
        "max_residual_ms": float(np.abs(residuals).max()),
        "residual_threshold_ms": SYNTHESIZED_RESIDUAL_THRESHOLD_MS,
        "source_frame_period_ms": source_frame_period_ms(data_dir / rel_path),
        "derived_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return (
        f"{rel_path}: {len(anchors)} anchors, frames {start_index}-{start_index + n_frames - 1} "
        f"of {len(pts_by_index)}, rate {clock_rate:.6f}, offset {clock_offset_ms:.1f} ms"
    )


def _retime_one(data_dir, rel_path, ground_truth, args):
    """(message, ok) for one video, so a bad clip does not abandon the rest."""
    try:
        return retime_video(data_dir, rel_path, ground_truth, args.jitter_ms, args.force), True
    except (RetimeError, OSError) as e:
        return f"ERROR: {e}", False


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite benchmark video timestamps onto a clock built from annotations"
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="validation_data",
        help="Path to validation data directory (default: validation_data)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Ground truth JSON to read and update (default: <data_dir>/ground_truth.json)",
    )
    parser.add_argument(
        "--jitter-ms",
        type=float,
        default=0.0,
        help="Noise added to frames without an annotation; anchors stay exact (default: 0)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild clips even when already up to date"
    )
    args = parser.parse_args()

    if av is None:
        print(
            "rocsync-retime needs PyAV, which ships in the dev dependency group.\n"
            "Run `uv sync` from RocSync/sw, then try again.",
            file=sys.stderr,
        )
        return 1

    data_dir = Path(args.data_dir)
    output_path = Path(args.output) if args.output else data_dir / "ground_truth.json"
    if not output_path.is_file():
        print(f"No ground truth at {output_path}", file=sys.stderr)
        return 1
    with open(output_path) as f:
        ground_truth = json.load(f)
    ground_truth.setdefault("images", {})
    ground_truth.setdefault("videos", {})

    sources = sorted(
        {
            parse_frame_key(ref.key)[0]
            for ref in collect_frames(data_dir, sources_only=True)
            if ref.index is not None and not is_retimed(ref.path)
        }
    )
    if not sources:
        print(f"No source videos found in {data_dir}", file=sys.stderr)
        return 1

    failed = False
    for rel_path in sources:
        message, ok = _retime_one(data_dir, rel_path, ground_truth, args)
        print(message) if ok else print(message, file=sys.stderr)
        failed |= not ok

    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
