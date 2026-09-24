"""Raw recordings to synced videos on full-length real footage, the way a user runs it.

Per session: `rocsync` on each recording within its board windows, `rocsync-align` on
the result both by stream copy and with `--compensate-drift`, then `rocsync` again on
every synced file. The windows, any further `rocsync` options and how each recording is
judged come from `e2e_real_samples.json` (see `generate_real_samples_manifest.py`).

The recordings are first downscaled into the e2e cache (`tests.e2e.downscaled`), which
keeps every frame's timestamp, so a clock fitted to a copy holds for its recording.

What this verifies is the path around the clock fit: windowed runs on long files, the
`output.json` handoff, where align cuts real containers, whether rocsync reads the synced
files the way a player shows them, and whether drift compensation puts every camera on
board time. It does not verify the fit itself; `test_e2e_benchmark.py` scores fits against
annotations.

A synced file keeps only the board windows inside the span every camera covers, often
just one, and a fit over one window pins its rate poorly. So the second run fits each
window on its own and is read at that window's centre, then carried back to the file's
first frame. A synced frame is the source frame nearest its board time, so any two such
readings agree to within half a frame of each camera, plus each fit's own uncertainty.

Set ROCSYNC_REAL_SAMPLES_DIR to run it. The synced files are deleted once a session is done.
"""

import json
import math
import shutil
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path

import pytest

from rocsync.recording_statistics import RATE_STDERR_COVERAGE
from rocsync.timeline import frame_pts, per_frame_times
from tests.e2e import (
    cli,
    displayed_frames,
    downscaled,
    env_dir,
    hms,
    output_of,
    plausibility_problems,
    real_sample_copy,
    requires_ffmpeg,
)

pytestmark = [pytest.mark.e2e, requires_ffmpeg]

DATA_DIR_VAR = "ROCSYNC_REAL_SAMPLES_DIR"
MANIFEST = Path(__file__).with_name("e2e_real_samples.json")
MODES = {"stream_copy": (), "compensated": ("--compensate-drift",)}


def _manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None


def _sessions():
    manifest = _manifest()
    if manifest is None:
        reason = "no manifest; run tests/generate_real_samples_manifest.py first"
        return [pytest.param(None, marks=pytest.mark.skip(reason=reason))]
    return sorted(manifest["sessions"])


def run_arguments(recording, windows):
    """The recording's own `rocsync` options from the manifest, and one `--window` per window."""
    windowed = [a for start, end in windows for a in ("--window", hms(start), hms(end))]
    return [*recording.get("arguments", []), *windowed]


def by_name(output):
    """{file stem: entry} of an output.json, or empty if the run wrote none."""
    if not output.exists():
        return {}
    return {Path(path).stem: entry for path, entry in json.loads(output.read_text()).items()}


def listing(*directories):
    return {
        p: (p.stat().st_size, p.stat().st_mtime_ns)
        for directory in directories
        for p in directory.iterdir()
    }


@dataclass
class WindowFit:
    """The second `rocsync` run on one board window of a synced file."""

    window: tuple  # (start, end) in s of the synced file
    sighted: bool  # whether the synced file keeps every sighting in it
    run: object
    fit: dict | None


@dataclass
class Synced:
    """One `rocsync-align` mode's outputs and the second `rocsync` runs on them."""

    align: object
    files: dict = field(default_factory=dict)  # by stem
    windows: dict = field(default_factory=dict)  # [WindowFit], by stem


@dataclass
class SessionRun:
    recordings: dict  # manifest entries, by stem
    copies: dict  # downscaled copies, by stem
    listing_before: dict
    runs: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    synced: dict = field(default_factory=dict)  # by mode


def output_windows(windows, entry, origin_ms, mode, duration_s, padding_s):
    """[(start, end, sighted)]: a recording's board windows, moved onto its synced file.

    Placed by the raw fit and the common start, which only need to be right to well
    within the windows' padding. A window keeps its sightings if the synced file's ends
    cut no deeper than half that padding.
    """
    rate, offset = entry["clock_rate"], entry["clock_offset_ms"]
    scale = rate if mode == "stream_copy" else 1.0  # a stream copy keeps its own clock

    def moved(t):
        return (rate * t * 1000 + offset - origin_ms) / scale / 1000

    placed = []
    for start, end in windows:
        s, e = moved(start), moved(end)
        if max(s, 0.0) < min(e, duration_s):
            sighted = s >= -padding_s / 2 and e <= duration_s + padding_s / 2
            placed.append((max(s, 0.0), min(e, duration_s), sighted))
    return placed


@pytest.fixture(scope="module", params=_sessions())
def session(request, tmp_path_factory):
    name = request.param
    manifest = _manifest()
    assert manifest is not None
    data_dir = env_dir(DATA_DIR_VAR)
    recordings, copies = {}, {}
    for file, entry in manifest["sessions"][name].items():
        source = data_dir / name / file
        if entry["check"] != "excluded" and source.is_file():
            stem = Path(file).stem
            recordings[stem] = entry
            copies[stem] = downscaled(source, real_sample_copy(data_dir, source))
    copy_dir = next(iter(copies.values())).parent if copies else data_dir / name
    run = SessionRun(recordings, copies, listing(data_dir / name, copy_dir))
    work = tmp_path_factory.mktemp(f"real-{name.replace('/', '_')}")

    # One run per recording, as windows apply to every file of a run; all share one output
    raw_output = work / "output.json"
    for stem, copy in copies.items():
        arguments = run_arguments(recordings[stem], recordings[stem]["windows"])
        run.runs[stem] = cli("rocsync", copy, "-o", raw_output, *arguments)
    run.raw = by_name(raw_output)
    origin_ms = max((per_frame_times(copies[s], e)[0] for s, e in run.raw.items()), default=0.0)
    padding_s = manifest["generated"]["padding_s"]

    for mode, arguments in MODES.items():
        synced_dir = work / mode
        synced = Synced(cli("rocsync-align", raw_output, "--output_dir", synced_dir, *arguments))
        for stem, entry in run.raw.items():
            path = synced_dir / f"{stem}.mp4"
            if not path.is_file():
                continue
            synced.files[stem] = path
            duration_s = frame_pts(path)[-1] / 1000
            placed = output_windows(
                recordings[stem]["windows"], entry, origin_ms, mode, duration_s, padding_s
            )
            # A window at a time, so each fit is judged where it has seen the board
            synced.windows[stem] = []
            for i, (start, end, sighted) in enumerate(placed):
                output = work / f"{mode}-{stem}-{i}.json"
                arguments = run_arguments(recordings[stem], [(start, end)])
                result = cli("rocsync", path, "-o", output, *arguments)
                fit = by_name(output).get(stem)
                synced.windows[stem].append(WindowFit((start, end), sighted, result, fit))
        run.synced[mode] = synced

    yield run
    shutil.rmtree(work, ignore_errors=True)


def board_at(entry, q_ms):
    return entry["clock_rate"] * q_ms + entry["clock_offset_ms"]


def half_frame_ms(entry):
    """Half a source frame in board time: how far the frame nearest any instant can be."""
    return entry["clock_rate"] * entry["median_frame_period"] / 2


def start_estimates(session, mode, stem):
    """[(window, board time at the synced file's first frame, tolerance)], one per fit.

    Each window's fit is read at the window's centre, where even a fit whose rate is
    poorly pinned is not extrapolating. Only windows that keep their sightings count: one
    cut deeper may keep just a few at one edge, far from its centre. Board time
    runs at the camera's own rate in a stream copy and at board rate in a compensated file.
    """
    raw = session.raw[stem]
    rate = raw["clock_rate"] if mode == "stream_copy" else 1.0
    estimates = []
    for judged in session.synced[mode].windows[stem]:
        fit = judged.fit
        if fit is None or not judged.sighted:
            continue
        start_s, end_s = judged.window
        centre_ms = (start_s + end_s) * 500
        # The fit's own 3 sigma at the centre: its offset, and its rate over half the window
        rate_stderr = fit["clock_rate_stderr"] / (fit.get("source_tick_ms") or 1.0)
        fit_ms = RATE_STDERR_COVERAGE * (
            fit["rmse_after"] / math.sqrt(fit["n_considered_frames"])
            + rate_stderr * (end_s - start_s) * 500
        )
        # Carrying a stream copy back to its first frame leans on the raw fit's rate
        carry_ms = raw["extrapolation_stderr_ms"] if mode == "stream_copy" else 0.0
        estimates.append(
            (
                judged.window,
                board_at(fit, centre_ms) - rate * centre_ms,
                half_frame_ms(raw) + fit_ms + carry_ms,
            )
        )
    return estimates


def judged(session, mode):
    """{stem: start estimates} of every synced recording, or a skip if fewer than two.

    A frozen recording joins too: the frozen check vouches for its raw fit.
    """
    synced = session.synced[mode]
    checked = [s for s in session.recordings if s in synced.files]
    if len(checked) < 2:
        pytest.skip("fewer than two recordings to compare")
    # A window that keeps its sightings holds every one the raw run had there
    failures = [
        f"{stem}: rocsync could not time-sync the window {w.window[0]:.1f}-{w.window[1]:.1f} s "
        f"of the synced file\n{output_of(w.run)[-2000:]}"
        for stem in checked
        for w in synced.windows[stem]
        if w.sighted and w.fit is None
    ]
    estimates = {stem: start_estimates(session, mode, stem) for stem in checked}
    failures += [
        f"{stem}: no board window keeps its sightings in the synced file"
        for stem in checked
        if not estimates[stem]
    ]
    assert not failures, "\n".join(failures)
    return estimates


def disagreements(pairs):
    """Pairs of start estimates further apart than their tolerances allow."""
    failures = []
    for (a, (wa, sa, ta)), (b, (wb, sb, tb)) in pairs:
        if abs(sa - sb) > ta + tb:
            failures.append(
                f"{a} at {wa[0]:.0f}-{wa[1]:.0f} s starts at board {sa:.1f} ms, "
                f"{b} at {wb[0]:.0f}-{wb[1]:.0f} s at {sb:.1f} ms (tolerance {ta + tb:.1f} ms)"
            )
    return failures


def test_rocsync_syncs_every_recording(session):
    failures = [
        f"{stem}: exit {result.returncode}\n{output_of(result)[-2000:]}"
        for stem, result in session.runs.items()
        if result.returncode != 0 or stem not in session.raw
    ]
    assert not failures, "\n".join(failures)
    assert all(entry["timeline_windowed"] for entry in session.raw.values())


def test_clock_fits_are_plausible_or_match_the_frozen_clock(session):
    failures = []
    for stem, entry in session.raw.items():
        recording = session.recordings[stem]
        if recording["check"] == "round_trip":
            failures += [f"{stem}: {problem}" for problem in plausibility_problems(entry)]
            continue
        frozen = recording["frozen_clock"]
        pts = frame_pts(session.copies[stem])
        for label, p in (("first", pts[0]), ("last", pts[-1])):
            got = board_at(entry, p)
            expected = board_at(frozen, p)
            if abs(got - expected) > frozen["tolerance_ms"]:
                failures.append(
                    f"{stem}: {got - expected:+.1f} ms from the frozen clock at the {label} "
                    f"frame (tolerance {frozen['tolerance_ms']:.1f} ms)"
                )
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mode", MODES)
def test_align_writes_every_recording_and_leaves_its_inputs_alone(session, mode):
    synced = session.synced[mode]
    assert synced.align.returncode == 0, output_of(synced.align)[-4000:]
    assert set(synced.files) == set(session.raw)
    parents = {p.parent for p in session.listing_before}
    assert listing(*parents) == session.listing_before


@pytest.mark.parametrize("mode", MODES)
def test_rocsync_reads_a_synced_recording_like_a_player(session, mode):
    for stem, path in session.synced[mode].files.items():
        player = displayed_frames(path, read_intervals="%+#5")
        assert frame_pts(path)[: len(player)] == pytest.approx(player, abs=1e-3), stem


@pytest.mark.parametrize("mode", MODES)
def test_synced_recordings_start_on_the_same_board_time(session, mode):
    estimates = judged(session, mode)
    pairs = [
        pair
        for a, b in combinations(sorted(estimates), 2)
        for pair in product([(a, e) for e in estimates[a]], [(b, e) for e in estimates[b]])
    ]
    failures = disagreements(pairs)
    assert not failures, "\n".join(failures)


def test_compensated_recordings_run_on_board_time(session):
    # Read at windows far apart, one camera's first frame must come out the same
    estimates = judged(session, "compensated")
    pairs = [
        ((stem, a), (stem, b)) for stem, own in estimates.items() for a, b in combinations(own, 2)
    ]
    failures = disagreements(pairs)
    assert not failures, "\n".join(failures)
