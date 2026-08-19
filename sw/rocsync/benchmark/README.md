# Benchmark Tools

Tools for evaluating and annotating the RocSync vision pipeline against validation images and
videos.

Inputs are collected recursively: images (`.png`, `.jpg`, `.jpeg`) and videos (`.mp4`, `.mov`,
`.avi`, `.mkv`). **Every frame of a video is a benchmark frame** — video frames are decoded on
demand, so nothing is extracted to disk. A dataset should hold a video *or* its extracted frames,
not both, or the same content is benchmarked twice under two sets of keys. A
[retimed clip](#retimed-videos) is the one exception: it sits beside its source and shadows it,
so validation and scoring see one of the two and annotation the other.

These are development tools: they run against a checkout, not an installed copy. Create the
environment once with `uv sync` from `RocSync/sw`, then prefix each command with `uv run` to use
it without activating anything. See the [Development section](../../README.md#development) of the
main README.

## Annotation Tool

Interactive GUI for creating ground truth annotations. Runs the pipeline on each frame, displays the result with LED overlays, and lets you verify or correct the decoded values. Produces a `ground_truth.json` file for benchmark evaluation.

### Usage

```bash
uv run rocsync-annotate [data_dir] [-o ground_truth.json] [--fit-clocks] [--prune [--dry-run]]
```

- `data_dir`: Directory containing validation images and videos (default: `validation_data/`).
- `-o`: Output file (default: `<data_dir>/ground_truth.json`).
- `--fit-clocks`: Re-derive every video's reference clock and exit, without opening the GUI.
  Run it after editing a ground truth file by hand. It exits non-zero, naming the offending
  frames, when a video's annotations do not agree on a single clock.
- `--prune`: Remove entries whose input is gone and exit, without opening the GUI. See
  [Removing inputs from the benchmark](#removing-inputs-from-the-benchmark).
- `--dry-run`: With `--prune`, list what would go without writing anything.

The tool resumes from the first unannotated frame on restart, and names on startup any entry
whose image or video it no longer finds.

### Display

The window shows two panels side-by-side:

- **Left**: Original image
- **Right**: Rectified 640×640 board with color-coded LED overlays, drawn inside a 10 px margin
  so LEDs that a loose fit pushes past the board edge stay visible and draggable
  - Red circle = ON, Blue circle = OFF, Gray circle = not visible
  - When the pipeline found the ArUco marker but failed to detect the corner LEDs, the board is
    rectified from the marker alone and the counter and ring are read off that coarse view. It is
    an estimate to correct, not an annotation: refine the corner LEDs and the fit tightens.
  - The pipeline's minimum marker-area gate is off here, so a board held far from the camera is
    still offered for annotation; whether the benchmark run rejects it is measured separately.
  - If no ArUco marker was detected, a black image is shown with LED positions at their expected locations.

On a video frame the top right also carries the frame's timing residual:

```
dt +0.31 ms
fit 54 frames  RMSE 0.21  max 0.63 ms
```

`dt` is how far the annotated timestamp sits from where the rest of the video puts this
frame, green within [the video's tolerance](#ground-truth-format) and red beyond it. The clock behind it is fitted
over the video's *other* annotated frames and asked to predict this one, so a frame cannot
drag the line towards itself and hide half its own error. It reads the annotation on screen
rather than the saved one, so a correction shows in the number before it is accepted.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Enter / Space | Accept annotation, save, advance to next frame |
| D / Right arrow | Skip to next frame without saving |
| A / Left arrow | Go back to previous frame |
| N | Jump to next unannotated frame |
| B | Jump to previous unannotated frame |
| . | Jump to the first frame of the next input file |
| , | Jump to the first frame of the previous input file |
| C / Backspace | Clear annotation for current frame (deletes from ground truth, re-runs pipeline) |
| Q | Quit |
| H | Toggle help overlay |
| 0-9 | Select which corner the next left-panel click places |
| Esc | Cancel ring LED selection |

### Mouse Interactions (left panel)

**Placing corners** — Click a corner directly in the image to place the pending corner there. The click marks that corner visible and advances the pending corner to the next one, so clicking the corners in order (0, 1, 2, …) annotates them in one pass and then wraps back to corner 0. Press `0`-`9` to jump to a specific corner. The pending corner is ringed in green and named in the top-left of the panel. A zoom inset follows the cursor so clicks stay accurate despite the downscaled display.

**Corner LEDs** — Drag a corner LED circle to refine its position.

### Mouse Interactions (right panel)

**Corner LEDs** — Click on a corner LED circle to toggle visible/hidden. Drag a corner LED to refine its position.

Every corner edit re-fits the homography from the visible corner LEDs and, on a frame whose
counter and ring you have not yet annotated by hand, re-reads both off the new fit. Toggling a
counter LED or setting the ring hands that reading over to you, and the auto-decode leaves it
alone from then on; so does opening a frame that was already annotated.

**Counter LEDs** — Click on a counter LED circle to toggle ON/OFF. Click the counter bounding box (outside individual LEDs) to toggle counter visibility.

**ArUco area** — Click the ArUco region to cycle through marker IDs (0 → 21 → none).

**Ring LEDs** — Two-click protocol: click a ring LED to set the first ON LED (arc start), then click another to set the first OFF LED (arc end). Clicking the same LED for both sets the ring as undecodable. Press `Esc` to cancel after the first click.

### Ground Truth Format

Annotations are saved to a JSON file with the following structure:

Each key under `images` names one benchmark frame. A still image is keyed by its path relative to
`data_dir`; a video frame adds `#` and its absolute frame index, zero-padded to six digits, so a
video's frames sort in playback order next to the file itself. The index is the frame's position in
the file rather than a presentation timestamp, which would shift with the decoder.

```json
{
  "images": {
    "subdir/image.png": {
      "aruco": {"visible": true, "id": 0},
      "corners": [
        {"visible": true, "position": [123.4, 567.8]},
        {"visible": true, "position": [890.1, 567.8]},
        {"visible": false},
        {"visible": true, "position": [123.4, 890.1]}
      ],
      "homography": [[...], [...], [...]],
      "counter": {"visible": true, "value": 42},
      "ring": {"start": 10, "end": 55}
    },
    "subdir/clip.mp4#000042": {
      "...": "same fields; frame 42 of subdir/clip.mp4"
    }
  }
}
```

A `videos` section holds one reference clock per video, the affine map from container
presentation time to board time that the video's annotations agree on:

```json
{
  "images": {"...": "..."},
  "videos": {
    "subdir/clip.mp4": {
      "timeline": "measured",
      "clock_rate": 1.0000743,
      "clock_offset_ms": -4333.21,
      "pts_min_ms": 0.0,
      "pts_max_ms": 38500.0,
      "n_frames_fitted": 54,
      "rmse_ms": 0.21,
      "max_residual_ms": 0.63,
      "residual_threshold_ms": 11.111,
      "source_frame_period_ms": 33.333,
      "derived_at": "2026-08-18T09:12:00+00:00"
    },
    "subdir/clip.retimed.mp4": {
      "timeline": "synthesized",
      "source": "subdir/clip.mp4",
      "source_frame_offset": 0,
      "anchors_digest": "sha256:9f2c1a4b7d3e5068",
      "n_frames": 72,
      "residual_threshold_ms": 2.0,
      "...": "same clock fields as above"
    }
  }
}
```

Fields:
- `aruco.visible` / `aruco.id`: Whether the ArUco marker is visible and its detected ID (omitted when not visible).
- `corners[i].visible` / `corners[i].position`: Per-corner-LED visibility and position in original image coordinates (position omitted when not visible). Benchmark results use the same space, so the two are directly comparable.
- `homography`: 3×3 matrix mapping original image → rectified board coordinates.
- `counter.visible` / `counter.value`: Whether the binary counter is readable and its decoded value (omitted when not visible).
- `ring.start` / `ring.end`: First ON and first OFF LED indices (0–99), half-open interval
  `[start, end)`. `start == end` means no arc was readable. Annotate the arc as it appears,
  including one that wraps the end of the period (`start > end`) — reading the arc and timing
  it are separate things to be right about, and the benchmark scores them separately.
- `videos[path]`: `board_ms = clock_rate * pts_ms + clock_offset_ms`, with `pts_min_ms` and
  `pts_max_ms` spanning the *annotated* frames rather than the file, so scoring never
  extrapolates past the frames the reference was built from. Timestamps are as a decoder
  reports them, relative to the stream's start time, which is also what the pipeline reads.
- `videos[path].timeline`: `measured` for a recording, whose clock is fitted over its
  annotations, or `synthesized` for a [retimed clip](#retimed-videos), whose clock was drawn
  and whose timestamps were built to match it.
- `videos[path].residual_threshold_ms`: How far an annotated frame may sit from the clock. A
  synthesized timeline gets 2 ms, one ring LED at each end of the arc, because it is built from
  the annotations themselves. A measured one gets a third of a source frame, clamped to
  [2, 50] ms — tighter than the board itself resolves is meaningless, and a tolerance
  approaching the 100 ms ring period would stop flagging a whole counter step. The stored number is what the reference was checked
  against and what scoring reports, so a frozen reference stays valid under the rule it was
  frozen by.
- `videos[path].source_frame_period_ms`: Frame period of the recording, read from the packet
  duration rather than the timestamp spacing. A clip holding every 16th frame of a 30 fps
  recording still says 33.3 ms per packet while its timestamps sit 533 ms apart, and it is the
  camera's own rate that the tolerance has to scale from.
- `videos[path].source` / `source_frame_offset` / `anchors_digest` / `n_frames`: Present on a
  retimed clip only. Frame *j* of the clip is frame *j + source_frame_offset* of `source`, which
  is where its annotations live. The digest covers the annotations it was built from and
  `n_frames` what it ended up spanning, which together tell `rocsync-retime` when it has gone
  stale.

The reference clock is plain least squares over the annotated timestamps, not the RANSAC
fit `rocsync` uses on decoded ones. `rocsync-evaluate` measures fitted clocks against these
numbers, so they have to be a pure function of the annotations: derived with the production
fitter, a change to that fitter would move the ground truth and every error measured against
it at the same time. No outlier is tolerated either, which is all RANSAC would have bought —
an annotated frame further than `residual_threshold_ms` off the line is a bad annotation, and
refusing to store a reference is how it gets found. The annotator shows each frame's
leave-one-out residual against that same tolerance while you work.

The annotator derives a video's reference when it has at least 5 timestamp annotations and no
reference yet, or when one of its annotations was added, edited or cleared. A video nobody
touched keeps the exact number it had.

## Retimed Videos

A recording's container timestamps are not a record of when the camera exposed. The phone clips in
the dataset write a near-nominal grid while the sensor wanders ±7 ms around it, which puts a floor
under any clock fitted against them: fitting the undecimated original over 650 frames gives the
same 3.2 ms as the decimated copy, so the information was never written rather than lost in
processing.

That floor is the honest answer to how well a phone can be synchronized, and far too coarse to
catch a regression in decoding or fitting — a 3 ms error and a correct answer look alike.
`rocsync-retime` produces the other kind of benchmark video: the annotated board times become the
timeline, so the clock the pipeline should recover is known exactly instead of fitted. Both kinds
coexist in a dataset, labelled `measured` and `synthesized`, because each answers a question the
other cannot.

```bash
uv run rocsync-retime [data_dir] [-o ground_truth.json] [--jitter-ms 0] [--force]
```

- `data_dir`: Directory containing validation images and videos (default: `validation_data/`).
- `-o`: Ground truth JSON to read and update (default: `<data_dir>/ground_truth.json`).
- `--jitter-ms`: Gaussian noise on frames without an annotation. Anchors stay exact — perturbing
  them would put back the noise the retiming exists to remove (default: 0).
- `--force`: Rebuild clips already up to date.

This needs PyAV, which is in the `dev` dependency group: the tool authors datasets rather than
reading them, so a checkout has it after `uv sync` and an installed copy does not.

### What it writes

`clip.retimed.mp4` beside `clip.mp4`, a **stream copy** with rewritten packet timestamps. Pixels
stay bit-identical, which is what lets every existing corner, homography and LED annotation carry
over — they are what the clip is built from. Non-video streams are dropped and the output uses a
1/90000 time base, fine enough that rounding is negligible against the 2 ms tolerance.

Each clip's `clock_rate` is drawn from `U(1 ± 0.05)` — the widest deviation that stays inside
`process_video`'s own sanity warning — seeded on the source's relative path, so the target is never
trivially `1.0` and a regenerated clip reproduces. The offset follows from where the timeline
lands, in the tens of seconds. Anchors satisfy `clock_rate · pts + clock_offset_ms == board_time`
by construction, and the tool verifies that by reading the written file back before recording it.

### What it spans

The clip is trimmed to `[first annotated frame, last annotated frame]`, so no timestamp is ever
extrapolated and the strict tolerance holds across the whole file. A stream copy cannot start
mid-GOP, so the output actually starts at the last keyframe at or before the first anchor and may
carry a few frames past the last one; those continue the adjacent segment's slope and are not
anchors, so nothing checks them. The tool refuses a clip needing more than five such margin frames.

Every frame inside the window is kept, annotated or not — partially visible boards, wrapping ring
arcs, frames with no board at all. A clip of only cleanly decodable frames would stop exercising
the rejection paths that matter. Only frames with a reconstructable annotated timestamp anchor the
timeline; the rest are interpolated between anchors and still carry their real capture jitter,
which is the point.

### Where the annotations live

With the source, always. A retimed clip is a derived artifact that can be regenerated or deleted
without touching them, and its ground truth entry records `source` and `source_frame_offset` so a
prediction about its frame *j* resolves back to frame *j + offset* of the recording.

That is also why **the annotator walks the recordings and never a retimed clip**: the sources are
untrimmed, so frames outside the current window stay annotatable, which is how a window grows.
`rocsync-validate` and `rocsync-evaluate` do the opposite — a recording with a retimed clip beside
it is skipped in favour of it, so the same footage is never benchmarked twice.

### Re-running

Safe after every annotation session. Each entry stores an `anchors_digest` over the annotations it
was built from; a clip whose digest still matches and whose file is intact is reported as up to
date and left alone. Only clips whose annotations actually moved get rewritten, and `--force`
overrides that.

## Validation Benchmark

Runs the pipeline on every frame in a directory and saves per-frame results in a structure mirroring the ground truth format, plus per-step timing.

```bash
uv run rocsync-validate [data_dir] [-o results.json] [--debug DIR]
```

- `data_dir`: Directory containing validation images and videos (default: `validation_data/`).
- `-o`: Output JSON file (default: `benchmark_results.json`).
- `--debug`: Directory for debug images.

Results JSON contains a `config` section and an `images` section with per-frame aruco, corners, counter, ring, timestamp, success flag, and timing breakdown. Keys match the ground truth's, so `n_images` in `config` counts benchmark frames rather than files. Corner positions are in original image coordinates, matching the ground truth: the pipeline detects them in the rough-rectified grid, whose scale is a property of the checkout being measured, so they are un-warped through that grid's own homography before being stored.

Every video additionally gets its clock fitted, by the same code a `rocsync` run uses, and a
`videos` section records the resulting `VideoStatistics` — clock rate and offset, R²/RMSE
before and after outlier rejection, frame period, dropouts and exposure. A video whose
timeline could not be fitted carries an `error` instead of the fit. Each frame of a video
gains its presentation timestamp as `pts_ms` and, once fitted, a `fit` of
`{"residual_ms", "inlier"}`, so the per-frame outcome stays next to the frame it belongs to.

Unlike a real run, which analyzes roughly one frame per second, every frame that decoded
feeds the fit: the benchmark has already paid to decode them all, and using all of them keeps
fit quality separate from the sampling policy.

`config` records which checkout produced the file — branch, commit, whether the tree was dirty, run time, and the OpenCV and NumPy versions — because `rocsync-evaluate` labels its columns by filename alone.

Video frames are decoded in the order they are walked, so a run never seeks backwards through a file.

## Benchmark Evaluation

Compares benchmark results against ground truth annotations. Reports per-step detection metrics (TPR, FPR, precision, F1), value accuracy, and position errors.

```bash
uv run rocsync-evaluate [paths...] [-g ground_truth.json] [-t] [--allow-partial]
```

- `paths`: Directory with benchmark `.json` files, or one or more explicit `.json` filepaths (default: `output/benchmark/`).
- `-g`: Path to ground truth JSON (default: `validation_data/ground_truth.json`).
- `-t`: Include per-step timing statistics (all/positive/negative subsets).
- `--allow-partial`: Report instead of refusing when the columns cover different frames — see
  [Comparing branches](#comparing-branches).

Metrics are computed per pipeline step:
- **ArUco**: detection rates + corner pixel error in image space
- **Corners**: detection rates in image space and board space + pixel errors in both spaces
- **Counter**: detection rates + value accuracy
- **Ring**: detection rates + start/end value accuracy, wrapping arcs included
- **Overall**: timestamp detection + exact match accuracy + start/end/exposure error statistics,
  over the frames a timestamp actually follows from
- **Clock fit** (when the ground truth has a reference clock): clock rate error in ppm, clock
  offset error, sync error, residuals against the annotations, and whether the fit's own
  outlier rejection agrees with them. Reported per group of videos rather than per video —
  measured and retimed timelines are summarized separately, because a measured one also
  carries the camera's own timestamp error. Fit errors are the mean and the worst over the
  group's videos, frame counters their totals, residuals pooled over every frame the group
  scored. A recording whose retimed clip the run scored is not reported separately

Corner positions are compared against the annotated coordinates. Board space is derived by mapping
both sides through the annotated homography, which normalises the threshold to LED sample radii —
a fixed pixel threshold in image space means different things depending on how large the board
appears in the frame.

Reading position errors: the annotator pre-fills each image from a pipeline run, so on every frame
accepted without correction the annotation *is* that pipeline's output. Position error therefore
reads as zero for whichever checkout produced the annotations, and a non-zero error for another
checkout measures divergence from it rather than distance from truth. Detection rates do not suffer
this — the ground truth's positives include everything annotated by hand on frames where the
pipeline failed.

A ring arc that wraps the end of the period was exposed across a counter increment, so the
counter no longer says which period the arc belongs to and no timestamp follows from it. The
board decides this, in `board_time_from_ring`; the benchmark asks rather than reimplementing.
Such a frame is still scored on whether the arc was read correctly, but it is not a positive
for the overall timestamp step — the pipeline is right to refuse it — and it is kept out of
every clock fit, the reference included. Annotations are unaffected: record what is on screen
and the scoring sorts out what is decodable.

The clock-fit block reads the reference out of the ground truth and never re-fits it. Read a
sync error against the `reference tolerance` on the same block: a 0.5 ms error against a
synthesized timeline is a real measurement, while against a measured one it is inside the
camera's own noise. Its rows:

| Row | Meaning |
| --- | --- |
| `timeline` | `measured` for a recording, `synthesized` for a [retimed clip](#retimed-videos) |
| `reference tolerance [ms]` | how far an annotation may sit from the reference, which is what makes those two comparable |
| `clock rate error [ppm]` | fitted rate minus reference rate; the raw difference is ~1e-5 and unreadable otherwise |
| `clock offset error [ms]` | fitted board time at `pts == 0` minus the reference's |
| `sync error, first/last/worst [ms]` | how far the two clocks disagree at either end of the annotated span |
| `outliers rejected in error` | frames the fit threw out whose decode the annotation confirms |
| `misdecodes kept as inliers` | frames the fit kept whose decode the annotation contradicts |
| `residual vs annotations` | fitted board time minus annotated board time, per frame |

Read `sync error` first. Rate and offset trade off against each other — they cancel at
`pts == 0` and diverge everywhere else — so either one alone can look bad while the clock is
fine, or look fine while the clock is not. The sync error is what actually lands on a frame.

Unlike the position errors above, this block does not favour whichever checkout pre-filled
the annotations: the reference is fitted over annotated *values*, not copied from a pipeline's
output. A decoding bias present in both annotations and predictions still cancels.

## Comparing branches

`rocsync-evaluate` takes several result files and prints one column per file, named after the file,
which is how two checkouts are compared. Only `rocsync-validate` has to run inside the checkout
under test; annotation and scoring stay here, against one reference geometry.

Every branch installs as `rocsync` at the same version, so they cannot share an environment — give
each one a worktree with its own venv:

```bash
git worktree add ../RocSync-other <branch>
cd ../RocSync-other/sw && uv sync
```

Then produce a column per checkout and compare:

```bash
mkdir -p output/benchmark
../RocSync-other/sw/.venv/bin/rocsync-validate <data_dir> -o output/benchmark/other.json
uv run rocsync-validate <data_dir> -o output/benchmark/current.json
uv run rocsync-evaluate output/benchmark/other.json output/benchmark/current.json \
    -g <data_dir>/ground_truth.json -t
```

Running a checkout under the benchmark needs `rocsync/benchmark/{__init__,common,validate}.py`,
`rocsync/dataset.py` for the suffix sets `common.py` collects inputs by, `rocsync/timeline.py` and
`rocsync/video_statistics.py` for the clock fit, the `stats` hooks in `vision.py` — including
`ring_window`, which carries the decoded arc — and a `rocsync-validate` entry point. Beyond those,
the benchmark modules reach only for `board_profiles.py` and `camera.py`, which every checkout
already has, so an older one needs nothing but the files above copied in. `annotate.py`,
`retime.py` and `evaluate.py` need geometry that older checkouts lack, so they are not portable
and are not needed there.

`rocsync-evaluate` prints how many ground truth frames each column actually scores, and refuses to
print a report when the columns do not cover the same frames: a prediction the run never made is
skipped by every metric rather than counted as a miss, so columns over different frame sets are
different populations and the counts, rates and green winner mark between them mean nothing. It
names the files behind the difference; re-run `rocsync-validate` for the column that lags, or pass
`--allow-partial` to score each column over its own coverage anyway. Frames no column scores are
reported separately as a stale ground truth, which
[pruning](#removing-inputs-from-the-benchmark) clears.

Timing columns are only comparable when the environments agree — check the OpenCV and NumPy
versions that `rocsync-evaluate` prints per column before reading `-t` output. Video decoding is
part of that: annotations are in decoded coordinates, and OpenCV applies a container's rotation
metadata itself, so a backend that did not would put every position in a transposed frame.

## Adding videos to the benchmark

I recommend to subsample videos to 1-2 fps to obtain 

The following command subsamples videos to approximately 0.9 fps, while preserving exact frame timestamps:

`ffmpeg -i input_video.mp4 -vf "select='isnan(prev_selected_t)+gte(t-prev_selected_t,1/0.9)'" -fps_mode passthrough -c:v libx264 -crf 18 -preset slow -an rocsync_benchmark/<subset>/output_video.mp4`

When selecting the output framerate, avoid multiples of 100ms to collect frames with different ring arcs.
I recommend 0.9 - 1.9 fps to minimize annotation cost.

## Removing inputs from the benchmark

Delete the file, then clean up what described it:

```bash
rm <data_dir>/<subset>/clip.mp4
uv run rocsync-annotate <data_dir> --prune
```

`--prune` removes the annotations and the reference clock of every input that is no longer there,
and nothing else. It leaves alone a file that is present but would not decode — a truncated copy,
a codec this machine lacks — because that is a reason to fix the file, not to lose the annotation
work behind it; those are listed and make the exit code non-zero. `--dry-run` prints the same list
without writing. A retimed clip is cut again on demand, so its entry survives the clip being
deleted and goes only when the recording it was cut from does.

Annotations whose video was re-encoded shorter are pruned too, by index; re-derive that video's
clock afterwards with `--fit-clocks`, since its reference was fitted over frames that are now gone.

Result files in `output/benchmark/` still hold predictions for the removed frames, so re-run
`rocsync-validate` for every column you want to keep comparing — until then `rocsync-evaluate`
refuses to put an old column next to a fresh one.
