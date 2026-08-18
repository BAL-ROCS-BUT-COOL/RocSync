# Benchmark Tools

Tools for evaluating and annotating the RocSync vision pipeline against validation images and
videos.

Inputs are collected recursively: images (`.png`, `.jpg`, `.jpeg`) and videos (`.mp4`, `.mov`,
`.avi`, `.mkv`). **Every frame of a video is a benchmark frame** — video frames are decoded on
demand, so nothing is extracted to disk. A dataset should hold a video *or* its extracted frames,
not both, or the same content is benchmarked twice under two sets of keys.

These are development tools: they run against a checkout, not an installed copy. Create the
environment once with `uv sync` from `RocSync/sw`, then prefix each command with `uv run` to use
it without activating anything. See the [Development section](../../README.md#development) of the
main README.

## Annotation Tool

Interactive GUI for creating ground truth annotations. Runs the pipeline on each frame, displays the result with LED overlays, and lets you verify or correct the decoded values. Produces a `ground_truth.json` file for benchmark evaluation.

### Usage

```bash
uv run rocsync-annotate [data_dir] [-o ground_truth.json] [--fit-clocks]
```

- `data_dir`: Directory containing validation images and videos (default: `validation_data/`).
- `-o`: Output file (default: `<data_dir>/ground_truth.json`).
- `--fit-clocks`: Re-derive every video's reference clock and exit, without opening the GUI.
  Run it after editing a ground truth file by hand. It exits non-zero, naming the offending
  frames, when a video's annotations do not agree on a single clock.

The tool resumes from the first unannotated frame on restart.

### Display

The window shows two panels side-by-side:

- **Left**: Original image
- **Right**: Rectified 640×640 board with color-coded LED overlays
  - Red circle = ON, Blue circle = OFF, Gray circle = not visible
  - If the pipeline failed (e.g., no ArUco detected), a black image is shown with LED positions at their expected locations.

On a video frame the top right also carries the frame's timing residual:

```
dt +0.31 ms
fit 54 frames  RMSE 0.21  max 0.63 ms
```

`dt` is how far the annotated timestamp sits from where the rest of the video puts this
frame, green within the 2 ms tolerance and red beyond it. The clock behind it is fitted
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
      "clock_rate": 1.0000743,
      "clock_offset_ms": -4333.21,
      "pts_min_ms": 0.0,
      "pts_max_ms": 38500.0,
      "n_frames_fitted": 54,
      "rmse_ms": 0.21,
      "max_residual_ms": 0.63,
      "residual_threshold_ms": 2.0,
      "derived_at": "2026-08-18T09:12:00+00:00"
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
  extrapolates past the frames the reference was built from.

The reference clock is plain least squares over the annotated timestamps, not the RANSAC
fit `rocsync` uses on decoded ones. `rocsync-evaluate` measures fitted clocks against these
numbers, so they have to be a pure function of the annotations: derived with the production
fitter, a change to that fitter would move the ground truth and every error measured against
it at the same time. No outlier is tolerated either, which is all RANSAC would have bought —
an annotated frame more than `residual_threshold_ms` (one board LED is 1 ms) off the line is
a bad annotation, and refusing to store a reference is how it gets found.

The annotator derives a video's reference when it has at least 5 timestamp annotations and no
reference yet, or when one of its annotations was added, edited or cleared. A video nobody
touched keeps the exact number it had.

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
uv run rocsync-evaluate [paths...] [-g ground_truth.json] [-t]
```

- `paths`: Directory with benchmark `.json` files, or one or more explicit `.json` filepaths (default: `output/benchmark/`).
- `-g`: Path to ground truth JSON (default: `validation_data/ground_truth.json`).
- `-t`: Include per-step timing statistics (all/positive/negative subsets).

Metrics are computed per pipeline step:
- **ArUco**: detection rates + corner pixel error in image space
- **Corners**: detection rates in image space and board space + pixel errors in both spaces
- **Counter**: detection rates + value accuracy
- **Ring**: detection rates + start/end value accuracy, wrapping arcs included
- **Overall**: timestamp detection + exact match accuracy + start/end/exposure error statistics,
  over the frames a timestamp actually follows from
- **Clock fit** (per video, when the ground truth has a reference clock): clock rate error in
  ppm, clock offset error, sync error, residuals against the annotations, and whether the
  fit's own outlier rejection agrees with them

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

The clock-fit block reads the reference out of the ground truth and never re-fits it. Its
rows:

| Row | Meaning |
| --- | --- |
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
`ring_window`, which carries the decoded arc — and a `rocsync-validate` entry point. None of those three modules imports anything from `rocsync`
except each other, so an older checkout only needs the files copied in. `annotate.py` and `evaluate.py` need geometry that
older checkouts lack, so they are not portable and are not needed there.

`rocsync-evaluate` prints how many ground truth frames each column actually scores. A column that
scores fewer was run over a different set of inputs, and its rates describe that subset — a
prediction the run never made is skipped rather than counted as a miss.

Timing columns are only comparable when the environments agree — check the OpenCV and NumPy
versions that `rocsync-evaluate` prints per column before reading `-t` output. Video decoding is
part of that: annotations are in decoded coordinates, and OpenCV applies a container's rotation
metadata itself, so a backend that did not would put every position in a transposed frame.