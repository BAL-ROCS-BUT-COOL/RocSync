# Benchmark Tools

Tools for evaluating and annotating the RocSync vision pipeline against validation images.

These are development tools: they run against a checkout, not an installed copy. Create the
environment once with `uv sync` from `RocSync/sw`, then prefix each command with `uv run` to use
it without activating anything. See the [Development section](../../README.md#development) of the
main README.

## Annotation Tool

Interactive GUI for creating ground truth annotations. Runs the pipeline on each image, displays the result with LED overlays, and lets you verify or correct the decoded values. Produces a `ground_truth.json` file for benchmark evaluation.

### Usage

```bash
uv run rocsync-annotate [data_dir] [-o ground_truth.json]
```

- `data_dir`: Directory containing validation images (default: `validation_data/`). Images are collected recursively (`.png`, `.jpg`, `.jpeg`).
- `-o`: Output file (default: `<data_dir>/ground_truth.json`).

The tool resumes from the first unannotated image on restart.

### Display

The window shows two panels side-by-side:

- **Left**: Original image
- **Right**: Rectified 640×640 board with color-coded LED overlays
  - Red circle = ON, Blue circle = OFF, Gray circle = not visible
  - If the pipeline failed (e.g., no ArUco detected), a black image is shown with LED positions at their expected locations.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Enter / Space | Accept annotation, save, advance to next image |
| D / Right arrow | Skip to next image without saving |
| A / Left arrow | Go back to previous image |
| N | Jump to next unannotated image |
| B | Jump to previous unannotated image |
| C / Backspace | Clear annotation for current image (deletes from ground truth, re-runs pipeline) |
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
    }
  }
}
```

Fields:
- `aruco.visible` / `aruco.id`: Whether the ArUco marker is visible and its detected ID (omitted when not visible).
- `corners[i].visible` / `corners[i].position`: Per-corner-LED visibility and position in original image coordinates (position omitted when not visible). Benchmark results use the same space, so the two are directly comparable.
- `homography`: 3×3 matrix mapping original image → rectified board coordinates.
- `counter.visible` / `counter.value`: Whether the binary counter is readable and its decoded value (omitted when not visible).
- `ring.start` / `ring.end`: First ON and first OFF LED indices (0–99), half-open interval `[start, end)`. `start == end` means the ring is undecodable.

## Validation Benchmark

Runs the pipeline on all images in a directory and saves per-image results in a structure mirroring the ground truth format, plus per-step timing.

```bash
uv run rocsync-validate [data_dir] [-o results.json] [--debug DIR]
```

- `data_dir`: Directory containing validation images (default: `validation_data/`).
- `-o`: Output JSON file (default: `benchmark_results.json`).
- `--debug`: Directory for debug images.

Results JSON contains a `config` section and an `images` section with per-image aruco, corners, counter, ring, timestamp, success flag, and timing breakdown. Corner positions are in original image coordinates, matching the ground truth: the pipeline detects them in the rough-rectified grid, whose scale is a property of the checkout being measured, so they are un-warped through that grid's own homography before being stored.

`config` records which checkout produced the file — branch, commit, whether the tree was dirty, run time, and the OpenCV and NumPy versions — because `rocsync-evaluate` labels its columns by filename alone.

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
- **Ring**: detection rates + start/end value accuracy
- **Overall**: timestamp detection + exact match accuracy + start/end/exposure error statistics

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

Running a checkout under the benchmark needs `rocsync/benchmark/{__init__,common,validate}.py`, the
`stats` hooks in `vision.py`, and a `rocsync-validate` entry point. `annotate.py` and `evaluate.py`
need geometry that older checkouts lack, so they are not portable and are not needed there.

Timing columns are only comparable when the environments agree — check the OpenCV and NumPy
versions that `rocsync-evaluate` prints per column before reading `-t` output.