# RocSync decoding software

This folder contains the Python application for detecting and decoding the RocSync device in videos and images where it is visible.

## How it works:
1. **Find ArUco marker**: Detect the ArUco marker to determine the approximate position and orientation.
2. **Coarse homographic reprojection**: Use the detected marker's corners to perform a coarse homographic reprojection of the image.
3. **Locate corner LEDs**: Identify the corner LEDs in the reprojected image.
4. **Accurate reprojection**: Use the corner LEDs to refine the reprojection for higher accuracy.
5. **Decode LEDs**: Decode the circle and binary counter LEDs by thresholding their general areas to obtain an exact timestamp.
6. **Timestamp fitting**: If the input was a video, perform robust linear regression on all extracted timestamps to reject outliers and estimate timestamps for all frames.

## Installation
RocSync uses [uv](https://docs.astral.sh/uv/) as its package manager. Install it first if you
do not have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install RocSync as a tool, which puts the `rocsync` and `rocsync-align` commands on your
`PATH` in their own isolated environment:

```bash
git clone https://github.com/jaromeyer/RocSync.git
uv tool install ./RocSync/sw
```

To run it once without installing anything permanently:

```bash
uvx --from ./RocSync/sw rocsync -h
```

<details>
<summary>Installing with pip instead</summary>

```bash
git clone https://github.com/jaromeyer/RocSync.git
pip install ./RocSync/sw
```
</details>

Time-syncing videos with `rocsync-align` additionally requires `ffmpeg` on your `PATH`; hardware
encoding needs an `ffmpeg` built with `hevc_nvenc`.

## Usage
```bash
rocsync [OPTIONS] PATH [PATH ...]
```

Each `PATH` is a video, an image, an FTK tracker recording (`.csv`), or a directory containing
them. Results for every input are collected into a single JSON file, keyed by path, with a
`type` of `video`, `image`, or `ftk`.

Infrared input needs an explicit `--board-version`: auto-detection identifies the board from its
ArUco marker, which is not visible in IR.

```bash
rocsync -c ir --board-version v2 ir_recording.mp4
```

| Option | Description |
| --- | --- |
| `-c, --camera_type {rgb,ir}` | Type of camera (default: `rgb`) |
| `-s, --stride N` | Scan every N-th frame only (default: same as framerate, videos only) |
| `-e, --export_frames DIRECTORY` | Directory to store all raw frames as PNGs with timestamp (videos only) |
| `-o, --output FILE` | JSON file to store results (default: `output.json`) |
| `-y, --yes` | Do not ask for confirmation when processing multiple files |
| `--debug DIRECTORY` | Directory to store debug images (very slow) |
| `--board-version {auto,v1,v2}` | Board hardware revision (default: `auto`, detected from the ArUco marker ID) |
| `--window START END` | Time span to search, in `hh:mm:ss` format with optionally fractional seconds; `end` is the end of the file and `end-hh:mm:ss` counts back from it, e.g. `--window end-0:00:30 end`. Repeat for several spans; overlapping ones are merged (default: whole file) |
| `--recurse_in_dir` | Recursively search for videos and images in directories |

Run `rocsync -h` for the authoritative list.


## Example
```
$ rocsync recording.MP4
Working on recording.MP4
Analyzing frames: 100%|████████████████████| 6520/6520 [02:21<00:00, 45.93it/s]
-----------------------------------------------------------------------
Number of considered frames:                                       1885
Number of rejected outliers:                                          3
R2 (before/after outlier rejection):                      0.9998/1.0000
RMSE (before/after outlier rejection):                     1.83/0.44 ms
Dropped frames:                              0 in 0 gap(s), max 0.000 s
First frame:                                                   -4.333 s
Last frame:                                                    22.858 s
Framerate (nominal/measured):                       239.760/239.740 fps
Clock rate (board/container):                                 1.000074x
Duration (container/board):                 27.190/27.192 s (Δ=2.00 ms)
Exposure time (mean/min/max/std):                3.97/3.00/5.00/0.42 ms
-----------------------------------------------------------------------
Processing files: 100%|█████████████████████████| 1/1 [02:22<00:00, 142.04s/it]
```

In the terminal the five checked statistics — from `Number of considered frames` down to
`Dropped frames` — are colour-coded green when they pass their sanity threshold and red when
they do not. All times are board times, which is why the first frame here is negative: the
recording started before the board's clock reached zero.

## Time-syncing videos
`rocsync-align` turns the JSON that `rocsync` wrote into actually aligned video files. It reads
each entry's fitted clock and trims every video to a common start, so the outputs can be played
back side by side. Entries that are not videos are ignored.

```bash
rocsync-align output.json --output_dir synced
```

| Option | Description |
| --- | --- |
| `--output_dir DIR` | Where synchronized videos are written (default: `synced`) |
| `--compensate-drift` | Compensate clock drift by re-encoding; significantly slower but more accurate |
| `--fps FPS` | Target frame rate (default: the source rate of the first video) |
| `--jobs N` | Maximum concurrent `ffmpeg` processes, or `0` for no limit (default: `4`) |

This needs `ffmpeg` on your `PATH`; `--compensate-drift` uses `hevc_nvenc` when the available
`ffmpeg` provides it.

### FFmpeg
To visually inspect the synchronization you can use FFmpeg to combine the aligned videos side-by-side:
```bash
ffmpeg \
	-ss 4.2803 -i video1.mp4 \
	-ss -8.1024 -i video2.mp4 \
	-filter_complex "[0:v]setpts=PTS*1.000020938577059[v0];[1:v]setpts=PTS*1.000083866934668[v1];[v0][v1]hstack=inputs=2" \
	-c:v h264_nvenc \
	-preset p1 \
	-r 30 \
	synchronized.mp4
```

## Evaluation toolkit
`evaluation/` holds scripts for working with a whole multi-camera dataset once every camera has
been time-synced:

| Script | Purpose |
| --- | --- |
| `check_time_sync_all.py` | Extract one moment from every camera to verify the sync holds, including at the end of a long recording |
| `extract_synced_videos.py` | Cut time-aligned video clips across all cameras |
| `extract_clips_as_png.py` | Cut the same clips as PNG frame sequences |

They expect a dataset folder containing `raw_videos/` and a `time sync/` subfolder. See
[`evaluation/README.md`](evaluation/README.md) for the dataset layout and the full argument list.

## Benchmarking
`rocsync/benchmark/` measures the vision pipeline against a folder of validation images:

| Tool | Purpose |
| --- | --- |
| `annotate.py` | Interactive GUI for building a `ground_truth.json` from validation images |
| `validate.py` | Run the pipeline over every image and record what it decoded, with per-step timings |
| `evaluate.py` | Score a validation run against the ground truth and report where it fails |

See [`rocsync/benchmark/README.md`](rocsync/benchmark/README.md) for the annotation shortcuts and
the full argument list.

## Development
Everyone works against the same pinned environment, described by `pyproject.toml` and locked in
`uv.lock`. Create it with:

```bash
cd RocSync/sw
uv sync
```

This creates `.venv/` with the runtime dependencies plus the `dev` group (ruff, basedpyright,
pytest) at exactly the locked versions, and installs `rocsync` itself in editable mode. Re-run
`uv sync` after pulling; it is fast and idempotent. Prefix commands with `uv run` to use that
environment without activating it.

| Task | Command |
| --- | --- |
| Run the test suite | `uv run pytest` |
| Format the code | `uv run ruff format .` |
| Check formatting only | `uv run ruff format --check .` |
| Lint | `uv run ruff check .` |
| Lint and apply safe fixes | `uv run ruff check --fix .` |
| Type-check | `uv run basedpyright` |

Ruff handles both formatting and linting, so no separate formatter is needed. All three tools
read their configuration from `pyproject.toml`; do not pass overriding flags on the command line,
otherwise results differ between machines. Run the format, lint, and type-check commands before
opening a pull request.

Some tests synthesize video fixtures with `ffmpeg` and skip themselves automatically when it is
not installed. Install `ffmpeg` to run the full suite.

To add or change a dependency, edit `pyproject.toml` and run `uv sync`, then commit the updated
`uv.lock` alongside it.

## Ideas for future improvements
- [ ] Write debug images in separate thread to not slow down the main processing
- [ ] Implement additional filters for the IR pipeline (e.g., uniform corner distance)
- [ ] Implement movement filter to reject frames with motion blur
- [ ] Use partial ring LED intensity to determine sub-millisecond timestamps (e.g., compute distance to min and max detected intensity)
- [ ] Use gray code and partial LED detection to decode the counter even when it changed during the exposure
