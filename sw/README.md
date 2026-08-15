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

Time-syncing videos (`--sync_video` and `rocsync-align`) additionally requires `ffmpeg` on your
`PATH`; hardware encoding needs an `ffmpeg` built with `hevc_nvenc`.

## Usage
```
$ rocsync -h
usage: rocsync [-h] [-c {rgb,ir}] [-s N] [-e DIRECTORY] [-o FILE] [--debug DIRECTORY] PATH [PATH ...]

Extract timestamps from images and videos showing the RocSync device.

positional arguments:
  PATH                  path to a video, image, or directory containing videos and/or images

options:
  -h, --help            show this help message and exit
  -c, --camera_type {rgb,ir}
                        specify the type of camera (default: rgb)
  -s, --stride N        scan every N-th frame only (default: same as framerate, only applies to videos)
  -e, --export_frames DIRECTORY
                        directory to store all raw frames as PNGs with timestamp (only applies to videos)
  -o, --output FILE     JSON file to store results (default: output.json)
  -y, --yes             automatically run yes for all prompts (potentially overwrites existing files)
  --debug DIRECTORY     directory to store debug images (very slow)

  --window START END    time span to search, in hh:mm:ss format; 'end' is the end of the file and
                        'end-hh:mm:ss' counts back from it, e.g. --window end-0:00:30 end. Repeat
                        for several spans; overlapping ones are merged (default: whole file)

  --sync_video          automatically time-sync the videos using the estimated timestamps (requires ffmpeg)
  --synced_folder FOLDER 
                        output folder for time-synced videos
  --fps FPS             desired FPS for time-synced videos (default: desired FPS determined from input videos)
  ```


## Example
```
$ rocsync ./examples/h10.MP4 ./examples
Working on ./examples/h10.MP4
Analyzing frames: 100%|████████████████████| 6520/6520 [02:21<00:00, 45.93it/s]
Number of considered frames:                             1888
Number of rejected outliers:                                0
R2 score:                                              1.0000
RMSE:                                                 0.44 ms
First frame:                                       -4333.4 ms
Last frame:                                        22858.4 ms
Expected duration (fps):              27189.7 ms (239.76 fps)
Actual duration (fps):                27191.7 ms (239.74 fps)
Delta (actual - expected)              2.08 ms (0.010% speed)
Exposure time (mean/min/max):               3.97/3.00/5.00 ms
-------------------------------------------------------------
Processing files: 100%|█████████████████████████| 1/1 [02:22<00:00, 142.04s/it]
```

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
