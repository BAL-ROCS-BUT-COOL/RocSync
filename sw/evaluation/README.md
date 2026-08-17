# Multi-Camera Sync Toolkit

## ⚙️ Requirements

These scripts live in `sw/evaluation/` and import the shared timing code from the `rocsync`
package, so set that environment up once:

```bash
cd path/to/RocSync/sw
uv sync
```

Run them with `uv run` so they find the package:

```bash
uv run python evaluation/check_time_sync_all.py --dataset-folder /path/to/dataset ...
```

They read each frame's presentation timestamp out of the video itself and map it to
board time through the fit in `time_synchronization_*.json`, so a recording that
dropped frames keeps the missing span where it happened.

## 📂 Folder Setup

Your dataset folder should look like this:

```
dataset_folder/
├── raw_videos/
│   ├── camera1_raw.mp4
│   ├── camera2_raw.mp4
│   ├── camera3_raw.mp4
│   └── ...
│
└── time sync/
    ├── clips_config.json
    └── time_synchronization.json
```

### Description:

* **`raw_videos/`** – contains all raw camera recordings (e.g., `.mp4`, `.MOV`).
* **`time sync/`** – contains synchronization files:

  * `clips_config.json`: defines which clips to extract.
  * `time_synchronization_*.json`: contains timing information for each camera.

The scripts are not part of the dataset. Point them at it with `--dataset-folder`, which every
script accepts. Its default is the parent of the script's own location, so it is only useful if
you copy a script into the dataset folder; when running from `sw/evaluation/`, pass it
explicitly.

---

## ✅ Check synchronization of all clips

Once you have a finished `time_synchronization_*.json` file, check if all videos have been synchronized correctly by the RocSync script and/or the manual synchronization. Check a moment near the beginning of the Recording and a moment near the end of the Recording to make sure there is no drift. Run this before synchronizing huge datasets, it will save you a lot of time down the road.

Run the script:
```bash
uv run python evaluation/check_time_sync_all.py \
    --dataset-folder /path/to/dataset_folder \
    --time HH:MM:SS.mmm \
    --from-camera "cameraX"
```

### Arguments:

* `--time`
  The time of the reference video you want to display in this format: HH:MM:SS.mmm.

* `--from-camera`
  Name of the reference camera (e.g., `"camera1_raw"` or `"gopro7_raw"`) — this camera’s timing is used as the reference for synchronization.

---

## 🎥 Extract Synced Videos

This guide explains how to organize your dataset and run the `extract_synced_videos.py` script to generate **time-synchronized video clips** from multiple cameras.


### 🧭 Usage

Run the extraction script from your terminal:

```bash
uv run python evaluation/extract_synced_videos.py --dataset-folder /path/to/dataset_folder
```

### Optional Arguments:

* `--dataset-folder`
  Path to your dataset folder (the one containing `raw_videos/` and `time sync/`).

* `--target-fps`
  Desired output frame rate (e.g., 30).

* `--from-camera`
  Name of the reference camera (e.g., `"camera1_raw"` or `"gopro7_raw"`) — this camera’s timing is used as the reference for synchronization.

* `--time-sync-json`
  Path to your `time_synchronization_*.json` file

* `--clips-json`
  Path to the `clips_config.json` file

* `--ignore-overlap`
  If set, the script will not restrict synchronization to the time interval shared by **all** files (the global overlap).  
  Use this if some videos do **not** cover the requested timestamp/time range (e.g., some cameras started later or stopped earlier).

* `--only-for-camera`
  Add this to only sync one specific camera (basename without extension, e.g. `Cam1` or `Cam1_raw`)
  
### Output:
* Folder called synced_videos with the syncronized clips


## 📷 Extract Synced Frames
This guide explains how to organize your dataset and run the `extract_clips_as_png.py` script to generate **time-synchronized pictures** from multiple cameras.


### 🧭 Usage

Run the script from your terminal:

```bash
uv run python evaluation/extract_clips_as_png.py --dataset-folder /path/to/dataset_folder
```

### Optional Arguments:

* `--dataset-folder`
  Path to your dataset folder (the one containing `raw_videos/` and `time sync/`).

* `--target-fps`
  Desired output frame rate (e.g., 30).

* `--from-camera`
  Name of the reference camera (e.g., `"camera1_raw"` or `"gopro7_raw"`) — this camera’s timing is used as the reference for synchronization.

* `--time-sync-json`
  Path to your `time_synchronization_*.json` file

* `--clips-json`
  Path to the `clips_config.json` file

* `--only-for-camera`
  Add this to only produce PNGs for one specific camera (basename without extension, e.g. `Cam1` or `Cam1_raw`)
  

### Output:
* Folder called synced_clips_png with the synchronized png frames


