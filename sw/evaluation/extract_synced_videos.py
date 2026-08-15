import argparse
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from tqdm import tqdm

try:
    from rocsync.clips import (
        Clip,
        parse_clips_json,
        read_frames_at_indices,
        select_frame_indices,
    )
    from rocsync.dataset import (
        VIDEO_SUFFIXES,
        ClipExtractionConfig,
        add_clip_args,
        add_common_args,
        load_video_time_sync,
        resolve_clips_json,
        resolve_time_sync_json,
    )
    from rocsync.timecode import ms_to_timecode
    from rocsync.timeline import affine_from_statistics, per_frame_times
except ImportError:
    raise SystemExit(
        "This script needs the rocsync package: pip install -e <path to RocSync>/sw"
    )


@dataclass
class Config(ClipExtractionConfig):
    # if True, ignore global overlap and mark short cameras with _overlap
    ignore_overlap: bool = False
    only_for_camera: Optional[str] = None


def main(config: Config) -> None:
    print(f"Processing dataset folder: {config.dataset_folder}")

    time_sync_data_all = load_video_time_sync(config.time_sync_json_path)

    def _pretty_cam_name(cam_key: str) -> str:
        """Basename without extension; also strips a trailing '_raw'."""
        name = os.path.splitext(os.path.basename(cam_key))[0]
        return name[:-4] if name.endswith("_raw") else name

    def _matches_camera_name(cam_key: str, user_value: str) -> bool:
        """
        Match user_value against:
          - exact normalized '<parent>/<file>' key
          - last 2 path segments of user_value (parent/file)
          - basename without extension (with/without _raw stripped)
        """
        uv = user_value.replace("\\", "/").strip()
        uv_base = os.path.splitext(os.path.basename(uv))[0]

        key_base = os.path.splitext(os.path.basename(cam_key))[0]
        key_base_clean = key_base[:-4] if key_base.endswith("_raw") else key_base

        # Exact match on normalized key
        if uv == cam_key.replace("\\", "/"):
            return True

        # If user provided a path, match by last 2 segments
        if "/" in uv:
            uv_key = os.path.join(*uv.split("/")[-2:])
            if uv_key == cam_key:
                return True

        return (uv_base == key_base) or (uv_base == key_base_clean)

    def _select_single_camera_key(
        all_data: Dict[str, dict], user_value: str, flag: str
    ) -> str:
        matches = [k for k in all_data.keys() if _matches_camera_name(k, user_value)]
        if not matches:
            available = ", ".join(
                sorted({_pretty_cam_name(k) for k in all_data.keys()})
            )
            raise ValueError(
                f"{flag} '{user_value}' did not match any camera.\n"
                f"Available cameras: [{available}]"
            )
        if len(matches) > 1:
            opts = ", ".join(sorted(matches))
            raise ValueError(
                f"{flag} '{user_value}' is ambiguous; matches multiple cameras.\n"
                f"Disambiguate by passing one of these exact keys (last 2 path segments): [{opts}]"
            )
        return matches[0]

    def _compute_overlap(data_dict: Dict[str, dict]) -> tuple[float, float]:
        first_frames = [float(d["first_frame"]) for d in data_dict.values()]
        last_frames = [float(d["last_frame"]) for d in data_dict.values()]
        return max(first_frames), min(last_frames)

    def _cameras_not_covering_clip(data_dict: Dict[str, dict], clip: Clip) -> List[str]:
        bad = []
        for cam_key, d in data_dict.items():
            cam_first = float(d["first_frame"])
            cam_last = float(d["last_frame"])
            if clip.start < cam_first or clip.end > cam_last:
                bad.append(_pretty_cam_name(cam_key))
        return sorted(bad)

    if not time_sync_data_all:
        raise ValueError("Time synchronization JSON is empty – no cameras found.")

    # --- Parse clips (use ALL cameras for --from-camera resolution, even if we later filter extraction) ---
    if config.from_raw_camera_time_of_camera:
        defining_key = _select_single_camera_key(
            time_sync_data_all,
            config.from_raw_camera_time_of_camera,
            "--from-camera",
        )
        defining = time_sync_data_all[defining_key]
        defining_clock_rate, defining_clock_offset_ms = affine_from_statistics(defining)
        clips = parse_clips_json(
            config.clips_to_extract_json,
            from_camera_time=True,
            clock_offset_ms=defining_clock_offset_ms,
            clock_rate=defining_clock_rate,
        )
    else:
        clips = parse_clips_json(config.clips_to_extract_json)

    # --- Apply optional extraction filter AFTER clip parsing ---
    if getattr(config, "only_for_camera", None):
        only_key = _select_single_camera_key(
            time_sync_data_all,
            config.only_for_camera,
            "--only-for-camera",
        )
        time_sync_data: Dict[str, dict] = {only_key: time_sync_data_all[only_key]}
    else:
        time_sync_data = time_sync_data_all

    to_time_str = ms_to_timecode

    # Compute overlap for:
    # - "all cameras" (diagnostics)
    # - "active cameras" (the ones we actually process)
    global_overlap_start, global_overlap_end = _compute_overlap(time_sync_data_all)
    active_overlap_start, active_overlap_end = _compute_overlap(time_sync_data)

    # --- ACTIVE OVERLAP CHECK (optionally ignored) ---
    if active_overlap_start >= active_overlap_end:
        msg = (
            f"No temporal overlap between active cameras: "
            f"max(first_frame)={active_overlap_start:.2f} ms ({to_time_str(int(active_overlap_start))}) >= "
            f"min(last_frame)={active_overlap_end:.2f} ms ({to_time_str(int(active_overlap_end))})."
        )
        if not config.ignore_overlap:
            raise ValueError(msg)
        else:
            print("[WARN]", msg)
            print(
                "[WARN] --ignore-overlap is set, proceeding anyway (per-camera brute-force)."
            )

    # Per-clip overlap validation
    for clip in clips:
        if clip.start < active_overlap_start or clip.end > active_overlap_end:
            # Cameras that do NOT fully cover this requested clip
            active_not_included = _cameras_not_covering_clip(time_sync_data, clip)
            active_not_included_str = (
                ", ".join(active_not_included) if active_not_included else "None"
            )

            msg_lines = [
                "Requested clip is outside the common overlap of the active cameras.",
                f"  Clip: [{clip.start} ms ({to_time_str(int(clip.start))}), "
                f"{clip.end} ms ({to_time_str(int(clip.end))})]",
                f"  Overlap (active cameras): "
                f"[{active_overlap_start:.2f} ms ({to_time_str(int(active_overlap_start))}), "
                f"{active_overlap_end:.2f} ms ({to_time_str(int(active_overlap_end))})]",
                f"  Not included in overlap (active): [{active_not_included_str}]",
            ]

            # If we're filtering to a single camera, also show the global (all-cameras) diagnostic overlap
            if len(time_sync_data) != len(time_sync_data_all):
                global_not_included = _cameras_not_covering_clip(
                    time_sync_data_all, clip
                )
                global_not_included_str = (
                    ", ".join(global_not_included) if global_not_included else "None"
                )
                msg_lines.extend(
                    [
                        f"  Overlap (all cameras): "
                        f"[{global_overlap_start:.2f} ms ({to_time_str(int(global_overlap_start))}), "
                        f"{global_overlap_end:.2f} ms ({to_time_str(int(global_overlap_end))})]",
                        f"  Not included in overlap (all): [{global_not_included_str}]",
                    ]
                )

            msg_lines.append("")
            msg_lines.append(
                "Choose start >= latest camera start and end <= earliest camera end, "
                "or pass --ignore-overlap to process anyway."
            )

            msg = "\n".join(msg_lines)

            if not config.ignore_overlap:
                raise ValueError(msg)
            else:
                print("[WARN]", msg)
                print(
                    "[WARN] --ignore-overlap is set, continuing for per-camera extraction.\n"
                )

    # --- MAIN EXTRACTION LOOP ---
    for clip in clips:
        for camera, camera_time_sync_data in time_sync_data.items():
            raw_video_path = os.path.join(config.dataset_folder, camera)
            print(f"Processing {raw_video_path}.")
            if not os.path.exists(raw_video_path):
                print(f"No video found at {raw_video_path}. Will skip this camera.")
                continue

            suffix = Path(raw_video_path).suffix.lower()
            if suffix not in VIDEO_SUFFIXES:
                print(f"Not a video: {raw_video_path}. Will skip this camera.")
                continue

            # Per-camera coverage vs clip
            cam_first = float(camera_time_sync_data["first_frame"])
            cam_last = float(camera_time_sync_data["last_frame"])
            has_full_overlap = (cam_first <= clip.start) and (clip.end <= cam_last)

            # Build per-frame real-time timestamps from the frames' own
            # presentation timestamps, so dropped frames stay where they are
            actual_timestamps = per_frame_times(raw_video_path, camera_time_sync_data)

            # One source frame per output frame at target_fps
            frames_to_extract = select_frame_indices(
                config.target_fps, actual_timestamps, clip.start, clip.end
            )

            # --- Clean camera name and define output path ---
            camera_name_raw = os.path.splitext(os.path.basename(camera))[0]

            # 1. Remove "_raw" suffix if present
            if camera_name_raw.endswith("_raw"):
                camera_name_cleaned = camera_name_raw[:-4]
            else:
                camera_name_cleaned = camera_name_raw

            # 2. If ignoring overlap and this camera is too short, mark with _overlap
            if config.ignore_overlap and not has_full_overlap:
                camera_name_cleaned = f"{camera_name_cleaned}_overlap"

            # 3. Set output filename to *_synced.mp4
            output_video_filename = f"{camera_name_cleaned}_synced.mp4"
            output_video_path = os.path.join(
                config.dataset_folder,
                "synced_videos",
                f"{clip.start_string_formatted}-{clip.end_string_formatted}",
                output_video_filename,
            )
            # --- END MODIFICATION ---

            # Skip if file already exists
            if os.path.exists(output_video_path):
                print(
                    f"Output video already exists at {output_video_path}. Skipping synchronization."
                )
                continue

            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

            # Write exactly those sampled frames (not a contiguous range)
            write_sampled_video_by_indices(
                video_path=raw_video_path,
                frame_indices=frames_to_extract,
                output_path=output_video_path,
                target_fps=config.target_fps,
            )


def write_sampled_video_by_indices(
    video_path: str, frame_indices: List[int], output_path: str, target_fps: float
):
    """
    Writes a video composed EXACTLY of the frames at `frame_indices`, in that order,
    encoded at `target_fps`. One frame per index, unless the source runs out early.
    """
    if not frame_indices:
        print(f"[write_sampled_video_by_indices] No frames requested for {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    try:
        for _, frame in read_frames_at_indices(
            video_path,
            frame_indices,
            desc=f"Writing {os.path.basename(output_path)} (sampled)",
        ):
            out.write(frame)
    finally:
        out.release()


def cut_video(
    video_path: str,
    start_frame: int,
    end_frame: int,
    output_path: str,
    target_fps: float = None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) if target_fps is None else target_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, start_frame)
    end_frame = min(end_frame, total_frames - 1)
    n_frames = end_frame - start_frame + 1
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pbar = tqdm(total=n_frames, desc=f"Cutting {os.path.basename(video_path)}")
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        pbar.update(1)
    cap.release()
    out.release()
    pbar.close()


def cut_video_threaded(
    video_path: str,
    start_frame: int,
    end_frame: int,
    output_path: str,
    target_fps: float = None,
):
    if start_frame >= end_frame:
        print(f"[cut_video_threaded] Invalid frame range: {start_frame} >= {end_frame}")
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = target_fps or original_fps
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(0, min(end_frame, total_frames - 1))
    n_frames = end_frame - start_frame + 1
    print(f"[Main] Cutting frames {start_frame}–{end_frame} from {video_path}")
    print(f"[Main] FPS={fps}, Resolution={width}x{height}, Frames={n_frames}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_queue = queue.Queue(maxsize=64)
    stop_token = object()
    write_lock = threading.Lock()
    pbar = tqdm(total=n_frames, desc=f"Saving {os.path.basename(output_path)}")

    def extractor():
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put(frame)
        frame_queue.put(stop_token)
        cap.release()

    def writer():
        while True:
            frame = frame_queue.get()
            if frame is stop_token:
                frame_queue.task_done()
                break
            with write_lock:
                out.write(frame)
            pbar.update(1)
            frame_queue.task_done()
        out.release()
        pbar.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(extractor), executor.submit(writer)]
        futures[0].result()
        frame_queue.join()
        futures[1].result()
    print(f"[Main] Finished writing {output_path}")


def _pretty_cam_name(cam_key: str) -> str:
    """Basename without extension; also strips a trailing '_raw'."""
    name = os.path.splitext(os.path.basename(cam_key))[0]
    return name[:-4] if name.endswith("_raw") else name


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract synchronized clips at a fixed target FPS."
    )
    add_common_args(p, __file__)
    add_clip_args(p)
    p.add_argument(
        "--ignore-overlap",
        action="store_true",
        help=(
            "Ignore global common-overlap checks and process clips anyway. "
            "Cameras that do not fully cover the clip will get an '_overlap' "
            "suffix in their output filename."
        ),
    )
    p.add_argument(
        "--only-for-camera",
        dest="only_for_camera",
        help=(
            "Only sync this camera (basename without extension, e.g. 'Cam1' or 'Cam1_raw'). "
            "If ambiguous, you can pass 'parent_folder/filename.ext' and it will match the last 2 path segments."
        ),
    )
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_folder = Path(args.dataset_folder)
    time_sync_path = resolve_time_sync_json(dataset_folder, args.time_sync_json_path)
    clips_path = resolve_clips_json(dataset_folder, args.clips_to_extract_json)

    cfg = Config(
        dataset_folder=str(dataset_folder),
        time_sync_json_path=str(time_sync_path),
        clips_to_extract_json=str(clips_path),
        target_fps=float(args.target_fps),
        from_raw_camera_time_of_camera=args.from_raw_camera_time_of_camera,
        ignore_overlap=args.ignore_overlap,
        only_for_camera=args.only_for_camera,
    )
    main(cfg)
