#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import csv
import os
import os.path as osp
from datetime import datetime

import cv2
import numpy as np

from .image_io import imread_unicode, imwrite_unicode

# For type hints
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def build_combined_label_image(frame_hw: tuple[int, int], labeled_masks: list[tuple[int, ndarray]]) -> ndarray:
    """
    Build a single 8-bit label image where each object is encoded by its object number (gray value).
    Later entries in labeled_masks overwrite earlier ones where masks overlap.
    """

    label_img = np.zeros(frame_hw, dtype=np.uint8)
    for label_value, mask_1ch in labeled_masks:
        label_img[mask_1ch > 0] = np.uint8(label_value)
    return label_img


# .....................................................................................................................


def make_mt_results_folder_name(video_base_name: str, timestamp: datetime | None = None) -> str:
    """Build the MT-Results output folder name for a given source video."""

    if timestamp is None:
        timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d-%H%M%S")
    return f"{video_base_name}_MT-Results_{timestamp_str}"


# .....................................................................................................................


def label_filename_width(frame_indices: list[int]) -> int:
    """Digit width for frame-index filenames. At least 5, wider when the index needs it."""

    if len(frame_indices) == 0:
        return 5
    return max(5, len(str(max(int(frame_idx) for frame_idx in frame_indices))))


def label_tif_filename(frame_idx: int, width: int = 5) -> str:
    """TIFF name for a video frame index, e.g. frame 240 -> 00240.tif."""

    return f"{int(frame_idx):0{int(width)}d}.tif"


def save_tracking_label_tif_sequence(
    frames_dict: dict[int, ndarray],
    save_folder: str,
    progress_cb=None,
    fps: float | None = None,
    source_info: dict | None = None,
) -> tuple[str, int]:
    """
    Save combined label frames as an 8-bit TIF image sequence.

    Each file is named with the source frame index (00240.tif is video frame 240).
    frame_index.csv maps those filenames back to frame_index and time_s.

    'progress_cb', if given, is called as progress_cb(stage, current, total).

    Returns:
        save_folder, num_saved_frames
    """

    os.makedirs(save_folder, exist_ok=True)
    sorted_keys = sorted(int(frame_idx) for frame_idx in frames_dict.keys())
    width = label_filename_width(sorted_keys)
    total = len(sorted_keys)
    num_saved = 0
    manifest_rows = []
    for frame_idx in sorted_keys:
        filename = label_tif_filename(frame_idx, width)
        save_path = osp.join(save_folder, filename)
        label_img = frames_dict[frame_idx]
        if label_img.dtype != np.uint8:
            label_img = label_img.astype(np.uint8)
        if not imwrite_unicode(save_path, label_img):
            raise IOError(f"Failed to write tracking result image: {save_path}")
        time_s = (frame_idx / fps) if fps and fps > 0 else 0.0
        manifest_rows.append({"filename": filename, "frame_index": frame_idx, "time_s": time_s})
        num_saved += 1
        if progress_cb is not None:
            progress_cb("Saving label images", num_saved, total)

    manifest_path = osp.join(save_folder, "frame_index.csv")
    with open(manifest_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("filename", "frame_index", "time_s"))
        writer.writeheader()
        writer.writerows(manifest_rows)

    if source_info:
        import json

        with open(osp.join(save_folder, "source_info.json"), "w", encoding="utf-8") as handle:
            json.dump(source_info, handle, indent=2)

    return save_folder, num_saved


def load_saved_label_frames(save_folder: str) -> dict[int, ndarray]:
    """
    Reload label images written by save_tracking_label_tif_sequence.

    Uses frame_index.csv when it is present. Otherwise treats a numeric TIFF
    stem as the frame index.
    """

    manifest_path = osp.join(save_folder, "frame_index.csv")
    frames: dict[int, ndarray] = {}
    if osp.isfile(manifest_path):
        with open(manifest_path, newline="") as handle:
            for row in csv.DictReader(handle):
                frame_idx = int(row["frame_index"])
                image_path = osp.join(save_folder, row["filename"])
                label_img = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
                if label_img is None:
                    raise IOError(f"Failed to read label image: {image_path}")
                frames[frame_idx] = label_img
        return frames

    for name in sorted(os.listdir(save_folder)):
        stem, ext = osp.splitext(name)
        if ext.lower() != ".tif" or not stem.isdigit():
            continue
        image_path = osp.join(save_folder, name)
        label_img = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
        if label_img is None:
            raise IOError(f"Failed to read label image: {image_path}")
        frames[int(stem)] = label_img
    return frames
