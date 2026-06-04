#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os
import os.path as osp
from datetime import datetime

import cv2
import numpy as np

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


def save_tracking_label_tif_sequence(
    frames_dict: dict[int, ndarray], save_folder: str, progress_cb=None
) -> tuple[str, int]:
    """
    Save combined label frames as an 8-bit TIF image sequence.
    Files are named 00001.tif, 00002.tif, ... in ascending frame-index order.

    'progress_cb', if given, is called as progress_cb(stage, current, total).

    Returns:
        save_folder, num_saved_frames
    """

    os.makedirs(save_folder, exist_ok=True)
    sorted_keys = sorted(frames_dict.keys())
    total = len(sorted_keys)
    num_saved = 0
    for seq_idx, frame_idx in enumerate(sorted_keys, start=1):
        save_path = osp.join(save_folder, f"{seq_idx:05d}.tif")
        label_img = frames_dict[frame_idx]
        if label_img.dtype != np.uint8:
            label_img = label_img.astype(np.uint8)
        if not cv2.imwrite(save_path, label_img):
            raise IOError(f"Failed to write tracking result image: {save_path}")
        num_saved += 1
        if progress_cb is not None:
            progress_cb("Saving label images", num_saved, total)

    return save_folder, num_saved
