#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sparse storage for recorded label frames. Author: Lucien, 2026-09-23.

Each object is stored as a packed bit crop of its bounding box. Save and
analysis still receive a dense uint8 label image.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def compress_label(label_img: np.ndarray) -> dict:
    """Pack each non-zero label into its bounding crop."""

    image = np.asarray(label_img)
    if image.ndim != 2:
        raise ValueError("A label image must be 2-D")
    height, width = int(image.shape[0]), int(image.shape[1])
    objects: dict[int, tuple[int, int, int, int, np.ndarray]] = {}
    for value in np.unique(image):
        value_int = int(value)
        if value_int == 0:
            continue
        ys, xs = np.nonzero(image == value)
        y0, x0 = int(ys.min()), int(xs.min())
        crop_h = int(ys.max()) - y0 + 1
        crop_w = int(xs.max()) - x0 + 1
        crop = image[y0 : y0 + crop_h, x0 : x0 + crop_w] == value
        objects[value_int] = (y0, x0, crop_h, crop_w, np.packbits(crop.reshape(-1)))
    return {"hw": (height, width), "objects": objects}


def decompress_label(stored: dict) -> np.ndarray:
    """Rebuild a dense uint8 label image from packed object crops."""

    height, width = stored["hw"]
    label = np.zeros((int(height), int(width)), dtype=np.uint8)
    for value, (y0, x0, crop_h, crop_w, packed) in stored["objects"].items():
        bits = np.unpackbits(np.asarray(packed))[: crop_h * crop_w].reshape(crop_h, crop_w)
        region = label[y0 : y0 + crop_h, x0 : x0 + crop_w]
        region[bits.astype(bool)] = np.uint8(value)
    return label


@dataclass
class TrackingResultsBuffer:
    """Per-frame labels stored as packed object crops."""

    _frames: dict[int, dict] = field(default_factory=dict)

    @classmethod
    def create(cls) -> "TrackingResultsBuffer":
        return cls()

    def clear(self) -> "TrackingResultsBuffer":
        self._frames.clear()
        return self

    def has_data(self) -> bool:
        return len(self._frames) > 0

    def record_frame(self, frame_idx: int, label_img: np.ndarray) -> "TrackingResultsBuffer":
        self._frames[int(frame_idx)] = compress_label(label_img)
        return self

    def erase_label(self, stable_id: int) -> "TrackingResultsBuffer":
        """Remove one object id from frames already stored in memory."""

        value = int(stable_id)
        for stored in self._frames.values():
            stored["objects"].pop(value, None)
        return self

    def contains_label(self, stable_id: int) -> bool:
        value = int(stable_id)
        return any(value in stored["objects"] for stored in self._frames.values())

    @property
    def frames_dict(self) -> dict[int, np.ndarray]:
        """Dense label images, decoded for save and analysis."""

        return {frame_idx: decompress_label(stored) for frame_idx, stored in self._frames.items()}
