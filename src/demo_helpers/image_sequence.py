#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Frame source for multipage TIFF stacks and folders of still images.

The public methods match the video reader used by the playback loop and slider.
"""


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os
import os.path as osp
import re
from collections import OrderedDict

import cv2
import numpy as np
import tifffile

from .image_io import imread_unicode

# For type hints
from numpy import ndarray


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
STACK_EXTENSIONS = {".tif", ".tiff"}


# ---------------------------------------------------------------------------------------------------------------------
# %% Path helpers


def natural_sort_key(name: str) -> list:
    """Sort frame_2 before frame_10."""

    parts = re.split(r"(\d+)", name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_image_files(folder: str) -> list[str]:
    """Return still-image paths in a folder, in natural filename order."""

    names = [
        name
        for name in os.listdir(folder)
        if osp.splitext(name)[1].lower() in IMAGE_EXTENSIONS and osp.isfile(osp.join(folder, name))
    ]
    names.sort(key=natural_sort_key)
    return [osp.join(folder, name) for name in names]


def is_image_sequence_path(path: str) -> bool:
    """True for a directory of images or a still/stack image file."""

    if osp.isdir(path):
        return len(list_image_files(path)) > 0
    return osp.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def open_frame_source(path: str, intensity_range: str | tuple[float, float] = "auto"):
    """
    Open a video file, a TIFF stack, a still image, or a folder of stills.

    Video containers stay on the OpenCV video reader. Image paths use ImageSequenceReader.
    intensity_range applies to 16-bit stills: "auto", "full", or (low, high).
    """

    from .ui.video import ReversibleLoopingVideoReader

    if osp.isdir(path) or osp.splitext(path)[1].lower() in IMAGE_EXTENSIONS:
        return ImageSequenceReader(path, intensity_range=intensity_range)
    return ReversibleLoopingVideoReader(path)


def make_uint16_lut(low: float, high: float) -> ndarray:
    """Map uint16 values from [low, high] onto 0..255. One table is shared by every frame."""

    span = max(float(high) - float(low), 1.0)
    values = np.arange(65536, dtype=np.float32)
    return np.clip((values - float(low)) * (255.0 / span), 0, 255).astype(np.uint8)


def to_bgr_uint8(image: ndarray, uint16_lut: ndarray | None = None) -> ndarray:
    """Convert a decoded still to 3-channel uint8 BGR for the display and the model."""

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.dtype == np.uint16:
        if uint16_lut is None:
            image = (image.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)
        else:
            image = uint16_lut[image]
    elif image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.ascontiguousarray(image)


def read_tiff_page(
    path: str, index: int, encoded: ndarray | None = None
) -> tuple[ndarray | None, ndarray | None]:
    """
    Read one TIFF page.

    Returns (page, encoded_bytes). encoded_bytes is set when decoding had to
    use the in-memory OpenCV fallback, so the caller can reuse those bytes.
    The page is None when it does not exist or cannot be decoded.
    """

    index = int(index)
    if encoded is None:
        try:
            with tifffile.TiffFile(path) as tif:
                if index < 0 or index >= len(tif.pages):
                    return None, None
                page = tif.pages[index].asarray()
                if page is not None and page.size > 0:
                    return page, None
        except Exception:
            pass
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
        except OSError:
            return None, None
    if encoded is None or encoded.size == 0:
        return None, encoded
    ok, pages = cv2.imdecodemulti(encoded, cv2.IMREAD_UNCHANGED, None, (index, index + 1))
    if not ok or pages is None or len(pages) == 0:
        return None, encoded
    page = pages[0]
    if page is None or page.size == 0:
        return None, encoded
    return page, encoded


def tiff_page_count(path: str) -> int:
    """Count TIFF pages without decoding them."""

    try:
        with tifffile.TiffFile(path) as tif:
            return len(tif.pages)
    except Exception:
        return 0


# ---------------------------------------------------------------------------------------------------------------------
# %% Reader


class ImageSequenceReader:
    """
    Random-access reader for a TIFF stack or a folder of stills.

    Frames are decoded on demand. get_fps() is 0 because stills have no timestamp;
    the export dialog asks for the frame rate used in the metrics.
    """

    def __init__(self, path: str, intensity_range: str | tuple[float, float] = "auto"):
        self._video_path = path
        self._is_reversed = False
        self._is_paused = False
        self._frame_idx = 0
        self._frame_buffer: OrderedDict[int, ndarray] = OrderedDict()
        self._frame_buffer_max = 0
        self._tiff_encoded: ndarray | None = None
        self._intensity_range = intensity_range
        self._uint16_lut: ndarray | None = None
        self.source_info: dict | None = None

        if osp.isdir(path):
            self._paths = list_image_files(path)
            self._tiff_path = None
            if len(self._paths) == 0:
                raise IOError(f"No still images found in folder: {path}")
        else:
            suffix = osp.splitext(path)[1].lower()
            if suffix in STACK_EXTENSIONS:
                count = tiff_page_count(path)
                if count <= 0:
                    raise IOError(f"Can't read TIFF pages from: {path}")
                self._paths = None
                self._tiff_path = path
                self._page_count = count
            else:
                image = imread_unicode(path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise IOError(f"Can't read image: {path}")
                self._paths = [path]
                self._tiff_path = None

        self.total_frames = self._page_count if self._tiff_path else len(self._paths)
        self._max_frame_idx = self.total_frames - 1
        self._fps = 0.0
        self._prepare_intensity()

        first = self._read_frame_at(0)
        if first is None:
            raise IOError(f"Can't read frames from image sequence: {path}")
        self.sample_frame = first
        self._pause_frame = first
        self.shape = first.shape

    # .................................................................................................................

    def set_frame_buffer_size(self, num_frames: int):
        self._frame_buffer_max = max(0, int(num_frames))
        while len(self._frame_buffer) > self._frame_buffer_max:
            self._frame_buffer.popitem(last=False)
        return self

    def release(self):
        """Image sequences keep their path list; nothing has to be closed."""
        return self

    def pause(self, set_is_paused=True) -> bool:
        self._is_paused = set_is_paused
        return self._is_paused

    def toggle_pause(self) -> bool:
        self._is_paused = not self._is_paused
        return self._is_paused

    def get_pause_state(self) -> bool:
        return self._is_paused

    def get_fps(self) -> float:
        return self._fps

    def get_sample_frame(self) -> ndarray:
        return self.sample_frame.copy()

    def get_frame_index(self) -> int:
        return self._frame_idx

    def get_current_frame(self) -> ndarray:
        return self._pause_frame.copy()

    def get_playback_position(self, normalized=True) -> int | float:
        if normalized:
            if self._max_frame_idx <= 0:
                return 0.0
            return self._frame_idx / self._max_frame_idx
        return int(self._frame_idx)

    def get_reverse_state(self) -> bool:
        return self._is_reversed

    def toggle_reverse_state(self, set_is_reversed: bool | None = None) -> bool:
        self._is_reversed = (not self._is_reversed) if set_is_reversed is None else bool(set_is_reversed)
        return self._is_reversed

    # .................................................................................................................

    def _store_buffer(self, frame_idx: int, frame: ndarray) -> None:
        if self._frame_buffer_max <= 0:
            return
        self._frame_buffer[frame_idx] = frame
        self._frame_buffer.move_to_end(frame_idx)
        while len(self._frame_buffer) > self._frame_buffer_max:
            self._frame_buffer.popitem(last=False)

    def _read_frame_at(self, frame_idx: int) -> ndarray | None:
        if frame_idx < 0 or frame_idx > self._max_frame_idx:
            return None
        cached = self._frame_buffer.get(frame_idx)
        if cached is not None:
            self._frame_buffer.move_to_end(frame_idx)
            return cached

        if self._tiff_path is not None:
            raw, encoded = read_tiff_page(self._tiff_path, frame_idx, self._tiff_encoded)
            if encoded is not None:
                self._tiff_encoded = encoded
        else:
            raw = imread_unicode(self._paths[frame_idx], cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None

        frame = to_bgr_uint8(raw, self._uint16_lut)
        self._store_buffer(frame_idx, frame)
        return frame

    def read_frame(self, frame_idx: int) -> ndarray | None:
        """Return one display frame, or None when it cannot be decoded."""

        frame = self._read_frame_at(int(frame_idx))
        return None if frame is None else frame.copy()

    def _prepare_intensity(self) -> None:
        """Build one uint16 lookup table for the whole sequence."""

        count = max(self.total_frames, 1)
        sample_count = min(16, count)
        if sample_count == 1:
            indices = [0]
        else:
            indices = [int(round(i * (count - 1) / (sample_count - 1))) for i in range(sample_count)]
        samples = []
        for index in indices:
            raw = self._raw_frame(index)
            if raw is not None and raw.dtype == np.uint16:
                samples.append(raw)
        if len(samples) == 0:
            return

        mode = self._intensity_range
        if isinstance(mode, tuple):
            low, high = float(mode[0]), float(mode[1])
            mode_name = "manual"
        elif mode == "full":
            low, high = 0.0, 65535.0
            mode_name = "full"
        else:
            flat = [sample.reshape(-1) for sample in samples]
            merged = np.concatenate([values[:: max(1, values.size // 200_000)] for values in flat])
            low, high = (float(value) for value in np.percentile(merged, (0.1, 99.9)))
            mode_name = "auto"
        if high <= low:
            high = low + 1.0
        self._uint16_lut = make_uint16_lut(low, high)
        self.source_info = {
            "intensity_mode": mode_name,
            "uint16_low": low,
            "uint16_high": high,
        }

    def _raw_frame(self, frame_idx: int) -> ndarray | None:
        if self._tiff_path is not None:
            raw, encoded = read_tiff_page(self._tiff_path, frame_idx, self._tiff_encoded)
            if encoded is not None:
                self._tiff_encoded = encoded
            return raw
        if self._paths is None or not (0 <= frame_idx < len(self._paths)):
            return None
        return imread_unicode(self._paths[frame_idx], cv2.IMREAD_UNCHANGED)

    def set_playback_position(self, position: int | float, is_normalized=False) -> int:
        frame_idx = round(position * self._max_frame_idx) if is_normalized else int(position)
        frame_idx = max(0, min(frame_idx, self._max_frame_idx))
        self._frame_idx = frame_idx
        if self._is_paused:
            frame = self._read_frame_at(frame_idx)
            if frame is not None:
                self._pause_frame = frame
        return frame_idx

    def step_frame_by(self, delta: int) -> int:
        if delta == 0:
            return self._frame_idx
        new_idx = max(0, min(self._max_frame_idx, self._frame_idx + int(delta)))
        if new_idx == self._frame_idx:
            return self._frame_idx
        self.pause(True)
        self.set_playback_position(new_idx)
        return new_idx

    # .................................................................................................................

    def __iter__(self):
        return self

    def __next__(self) -> tuple[bool, int, ndarray]:
        if self._is_paused:
            return self._is_paused, self._frame_idx, self._pause_frame.copy()

        step = -1 if self._is_reversed else 1
        target_idx = self._frame_idx + step
        if target_idx < 0 or target_idx > self._max_frame_idx:
            edge = 0 if self._is_reversed else self._max_frame_idx
            self._is_paused = True
            self._frame_idx = edge
            frame = self._read_frame_at(edge)
            if frame is not None:
                self._pause_frame = frame
            return self._is_paused, self._frame_idx, self._pause_frame.copy()

        frame = self._read_frame_at(target_idx)
        if frame is None:
            self._is_paused = True
            return self._is_paused, self._frame_idx, self._pause_frame.copy()

        self._frame_idx = target_idx
        self._pause_frame = frame
        return self._is_paused, self._frame_idx, frame.copy()
