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


def open_frame_source(path: str):
    """
    Open a video file, a TIFF stack, a still image, or a folder of stills.

    Video containers stay on the OpenCV video reader. Image paths use ImageSequenceReader.
    """

    from .ui.video import ReversibleLoopingVideoReader

    if osp.isdir(path) or osp.splitext(path)[1].lower() in IMAGE_EXTENSIONS:
        return ImageSequenceReader(path)
    return ReversibleLoopingVideoReader(path)


def to_bgr_uint8(image: ndarray) -> ndarray:
    """Convert a decoded still to 3-channel uint8 BGR for the display and the model."""

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.dtype == np.uint16:
        image = (image >> 8).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.ascontiguousarray(image)


def read_tiff_page(path: str, index: int) -> ndarray | None:
    """Read one page from a multipage TIFF. Returns None when the page does not exist."""

    ok, pages = cv2.imreadmulti(path, start=int(index), count=1, flags=cv2.IMREAD_UNCHANGED)
    if not ok or pages is None or len(pages) == 0:
        return None
    page = pages[0]
    if page is None or page.size == 0:
        return None
    return page


def tiff_page_count(path: str) -> int:
    """Count TIFF pages with a logarithmic probe so the whole stack is not decoded up front."""

    if read_tiff_page(path, 0) is None:
        return 0

    upper = 1
    while read_tiff_page(path, upper) is not None:
        upper *= 2
        if upper > 1_000_000:
            break

    lower = upper // 2
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if read_tiff_page(path, middle) is not None:
            lower = middle
        else:
            upper = middle
    return lower + 1


# ---------------------------------------------------------------------------------------------------------------------
# %% Reader


class ImageSequenceReader:
    """
    Random-access reader for a TIFF stack or a folder of stills.

    Frames are decoded on demand. get_fps() is 0 because stills have no timestamp;
    the export dialog asks for the frame rate used in the metrics.
    """

    def __init__(self, path: str):
        self._video_path = path
        self._is_reversed = False
        self._is_paused = False
        self._frame_idx = 0
        self._frame_buffer: OrderedDict[int, ndarray] = OrderedDict()
        self._frame_buffer_max = 0

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
                image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise IOError(f"Can't read image: {path}")
                self._paths = [path]
                self._tiff_path = None

        self.total_frames = self._page_count if self._tiff_path else len(self._paths)
        self._max_frame_idx = self.total_frames - 1
        self._fps = 0.0

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
            raw = read_tiff_page(self._tiff_path, frame_idx)
        else:
            raw = cv2.imread(self._paths[frame_idx], cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None

        frame = to_bgr_uint8(raw)
        self._store_buffer(frame_idx, frame)
        return frame

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
