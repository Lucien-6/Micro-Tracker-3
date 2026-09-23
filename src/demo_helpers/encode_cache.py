#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from collections import OrderedDict

import torch

# Typing
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def tensor_tree_nbytes(obj) -> int:
    """Byte size of every tensor in a nested encoding. None leaves count as zero."""

    if isinstance(obj, Tensor):
        return int(obj.numel()) * int(obj.element_size())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_tree_nbytes(item) for item in obj)
    if isinstance(obj, dict):
        return sum(tensor_tree_nbytes(value) for value in obj.values())
    return 0


def strip_detector_branch(encoded, family: str):
    """
    Drop the SAM 3 / 3.1 detector features.

    Tracking and interactive segmentation read the earlier branches only.
    Keeping the detector branch made each cached frame several times larger.
    """

    drop = {"v3": 1, "v3p1": 2}.get(family)
    if drop is None or not isinstance(encoded, tuple) or drop >= len(encoded):
        return encoded
    return tuple(None if index == drop else part for index, part in enumerate(encoded))


def _move_tensor_tree(obj, device):
    """
    Recursively move every tensor inside an arbitrarily nested list/tuple/dict
    structure to the given device. Non-tensor leaves are returned unchanged.

    This is needed because the SAM image-encoding outputs differ by model
    version (a flat list of tensors for v2, nested tuples of lists for v3/v3.1).
    """

    if isinstance(obj, Tensor):
        return obj.detach().to(device)
    if isinstance(obj, list):
        return [_move_tensor_tree(item, device) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tensor_tree(item, device) for item in obj)
    if isinstance(obj, dict):
        return {key: _move_tensor_tree(val, device) for key, val in obj.items()}
    return obj


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class EncodedImageCache:
    """
    LRU cache for model image-encodings, keyed by frame index.

    Encoding a frame runs the (expensive) SAM image encoder. When scrubbing,
    stepping or playing in reverse, the same frames are revisited repeatedly,
    so caching their encodings avoids recomputation. To conserve GPU memory the
    cached encodings are stored on CPU by default and moved back to the compute
    device on a cache hit (a comparatively cheap host/device transfer).

    The cache is only valid for a fixed (video, model, encoder-config) triplet;
    call clear() whenever any of those change.
    """

    # .................................................................................................................

    def __init__(self, max_items: int = 64, store_device: str = "cpu", max_bytes: int | None = None):
        self._cache: "OrderedDict[int, object]" = OrderedDict()
        self._sizes: dict[int, int] = {}
        self._nbytes = 0
        self._max_items = max(0, int(max_items))
        self._max_bytes = None if max_bytes is None else max(0, int(max_bytes))
        self._store_device = torch.device(store_device)

    # .................................................................................................................

    def is_enabled(self) -> bool:
        return self._max_items > 0

    # .................................................................................................................

    def get(self, key: int, target_device):
        """Return a device-resident copy of the cached encoding, or None on a miss."""
        if self._max_items == 0 or key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return _move_tensor_tree(self._cache[key], target_device)

    # .................................................................................................................

    def store(self, key: int, encoded_on_device) -> None:
        """Store an encoding (copied to the cache device) and evict the oldest entry if needed."""
        if self._max_items == 0:
            return
        if key in self._cache:
            self._nbytes -= self._sizes.pop(key, 0)
        stored = _move_tensor_tree(encoded_on_device, self._store_device)
        self._cache[key] = stored
        self._sizes[key] = tensor_tree_nbytes(stored)
        self._nbytes += self._sizes[key]
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_items or (self._max_bytes and self._nbytes > self._max_bytes and len(self._cache) > 1):
            old_key, _old_value = self._cache.popitem(last=False)
            self._nbytes -= self._sizes.pop(old_key, 0)

    # .................................................................................................................

    def clear(self) -> None:
        self._cache.clear()
        self._sizes.clear()
        self._nbytes = 0

    # .................................................................................................................

    def __len__(self) -> int:
        return len(self._cache)

    # .................................................................................................................
