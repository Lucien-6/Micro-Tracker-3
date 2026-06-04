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

    def __init__(self, max_items: int = 64, store_device: str = "cpu"):
        self._cache: "OrderedDict[int, object]" = OrderedDict()
        self._max_items = max(0, int(max_items))
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
        self._cache[key] = _move_tensor_tree(encoded_on_device, self._store_device)
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_items:
            self._cache.popitem(last=False)

    # .................................................................................................................

    def clear(self) -> None:
        self._cache.clear()

    # .................................................................................................................

    def __len__(self) -> int:
        return len(self._cache)

    # .................................................................................................................
