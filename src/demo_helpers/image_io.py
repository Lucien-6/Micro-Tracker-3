#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Image read/write that works for non-ASCII paths on Windows. Author: Lucien, 2026-09-23."""


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os.path as osp

import cv2
import numpy as np

# For type hints
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def imwrite_unicode(path: str, image: ndarray) -> bool:
    """Write an image. cv2.imwrite fails for non-ASCII paths on Windows."""

    extension = osp.splitext(path)[1] or ".tif"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        return False
    encoded.tofile(path)
    return True


def imread_unicode(path: str, flags: int = cv2.IMREAD_UNCHANGED) -> ndarray | None:
    """Read an image. cv2.imread fails for non-ASCII paths on Windows."""

    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    image = cv2.imdecode(encoded, flags)
    if image is None or image.size == 0:
        return None
    return image
