#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

# For type hints
from numpy import ndarray
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def resolve_overlapping_masks(
    labeled_masks: list[tuple[int, ndarray, float]],
) -> dict[int, ndarray]:
    """
    Give each foreground pixel to the object with the highest score.

    labeled_masks entries are (stable_id, uint8 mask, score). Masks must share
    one height and width. Equal scores keep the smaller stable id.

    Returns a new uint8 mask per stable id. Masks in the result do not overlap.
    """

    resolved: dict[int, ndarray] = {}
    if len(labeled_masks) == 0:
        return resolved

    height, width = labeled_masks[0][1].shape[:2]
    winner_score = np.full((height, width), -np.inf, dtype=np.float64)
    winner_id = np.zeros((height, width), dtype=np.int32)

    for stable_id, mask, score in labeled_masks:
        active = mask > 0
        better_score = active & (score > winner_score)
        empty_or_smaller_id = (winner_id == 0) | (stable_id < winner_id)
        tie = active & (score == winner_score) & empty_or_smaller_id
        take = better_score | tie
        winner_score[take] = score
        winner_id[take] = int(stable_id)

    for stable_id, mask, _score in labeled_masks:
        resolved[int(stable_id)] = np.where(winner_id == int(stable_id), 255, 0).astype(np.uint8)
        del mask

    return resolved


def suppress_lost_pixels(mask_predictions: Tensor, mask_index: int, removed_hires: ndarray) -> Tensor:
    """
    Return a clone of mask_predictions whose selected channel is driven below
    the display threshold wherever removed_hires is set.

    The original tensor is left unchanged so the on-screen mask can keep using
    the exclusive high-resolution result.
    """

    if removed_hires is None or not np.any(removed_hires):
        return mask_predictions

    low_h, low_w = mask_predictions.shape[-2:]
    removed_low = cv2.resize(removed_hires.astype(np.uint8), (low_w, low_h), interpolation=cv2.INTER_NEAREST)
    if not np.any(removed_low):
        return mask_predictions

    import torch

    preds = mask_predictions.clone()
    selected = preds[:, int(mask_index)]
    drop = torch.as_tensor(removed_low > 0, device=selected.device)
    selected[:, drop] = -1.0e4
    return preds
