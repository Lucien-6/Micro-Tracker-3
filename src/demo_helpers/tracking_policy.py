#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Mask selection and prompt-memory limits. Author: Lucien, 2026-09-23."""


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from collections.abc import Sequence


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def select_mask_index(
    iou_predictions,
    family: str,
    stage: str,
    mode: str = "legacy",
    num_points: int = 0,
    has_box: bool = False,
) -> int:
    """
    Choose a mask channel.

    legacy keeps the historical argmax over every channel.
    official follows SAM 2/3 tracking: tokens 1-3 for tracking and for a single
    point, token 0 for a box or for two or more points. SAM 3.1 tracking skips
    only the padded fourth channel.
    """

    ious = iou_predictions.reshape(-1)
    if mode != "official":
        return int(ious.argmax().item())
    if family == "v3p1" and stage == "track":
        return int(ious[: min(3, ious.numel())].argmax().item())
    use_multimask = stage == "track" or (num_points <= 1 and not has_box)
    if use_multimask and ious.numel() > 1:
        return 1 + int(ious[1:].argmax().item())
    return 0


def select_prompt_encodings(frame_indices: Sequence[int], encodings: Sequence, frame_idx: int, max_count: int | None):
    """
    Keep every prompt, or the first prompt plus the prompts closest in time.

    The result stays in the original insertion order. max_count None keeps all.
    """

    items = list(zip(frame_indices, encodings))
    if max_count is None or max_count <= 0 or len(items) <= max_count:
        return [encoding for _, encoding in items]
    keep = max(1, int(max_count))
    ranked = sorted(range(1, len(items)), key=lambda index: (abs(int(items[index][0]) - int(frame_idx)), index))
    chosen = [0, *sorted(ranked[: keep - 1])]
    return [items[index][1] for index in chosen]


def model_family_name(sam_core) -> str:
    """Return v2, v3, or v3p1 from the loaded core class name."""

    name = type(sam_core).__name__.lower()
    if "v3p1" in name or "3p1" in name:
        return "v3p1"
    if "v3" in name:
        return "v3"
    return "v2"
