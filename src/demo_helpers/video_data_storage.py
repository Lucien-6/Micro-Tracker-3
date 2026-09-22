#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from collections import deque

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Data types


class SAMVideoMemoryBank:
    """
    Simpler helper used to store prompt & per-frame memory encodings needed for video segmentation.
    It's just a wrapper around two fixed-length lists (deques)
    """

    def __init__(self, max_frame_memory: int = 6, max_prompt_memory: int = 32):
        self.frame_memory = deque([], maxlen=max_frame_memory)
        self.prompt_memory = deque([], maxlen=max_prompt_memory)
        # First frame index where object score fell below threshold; inference is skipped while
        # frame_idx >= this value until reconcile clears it or Store Prompt adds new prompts.
        self.tracking_stop_frame_idx: int | None = None
        # Playhead position of the last inference. Frame memory is kept only while the
        # playhead steps by exactly one frame in the same direction.
        self.last_tracked_frame_idx: int | None = None
        self.track_direction: int | None = None

    def store_prompt_result(self, memory_encoding: list[Tensor]):
        """Used to store prompt memory encodings"""
        self.prompt_memory.append(memory_encoding)
        return self

    def store_frame_result(self, memory_encoding: list[Tensor]):
        """Used to store per-frame memory encodings"""
        self.frame_memory.append(memory_encoding)
        return self

    def discard_frame_memory(self):
        """Drop frame memory and the playhead chain. Prompt memory is kept."""
        self.frame_memory.clear()
        self.last_tracked_frame_idx = None
        self.track_direction = None
        return self

    def begin_frame(self, frame_idx: int, direction: int) -> str:
        """
        Prepare frame memory for an inference at frame_idx.

        direction is +1 when time moves forward and -1 when it moves backward.

        Returns:
            "same" — this frame was already inferred; keep memory and do not append another copy
            "continue" — the frame is exactly one step in the current direction
            "start" — no chain yet, or the playhead jumped / changed direction (frame memory cleared)
        """

        frame_idx = int(frame_idx)
        direction = 1 if int(direction) >= 0 else -1
        if self.last_tracked_frame_idx is None:
            return "start"
        if frame_idx == self.last_tracked_frame_idx:
            return "same"
        if self.track_direction == direction and frame_idx == self.last_tracked_frame_idx + direction:
            return "continue"
        self.discard_frame_memory()
        return "start"

    def note_playhead(self, frame_idx: int, direction: int):
        """Remember a completed inference so the next adjacent frame can continue the chain."""
        self.last_tracked_frame_idx = int(frame_idx)
        self.track_direction = 1 if int(direction) >= 0 else -1
        return self

    def to_dict(self) -> dict:
        """Helper used to convert stored data into a dictionary, so it can be used as a 'kwargs' argument"""
        return {
            "prompt_memory_encodings": self.prompt_memory,
            "frame_memory_encodings": self.frame_memory,
        }

    def get_num_memories(self) -> tuple[int, int]:
        """Read the length of currently stored memory data. Returns: num_prompt_memory, num_frame_memory"""
        num_prompt_mems = len(self.prompt_memory)
        num_frame_mems = len(self.frame_memory)
        return num_prompt_mems, num_frame_mems

    def clear(self, clear_prompt_memory: bool = True, clear_frame_memory: bool = True):
        if clear_frame_memory:
            self.discard_frame_memory()
        if clear_prompt_memory:
            self.prompt_memory.clear()
        self.tracking_stop_frame_idx = None
        return self

    def clear_tracking_stop_frame(self):
        self.tracking_stop_frame_idx = None
        return self

    def reconcile_tracking_stop_frame(self, frame_idx: int) -> None:
        """Clear tracking_stop_frame_idx when the playhead is before the first loss frame index."""
        if self.tracking_stop_frame_idx is not None and frame_idx < self.tracking_stop_frame_idx:
            self.tracking_stop_frame_idx = None

    def set_tracking_stop_frame(self, frame_idx: int) -> None:
        """Remember the first loss frame (only the first assignment is kept until reconcile or clear)."""
        if self.tracking_stop_frame_idx is None:
            self.tracking_stop_frame_idx = int(frame_idx)

    def should_run_video_masking(self, frame_idx: int, keep_tracking_after_loss: bool) -> bool:
        """Return whether to call step_video_masking on this frame index."""
        if keep_tracking_after_loss:
            return True
        if self.tracking_stop_frame_idx is None:
            return True
        # Stop is set after one inference on the loss frame; skip when revisiting that frame or any later frame.
        return frame_idx < self.tracking_stop_frame_idx

    def check_has_prompts(self) -> bool:
        """Helper used to check if there is any stored prompt memory"""
        return len(self.prompt_memory) > 0
