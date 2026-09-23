#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Save and reload the raw prompts for a tracking session. Author: Lucien, 2026-09-23."""


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os.path as osp
import json


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def _plain(value):
    """Turn prompt coordinates into JSON-safe lists of floats."""

    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (int, float)):
        return float(value)
    return value


def session_from_manager(
    obj_mgr,
    model_path=None,
    encode_side=None,
    use_square_sizing=None,
    video_path=None,
) -> dict:
    """Collect stored prompt specs. Encodings are not saved; they are rebuilt on load.

    model_path, encode_side, use_square_sizing, and video_path are the settings
    that produced those prompts. Load applies the encode settings before encoding
    again, and refuses a different video.
    """

    objects = []
    for objidx, stable_id in enumerate(obj_mgr.stable_ids):
        memory = obj_mgr.memory_list[objidx]
        prompts = []
        for frame_idx, spec in zip(memory.prompt_frame_indices, memory.prompt_specs):
            if not spec:
                continue
            prompts.append(
                {
                    "frame_index": int(frame_idx),
                    "boxes": _plain(spec.get("boxes", [])),
                    "fg_points": _plain(spec.get("fg_points", [])),
                    "bg_points": _plain(spec.get("bg_points", [])),
                }
            )
        if prompts:
            objects.append({"stable_id": int(stable_id), "prompts": prompts})
    return {
        "version": 2,
        "model_path": None if model_path is None else str(model_path),
        "encode_side": None if encode_side is None else int(encode_side),
        "use_square_sizing": None if use_square_sizing is None else bool(use_square_sizing),
        "video_path": None if not video_path else osp.abspath(str(video_path)),
        "objects": objects,
    }


def session_object_ids(objects: list) -> list[int]:
    """Stable label ids in file order. Missing ids take the smallest free number."""

    ids: list[int] = []
    used: set[int] = set()
    for obj in objects:
        raw = obj.get("stable_id") if isinstance(obj, dict) else None
        if raw is None:
            stable_id = 1
            while stable_id in used:
                stable_id += 1
        else:
            stable_id = int(raw)
        if not 1 <= stable_id <= 255:
            raise ValueError(f"Session object id {stable_id} is outside 1..255.")
        if stable_id in used:
            raise ValueError(f"Session object id {stable_id} is repeated.")
        used.add(stable_id)
        ids.append(stable_id)
    return ids


def session_video_mismatch(session: dict, current_video) -> str | None:
    """A message when this session belongs to a different video. None when load may continue."""

    saved = session.get("video_path")
    if not saved:
        return None
    if not current_video:
        return "Open the video from this session before loading it."
    saved_key = osp.normcase(osp.abspath(str(saved)))
    current_key = osp.normcase(osp.abspath(str(current_video)))
    if saved_key == current_key:
        return None
    return f"This session was saved for a different video:\n{saved}"


def write_session(path: str, session: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(session, handle, indent=2)


def read_session(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        session = json.load(handle)
    if not isinstance(session, dict) or not isinstance(session.get("objects"), list):
        raise ValueError("Session file does not contain an object list.")
    return session
