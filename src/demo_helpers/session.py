#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Save and reload the raw prompts for a tracking session. Author: Lucien, 2026-09-23."""


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

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


def session_from_manager(obj_mgr, model_path=None, encode_side=None, use_square_sizing=None) -> dict:
    """Collect stored prompt specs. Encodings are not saved; they are rebuilt on load.

    model_path, encode_side, and use_square_sizing are the settings that produced
    those prompts. Load applies them before encoding again.
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
        "objects": objects,
    }


def write_session(path: str, session: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(session, handle, indent=2)


def read_session(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        session = json.load(handle)
    if not isinstance(session, dict) or not isinstance(session.get("objects"), list):
        raise ValueError("Session file does not contain an object list.")
    return session
