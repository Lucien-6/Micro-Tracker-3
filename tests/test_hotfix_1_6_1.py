#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Regression checks for the 1.6.1 hotfix. Author: Lucien, 2026-09-23."""

import os.path as osp

import numpy as np
import torch

from src.demo_helpers.history_keeper import HistoryKeeper
from src.demo_helpers.image_io import imwrite_unicode
from src.demo_helpers.image_sequence import ImageSequenceReader
from src.demo_helpers.loading import clean_path_str, resolve_model_candidates
from src.demo_helpers.saving import load_saved_label_frames, save_tracking_label_tif_sequence
from src.v2_sam.components.memory_image_fusion_attention import RPEComplexEncoder
from src.v2_sam.components.posenc_sine import SinusoidalPE2D as SinusoidalV2
from src.v3_sam.components.position_encoding import RPEComplex
from src.v3_sam.components.position_encoding import SinusoidalPE2D as SinusoidalV3
from src.v3p1_sam.components.position_encoding import SinusoidalPE2D as SinusoidalV31


def test_clean_path_keeps_an_apostrophe_inside_the_name(tmp_path):
    folder = tmp_path / "Lucien's data"
    folder.mkdir()
    clip = folder / "clip.avi"
    clip.write_bytes(b"")
    quoted = f'"{clip}"'
    assert osp.exists(clean_path_str(quoted))
    assert osp.exists(clean_path_str(str(clip)))


def test_explicit_model_selector_does_not_fall_back_to_history(tmp_path):
    weights = tmp_path / "model_weights"
    weights.mkdir()
    for name in ("sam2.1_hiera_large.pt", "sam2.1_hiera_tiny.pt", "sam3.pt"):
        (weights / name).write_bytes(b"")
    script = tmp_path / "main.py"
    script.write_text("", encoding="utf-8")
    history = str(weights / "sam3.pt")

    assert osp.basename(resolve_model_candidates(str(script), "large", history)[0]) == "sam2.1_hiera_large.pt"
    no_flag = resolve_model_candidates(str(script), None, history)
    assert osp.basename(no_flag[0]) == "sam3.pt"
    assert "sam2.1_hiera_tiny.pt" in [osp.basename(path) for path in no_flag]

    try:
        resolve_model_candidates(str(script), "missing-weights", history)
    except FileNotFoundError as err:
        assert "missing-weights" in str(err)
        return
    raise AssertionError("an explicit selector that matches nothing must not fall back")


def test_corrupt_history_is_reset_and_rewritten_atomically(tmp_path):
    script = tmp_path / "main.py"
    script.write_text("", encoding="utf-8")
    (tmp_path / ".history").write_text('{"model_path": ', encoding="utf-8")

    keeper = HistoryKeeper(str(script))
    assert keeper.read("model_path") == (False, None)
    backups = [path.name for path in tmp_path.iterdir() if path.name.startswith(".history.corrupt-")]
    assert len(backups) == 1

    keeper.store(model_path="G:/weights/sam.pt")
    assert not (tmp_path / ".history.tmp").exists()
    reloaded = HistoryKeeper(str(script))
    assert reloaded.read("model_path") == (True, "G:/weights/sam.pt")


def test_position_encoding_cache_accepts_a_width_change():
    for encoder in (SinusoidalV2(64), SinusoidalV3(64), SinusoidalV31(64)):
        wide = encoder(4, 6)
        taller_width = encoder(4, 8)
        assert tuple(wide.shape[-2:]) == (4, 6)
        assert tuple(taller_width.shape[-2:]) == (4, 8)


def test_sam2_rope_cache_rebuilds_for_a_transposed_grid():
    encoder = RPEComplexEncoder(64)
    tokens = torch.zeros(1, 1, 8, 64)
    encoder((2, 4), tokens, tokens)
    encoder((4, 2), tokens, tokens)
    assert encoder._rotvectors_hw == (4, 2)


def test_sam3_rope_axes_follow_token_width_and_height():
    encoder = RPEComplex(64, rope_hw=(24, 24))
    rotors = encoder.get_complex_rotors((72, 96))
    # Column 1 is token 1. Its x angle is 24/width when the axes are not swapped.
    # Angles wrap past pi, so the check uses this small index instead of the maximum.
    x_at_column_1 = float(torch.angle(rotors[0, 0, 1, 0]))
    assert abs(x_at_column_1 - (24 / 96)) < 1e-4


def test_label_sequence_round_trip_on_a_non_ascii_path(tmp_path):
    folder = tmp_path / "\u5b9e\u9a8c\u6570\u636e"
    frames = {240: np.array([[0, 4], [0, 0]], dtype=np.uint8)}
    save_tracking_label_tif_sequence(frames, str(folder), fps=10.0)
    loaded = load_saved_label_frames(str(folder))
    assert int(loaded[240][0, 1]) == 4


def test_image_folder_with_a_non_ascii_name(tmp_path):
    folder = tmp_path / "\u5e27"
    folder.mkdir()
    image = np.full((8, 6, 3), 40, dtype=np.uint8)
    assert imwrite_unicode(str(folder / "frame_01.png"), image)
    reader = ImageSequenceReader(str(folder))
    assert reader.total_frames == 1
    assert reader.get_sample_frame().shape == (8, 6, 3)


def test_shared_tk_root_keeps_dialog_variables():
    import tkinter as tk

    from src.demo_helpers.tk_host import close_tk_root, get_tk_root
    from src.demo_helpers.ui.user_guide_window import UserGuideWindow

    guide = UserGuideWindow("Micro Tracker 3", "1.6.1", "Lucien", "lucien-6@qq.com")
    try:
        guide.show()
        dialog = tk.Toplevel(get_tk_root())
        variable = tk.StringVar(master=dialog, value="30")
        entry = tk.Entry(dialog, textvariable=variable)
        entry.pack()
        dialog.update()
        assert entry.get() == "30"
        entry.delete(0, tk.END)
        entry.insert(0, "12.5")
        dialog.update()
        assert variable.get() == "12.5"
        dialog.destroy()
    finally:
        guide.close()
        close_tk_root()
