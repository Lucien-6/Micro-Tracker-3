#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Checks for the 1.7.0 tracking and export changes. Author: Lucien, 2026-09-23."""

import cv2
import numpy as np
import tifffile
import torch

import main as app
from src.demo_helpers.analysis import build_metric_rows, export_tracking_analysis
from src.demo_helpers.encode_cache import EncodedImageCache, strip_detector_branch
from src.demo_helpers.image_io import imwrite_unicode
from src.demo_helpers.image_sequence import ImageSequenceReader
from src.demo_helpers.tracking_policy import select_mask_index, select_prompt_encodings
from src.demo_helpers.video_data_storage import SAMVideoMemoryBank


def test_official_mask_selection_skips_the_untrained_channel():
    ious = torch.tensor([0.99, 0.4, 0.2, 0.1])
    assert select_mask_index(ious, "v2", "track", "legacy") == 0
    assert select_mask_index(ious, "v3", "track", "official") == 1
    assert select_mask_index(ious, "v2", "prompt", "official", num_points=1, has_box=False) == 1
    assert select_mask_index(ious, "v2", "prompt", "official", num_points=2, has_box=True) == 0
    assert select_mask_index(torch.tensor([0.1, 0.2, 0.05, 0.9]), "v3p1", "track", "official") == 1


def test_prompt_attention_keeps_the_first_and_the_closest():
    selected = select_prompt_encodings([0, 10, 11, 50], ["a", "b", "c", "d"], 12, 2)
    assert selected == ["a", "c"]
    assert select_prompt_encodings([0, 1], ["a", "b"], 1, None) == ["a", "b"]


def test_lost_patience_waits_for_consecutive_low_scores():
    bank = SAMVideoMemoryBank()
    assert bank.note_score(5, True, 3) is False
    assert bank.note_score(6, True, 3) is False
    assert bank.note_score(7, True, 3) is True
    assert bank.tracking_stop_frame_idx == 5
    assert bank.note_score(8, False, 3) is False
    assert bank.low_score_streak == 0


def test_velocity_is_blank_on_the_first_sample_and_across_a_gap():
    stats = {
        "centroid_x": 10.0,
        "centroid_y": 4.0,
        "area_px": 20.0,
        "orientation_deg": 0.0,
        "num_components": 1,
    }
    moved = dict(stats, centroid_x=12.0)
    rows = build_metric_rows({1: [(0, stats), (1, moved), (6, moved)]}, 1.0, 10.0, max_frame_gap=1)
    assert rows[0]["velocity_x_um_per_s"] == ""
    assert rows[1]["frame_gap"] == 1
    assert abs(rows[1]["speed_um_per_s"] - 20.0) < 1e-6
    assert rows[2]["velocity_x_um_per_s"] == ""
    assert "gap" in rows[2]["qc_flag"]


def test_uint16_auto_range_uses_the_data_instead_of_a_bit_shift(tmp_path):
    path = tmp_path / "stack.tif"
    dark = np.full((8, 8), 100, dtype=np.uint16)
    bright = np.full((8, 8), 4000, dtype=np.uint16)
    tifffile.imwrite(path, np.stack([dark, bright]), photometric="minisblack")
    reader = ImageSequenceReader(str(path), intensity_range="auto")
    reader.pause(True)
    reader.set_playback_position(1)
    bright_u8 = int(reader.get_current_frame()[0, 0, 0])
    dark_u8 = int(reader.read_frame(0)[0, 0, 0])
    assert bright_u8 > 200
    assert dark_u8 < 40
    assert reader.source_info["intensity_mode"] == "auto"


def test_overlay_uses_image_folder_frames(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(3):
        image = np.full((12, 16, 3), 180, dtype=np.uint8)
        assert imwrite_unicode(str(frames / f"f{index}.png"), image)
    labels = {index: np.full((12, 16), 1, dtype=np.uint8) for index in range(3)}
    out = tmp_path / "out"
    summary = export_tracking_analysis(str(out), labels, str(frames), 10.0, 1.0)
    capture = cv2.VideoCapture(summary["overlay_path"])
    ok, frame = capture.read()
    capture.release()
    assert ok
    assert float(frame.mean()) > 100
    assert summary["overlay_missed_frames"] == 0


def test_detector_branch_is_stripped_and_the_cache_obeys_a_byte_budget():
    encoded = (torch.zeros(2), torch.ones(4), torch.zeros(1))
    stripped = strip_detector_branch(encoded, "v3")
    assert stripped[1] is None and torch.equal(stripped[0], encoded[0])
    assert strip_detector_branch(encoded, "v3p1")[2] is None

    cache = EncodedImageCache(max_items=8, max_bytes=900)
    item = torch.zeros(100, dtype=torch.float32)  # 400 bytes
    cache.store(1, item)
    cache.store(2, item)
    cache.store(3, item)
    assert len(cache) == 2
    assert cache.get(1, "cpu") is None
    assert cache.get(3, "cpu") is not None


def test_recording_waits_for_a_tracked_mask_and_does_not_repeat_it(monkeypatch):
    class Memory:
        def check_has_prompts(self):
            return True

    class Result:
        def __init__(self):
            self.source = "preview"
            self.frame_idx = 4
            self.version = 1

    class Manager:
        def __init__(self):
            self.objiter = [0]
            self.memory_list = [Memory()]
            self.maskresults_list = [Result()]
            self.last_record_signature = None

    recorded = []
    monkeypatch.setattr(app, "record_combined_tracking_frame", lambda *_args: recorded.append(_args[2]))
    manager = Manager()
    app.maybe_record_current_frame(manager, None, 4, (8, 8), True)
    assert recorded == []
    manager.maskresults_list[0].source = "track"
    app.maybe_record_current_frame(manager, None, 4, (8, 8), True)
    app.maybe_record_current_frame(manager, None, 4, (8, 8), True)
    assert recorded == [4]
