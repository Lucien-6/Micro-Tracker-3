#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Checks for the tracking, export, and image-sequence fixes. Author: Lucien."""

import os
import os.path as osp
import tempfile

import cv2
import numpy as np
import torch

from src.demo_helpers.exclusive_masks import resolve_overlapping_masks, suppress_lost_pixels
from src.demo_helpers.image_sequence import ImageSequenceReader, list_image_files, open_frame_source
from src.demo_helpers.loading import resolve_startup_settings
from src.demo_helpers.saving import (
    label_tif_filename,
    load_saved_label_frames,
    save_tracking_label_tif_sequence,
)
from src.demo_helpers.video_data_storage import SAMVideoMemoryBank


def test_frame_memory_keeps_only_adjacent_steps():
    bank = SAMVideoMemoryBank(max_frame_memory=6)
    bank.store_prompt_result(["prompt"])

    assert bank.begin_frame(0, 1) == "start"
    bank.note_playhead(0, 1)
    bank.store_frame_result("f0")

    assert bank.begin_frame(1, 1) == "continue"
    bank.note_playhead(1, 1)
    bank.store_frame_result("f1")
    assert list(bank.frame_memory) == ["f0", "f1"]

    assert bank.begin_frame(1, 1) == "same"
    assert list(bank.frame_memory) == ["f0", "f1"]

    assert bank.begin_frame(8, 1) == "start"
    assert len(bank.frame_memory) == 0
    assert bank.check_has_prompts()
    bank.note_playhead(8, 1)
    bank.store_frame_result("f8")

    assert bank.begin_frame(7, -1) == "start"
    assert len(bank.frame_memory) == 0
    assert bank.check_has_prompts()


def test_store_prompt_can_drop_frame_memory():
    bank = SAMVideoMemoryBank()
    bank.store_prompt_result(["prompt"])
    bank.store_frame_result("f0")
    bank.note_playhead(0, 1)
    bank.discard_frame_memory()
    assert len(bank.frame_memory) == 0
    assert bank.last_tracked_frame_idx is None
    assert bank.check_has_prompts()


def test_overlap_keeps_higher_score_and_smaller_id_on_ties():
    left = np.zeros((2, 2), dtype=np.uint8)
    right = np.zeros((2, 2), dtype=np.uint8)
    left[:, :] = 255
    right[:, :] = 255
    resolved = resolve_overlapping_masks([(1, left, 0.2), (2, right, 0.9)])
    assert resolved[2].sum() == 255 * 4
    assert resolved[1].sum() == 0

    tie = resolve_overlapping_masks([(3, left, 0.5), (1, right.copy(), 0.5)])
    assert tie[1].sum() == 255 * 4
    assert tie[3].sum() == 0


def test_suppress_lost_pixels_leaves_the_original_tensor():
    preds = torch.ones((1, 2, 2, 2))
    removed = np.zeros((4, 4), dtype=np.uint8)
    removed[0:2, 0:2] = 255
    updated = suppress_lost_pixels(preds, 0, removed)
    assert torch.equal(preds, torch.ones_like(preds))
    assert float(updated[0, 0, 0, 0]) < 0
    assert float(updated[0, 1, 0, 0]) > 0


def test_label_filenames_use_frame_index():
    frames = {
        240: np.array([[1, 0], [0, 2]], dtype=np.uint8),
        10: np.array([[0, 3], [0, 0]], dtype=np.uint8),
    }
    with tempfile.TemporaryDirectory() as folder:
        save_tracking_label_tif_sequence(frames, folder, fps=10.0)
        assert osp.isfile(osp.join(folder, "00010.tif"))
        assert osp.isfile(osp.join(folder, "00240.tif"))
        assert not osp.isfile(osp.join(folder, "00001.tif"))
        loaded = load_saved_label_frames(folder)
        assert set(loaded) == {10, 240}
        assert int(loaded[240][0, 0]) == 1
        text = open(osp.join(folder, "frame_index.csv"), encoding="utf-8").read()
        assert "00240.tif" in text
        assert "24.0" in text
    assert label_tif_filename(240) == "00240.tif"


def test_startup_settings_honor_explicit_defaults():
    display, score, square = resolve_startup_settings(
        None, None, False, False, 900, 0.0, 1200, 0.2, False
    )
    assert (display, score, square) == (1200, 0.2, False)

    display, score, square = resolve_startup_settings(
        900, 0.0, False, True, 900, 0.0, 1200, 0.2, False
    )
    assert (display, score, square) == (900, 0.0, True)

    display, score, square = resolve_startup_settings(
        None, None, True, False, 900, 0.0, 900, 0.0, True
    )
    assert square is False

    try:
        resolve_startup_settings(None, None, True, True, 900, 0.0, None, None, None)
    except ValueError:
        return
    raise AssertionError("both square and aspect should be rejected")


def test_image_folder_and_tiff_stack_round_trip():
    with tempfile.TemporaryDirectory() as folder:
        for index, name in enumerate(("frame_10.png", "frame_2.png", "frame_1.png"), start=1):
            image = np.full((8, 6, 3), index * 20, dtype=np.uint8)
            cv2.imwrite(osp.join(folder, name), image)

        ordered = [osp.basename(path) for path in list_image_files(folder)]
        assert ordered == ["frame_1.png", "frame_2.png", "frame_10.png"]

        reader = open_frame_source(folder)
        assert isinstance(reader, ImageSequenceReader)
        assert reader.total_frames == 3
        assert reader.get_fps() == 0.0
        reader.pause()
        paused, frame_idx, frame = next(iter(reader))
        assert paused and frame_idx == 0 and frame.shape == (8, 6, 3)

        reader.pause(False)
        paused, frame_idx, _frame = next(reader)
        assert not paused and frame_idx == 1
        reader.toggle_reverse_state(True)
        _paused, frame_idx, _frame = next(reader)
        assert frame_idx == 0
        _paused, frame_idx, _frame = next(reader)
        assert frame_idx == 0 and reader.get_pause_state()

        stack_path = osp.join(folder, "stack.tif")
        pages = [np.full((5, 4), value, dtype=np.uint8) for value in (10, 20, 30)]
        assert cv2.imwritemulti(stack_path, pages)
        stack = ImageSequenceReader(stack_path)
        assert stack.total_frames == 3
        first = stack.get_sample_frame()
        assert first.shape[2] == 3
        assert int(first[0, 0, 0]) == 10
        stack.pause(True)
        stack.set_playback_position(2)
        assert int(stack.get_current_frame()[0, 0, 0]) == 30


def test_cli_rejects_square_and_aspect_together():
    import subprocess
    import sys

    root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    completed = subprocess.run(
        [sys.executable, "main.py", "--square", "--use_aspect_ratio"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert completed.returncode != 0
    assert "only one of --square and --use_aspect_ratio" in (completed.stderr + completed.stdout)


def test_remove_object_keeps_surviving_label_ids():
    from src.demo_helpers.shared_ui_layout import PromptUI
    from src.demo_helpers.ui.buttons import ImmediateButton, ToggleButton
    from src.demo_helpers.ui.layout import VStack

    import main as app

    ui = PromptUI(np.zeros((16, 16, 3), dtype=np.uint8))
    init = torch.zeros((1, 1, 4, 4))

    def build_layout(sidebar):
        return VStack(sidebar)

    manager = app.ObjectSlotManager(
        init,
        0,
        6,
        ToggleButton("Rec"),
        ImmediateButton("Save"),
        ImmediateButton("Retry"),
        ImmediateButton("Clear"),
        ImmediateButton("Add"),
        ImmediateButton("Remove"),
        ui,
        build_layout,
    )
    assert manager.stable_ids == [1]
    manager.add_object()
    manager.add_object()
    assert manager.stable_ids == [1, 2, 3]

    frame = np.zeros((4, 4), dtype=np.uint8)
    frame[0, 0] = 1
    frame[0, 1] = 2
    frame[0, 2] = 3
    manager.results_buffer.record_frame(4, frame)
    assert manager.remove_object(1)
    assert manager.stable_ids == [1, 3]
    saved = manager.results_buffer.frames_dict[4]
    assert int(saved[0, 0]) == 1
    assert int(saved[0, 1]) == 0
    assert int(saved[0, 2]) == 3

    manager.add_object()
    assert manager.stable_ids[0] == 1
    assert manager.stable_ids[1] == 3
    assert 2 in manager.stable_ids


def test_failed_metric_export_keeps_the_buffer_and_retry_writes_metrics(monkeypatch):
    import main as app

    frames = {3: np.array([[1]], dtype=np.uint8), 8: np.array([[2]], dtype=np.uint8)}
    buffer = app.TrackingResultsBuffer.create()
    for frame_idx, image in frames.items():
        buffer.record_frame(frame_idx, image)

    class Obj:
        pending_analysis = None

        def __init__(self):
            self.results_buffer = buffer

    obj = Obj()
    notes = []

    class Toast:
        def notify(self, message, level="info", duration_sec=2.5):
            notes.append((level, message))

    class Window:
        def refocus(self):
            return None

    class Progress:
        def __init__(self, *_args, **_kwargs):
            pass

        def update(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

    monkeypatch.setattr(app, "ProgressWindow", Progress)
    monkeypatch.setattr(app, "pick_save_folder", lambda _path: tempfile.mkdtemp())
    monkeypatch.setattr(app, "ask_export_parameters", lambda **_kwargs: (10.0, 1.0))
    monkeypatch.setattr(app, "show_message_dialog", lambda *_args, **_kwargs: None)
    def fail_export(*_args, **_kwargs):
        raise RuntimeError("metrics failed")

    monkeypatch.setattr(app, "export_tracking_analysis", fail_export)

    saved = app.prompt_and_save_tracking_results(obj, "clip.mp4", Window(), history=None, toast=Toast(), video_fps=10)
    assert saved is False
    assert buffer.has_data()
    assert obj.pending_analysis is not None
    assert any("Retry Metrics" in message for _level, message in notes)
    folder = obj.pending_analysis["folder"]
    assert osp.isfile(osp.join(folder, "00003.tif"))
    assert osp.isfile(osp.join(folder, "00008.tif"))

    written = {}

    def succeed_export(save_folder, frames_dict, *_args, **_kwargs):
        written["folder"] = save_folder
        written["keys"] = set(frames_dict)
        return {}

    monkeypatch.setattr(app, "export_tracking_analysis", succeed_export)
    assert app.retry_pending_metrics(obj, Toast())
    assert written["keys"] == {3, 8}
    assert written["folder"] == folder
    assert obj.pending_analysis is None
    assert not buffer.has_data()


def test_lost_toast_uses_stable_object_ids():
    import main as app

    notes = []

    class Toast:
        def notify(self, message, level="info", duration_sec=2.5):
            notes.append(message)

    class Obj:
        stable_ids = [1, 4, 7]

    app.notify_lost_objects(Toast(), Obj(), [1])
    app.notify_lost_objects(Toast(), Obj(), [0, 2])
    assert notes == [
        "Object 4 lost (low score)",
        "Objects 1, 7 lost (low score)",
    ]


def test_close_request_stays_open_when_save_is_cancelled(monkeypatch):
    import main as app

    buffer = app.TrackingResultsBuffer.create()
    buffer.record_frame(1, np.array([[1]], dtype=np.uint8))

    class Obj:
        def __init__(self):
            self.results_buffer = buffer

    monkeypatch.setattr(app, "ask_save_discard_cancel", lambda *_args, **_kwargs: "save")
    monkeypatch.setattr(app, "prompt_and_save_tracking_results", lambda *_args, **_kwargs: False)

    class Window:
        def refocus(self):
            return None

    assert app.handle_close_request(Obj(), "clip.mp4", Window()) is False
    assert buffer.has_data()
