#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checks for the 1.8.0 storage, metrics, and documentation changes. Author: Lucien, 2026-09-23."""

from pathlib import Path

import numpy as np

from src.demo_helpers.analysis import build_msd_rows
from src.demo_helpers.misc import encode_side_warning, native_encode_side, resolve_encode_side
from src.demo_helpers.session import (
    read_session,
    session_from_manager,
    session_object_ids,
    session_video_mismatch,
    write_session,
)
from src.demo_helpers.sparse_labels import TrackingResultsBuffer, compress_label, decompress_label
from src.demo_helpers.ui.helpers.images import get_image_hw_for_max_side_length


ROOT = Path(__file__).resolve().parents[1]

IDENTICAL_V3_FILES = (
    "README.md",
    "__init__.py",
    "components/__init__.py",
    "components/exemplar_detector_attention.py",
    "components/exemplar_detector_components.py",
    "components/image_encoder_attention.py",
    "components/image_exemplar_fusion_attention.py",
    "components/mask_decoder_attention.py",
    "components/mask_decoder_transformer.py",
    "components/position_encoding.py",
    "components/sampling_encoder_attention.py",
    "components/sampling_encoder_components.py",
    "components/shared.py",
    "components/text_tokenizer.py",
    "resources/README.md",
    "state_dict_conversion/__init__.py",
    "state_dict_conversion/key_regex.py",
)


def test_sparse_label_round_trip_and_erase():
    label = np.zeros((20, 30), dtype=np.uint8)
    label[2:5, 4:8] = 1
    label[10, 12] = 7
    stored = compress_label(label)
    assert np.array_equal(decompress_label(stored), label)
    assert stored["objects"][1][4].nbytes < label.nbytes

    buffer = TrackingResultsBuffer.create()
    buffer.record_frame(4, label)
    assert buffer.contains_label(1)
    buffer.erase_label(1)
    saved = buffer.frames_dict[4]
    assert int(saved[2, 4]) == 0
    assert int(saved[10, 12]) == 7
    assert not buffer.contains_label(1)
    assert buffer.has_data()


def test_msd_matches_the_pair_definition():
    sequence = [
        (0, {"centroid_x": 0.0, "centroid_y": 0.0}),
        (2, {"centroid_x": 3.0, "centroid_y": 4.0}),
        (3, {"centroid_x": 3.0, "centroid_y": 0.0}),
        (6, {"centroid_x": 0.0, "centroid_y": 0.0}),
    ]
    rows = build_msd_rows({1: sequence}, pixel_size_um=1.0, fps=10.0)
    by_lag = {row["lag_frames"]: row for row in rows}
    assert by_lag[1]["num_pairs"] == 1
    assert by_lag[1]["msd_um2"] == 16.0
    assert by_lag[2]["num_pairs"] == 1
    assert by_lag[2]["msd_um2"] == 25.0
    assert by_lag[3]["num_pairs"] == 2
    assert by_lag[3]["msd_um2"] == 9.0
    assert 4 not in by_lag
    assert by_lag[1]["lag_time_s"] == 0.1


def test_native_encode_side_and_sam3_warning():
    assert native_encode_side("v2") == 1024
    assert native_encode_side("v3") == 1008
    assert native_encode_side("v3p1") == 1008
    assert resolve_encode_side(None) == 1344
    assert resolve_encode_side(1008) == 1008
    assert encode_side_warning("v3", 1008) is None
    assert encode_side_warning("v3", 1344) is None
    assert encode_side_warning("v2", 1024) is None
    assert encode_side_warning("v3p1", 1024) is not None


def test_session_records_model_and_encode_settings(tmp_path):
    class _Memory:
        prompt_frame_indices = [4]
        prompt_specs = [{"boxes": [(0.1, 0.2, 0.3, 0.4)], "fg_points": [(0.5, 0.6)], "bg_points": []}]

    class _Manager:
        stable_ids = [2]
        memory_list = [_Memory()]

    session = session_from_manager(
        _Manager(),
        model_path="model_weights/sam3.pt",
        encode_side=1008,
        use_square_sizing=False,
        video_path="clips/sample.avi",
    )
    assert session["version"] == 2
    assert session["model_path"] == "model_weights/sam3.pt"
    assert session["encode_side"] == 1008
    assert session["use_square_sizing"] is False
    assert session["video_path"].endswith("sample.avi")
    assert session["objects"][0]["stable_id"] == 2
    assert session_object_ids(session["objects"]) == [2]
    assert session_object_ids([{"stable_id": 7}, {}]) == [7, 1]
    assert session_video_mismatch({"video_path": session["video_path"]}, session["video_path"]) is None
    assert session_video_mismatch({"video_path": session["video_path"]}, "other.avi") is not None
    assert session_video_mismatch({}, "other.avi") is None
    path = tmp_path / "session.json"
    write_session(str(path), session)
    loaded = read_session(str(path))
    assert loaded["encode_side"] == 1008
    legacy = {"version": 1, "objects": session["objects"]}
    write_session(str(path), legacy)
    assert read_session(str(path)).get("model_path") is None


def test_max_side_length_uses_both_image_sides():
    wide = np.zeros((100, 200, 3), dtype=np.uint8)
    assert get_image_hw_for_max_side_length(wide, 500) == (250, 500)


def test_v3_and_v3p1_shared_files_stay_identical():
    left = ROOT / "src" / "v3_sam"
    right = ROOT / "src" / "v3p1_sam"
    for relative in IDENTICAL_V3_FILES:
        assert (left / relative).read_bytes() == (right / relative).read_bytes(), relative


def test_guides_document_the_current_defaults():
    guide = (ROOT / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8")
    window = (ROOT / "src" / "demo_helpers" / "ui" / "user_guide_window.py").read_text(encoding="utf-8")
    for phrase in ("--lost_patience", "--mask_select", "--intensity_range", "1024", "1008", "clockwise", "default 1344"):
        assert phrase in guide, phrase
        assert phrase in window, phrase
    assert "默认 1344" in window
    assert "Omit for 1024" not in guide
    assert "Omit for 1024" not in window
    assert "顺时针" in window
