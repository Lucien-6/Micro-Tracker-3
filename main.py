#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.8.2"
__app_name__ = "Micro Tracker 3"
__author__ = "Lucien"
__author_email__ = "lucien-6@qq.com"


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse
import gc
import os
import os.path as osp
import sys
import traceback
from datetime import datetime
from time import perf_counter
from enum import Enum
from dataclasses import dataclass

import torch
import cv2
import numpy as np

from src.make_sam import make_sam_from_state_dict

from src.demo_helpers.ui.window import DisplayWindow, KEY
from src.demo_helpers.ui.shortcuts_help import ShortcutsHelpWindow
from src.demo_helpers.ui.user_guide_window import UserGuideWindow
from src.demo_helpers.ui.video import (
    LoopingVideoPlaybackSlider,
    ValueChangeTracker,
)
from src.demo_helpers.ui.layout import GridStack, HStack, ScrollableGridViewport, VStack
from src.demo_helpers.ui.buttons import ToggleButton, ImmediateButton, RadioConstraint
from src.demo_helpers.ui.text import ValueBlock
from src.demo_helpers.ui.base import force_same_min_width
from src.demo_helpers.ui.overlays import DrawPolygonsOverlay, MiddleClickCaptureOverlay
from src.demo_helpers.ui.toast import ToastManager
from src.demo_helpers.ui.helpers.images import linear_gradient_image

from src.demo_helpers.shared_ui_layout import PromptUIControl, PromptUI

from src.demo_helpers.history_keeper import HistoryKeeper
from src.demo_helpers.encode_cache import EncodedImageCache, strip_detector_branch
from src.demo_helpers.exclusive_masks import resolve_overlapping_masks, suppress_lost_pixels
from src.demo_helpers.image_sequence import open_frame_source
from src.demo_helpers.loading import (
    clean_path_str,
    resolve_model_candidates,
    resolve_startup_settings,
    ask_save_discard_cancel,
    ask_yes_no,
    parse_intensity_range,
    pick_model_file,
    pick_frame_source,
    pick_save_path,
    pick_save_folder,
    ask_export_parameters,
    show_message_dialog,
)
from src.demo_helpers.prompts import check_have_prompts
from src.demo_helpers.contours import get_contours_from_mask
from src.demo_helpers.video_data_storage import SAMVideoMemoryBank
from src.demo_helpers.saving import (
    build_combined_label_image,
    load_saved_label_frames,
    make_mt_results_folder_name,
    save_tracking_label_tif_sequence,
)
from src.demo_helpers.misc import (
    DEFAULT_ENCODE_SIDE,
    PeriodicVRAMReport,
    encode_side_warning,
    get_default_device_string,
    make_device_config,
    resolve_encode_side,
)
from src.demo_helpers.sparse_labels import TrackingResultsBuffer
from src.demo_helpers.analysis import export_tracking_analysis
from src.demo_helpers.session import (
    read_session,
    session_from_manager,
    session_object_ids,
    session_video_mismatch,
    write_session,
)
from src.demo_helpers.tracking_policy import model_family_name, select_mask_index
from src.demo_helpers.ui.progress import ProgressWindow
from src.demo_helpers.tk_host import close_tk_root
from src.demo_helpers.model_info import get_token_hw, get_preencoding_hw


# ---------------------------------------------------------------------------------------------------------------------
# %% Helper Data types


@dataclass
class MaskResults:
    """Storage for (per-object) displayable masking results"""

    preds: torch.Tensor
    idx: int = 0
    objscore: float = 0.0
    frame_idx: int | None = None
    source: str = "none"
    version: int = 0

    @classmethod
    def create(cls, mask_predictions, mask_index=1, object_score=0.0):
        """Helper used to create an empty instance of mask results"""
        empty_predictions = torch.full_like(mask_predictions, -7)
        return cls(empty_predictions, mask_index, object_score)

    def clear(self):
        self.preds = torch.zeros_like(self.preds)
        self.objscore = 0.0
        self.frame_idx = None
        self.source = "none"
        self.version += 1
        return self

    def update(self, mask_predictions, mask_index, object_score=None, frame_idx=None, source=None):
        if mask_predictions is not None:
            self.preds = mask_predictions
        if mask_index is not None:
            self.idx = mask_index
        if object_score is not None:
            self.objscore = object_score
        if frame_idx is not None:
            self.frame_idx = int(frame_idx)
        if source is not None:
            self.source = source
        self.version += 1
        return self


def get_best_mask_index(iou_predictions: torch.Tensor) -> int:
    """Return the mask index with the highest IoU prediction."""
    return int(iou_predictions.argmax(dim=-1).squeeze().item())


def format_resource_button_label(kind: str, resource_name: str) -> str:
    return f"{kind}: {resource_name}"


NO_VIDEO_LABEL = "(select video)"


def create_placeholder_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Build a visual placeholder shown before the user selects a video file."""

    def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=14):
        r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)

    def put_text_centered(img, text, center_y, scale, color, thickness=1):
        (txt_w, txt_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
        x = max(8, (w - txt_w) // 2)
        y = center_y + txt_h // 2
        cv2.putText(
            img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_DUPLEX, scale, (18, 16, 24), thickness + 1, cv2.LINE_AA
        )
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

    img = linear_gradient_image(h, w, start_color=(30, 28, 52), end_color=(68, 52, 78), vertical=True)

    # Soft vignette
    vignette = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(vignette, (w // 2, h // 2), (int(w * 0.58), int(h * 0.62)), 0, 0, 360, 1.0, -1)
    vignette = cv2.GaussianBlur(vignette, (0, 0), sigmaX=w * 0.08, sigmaY=h * 0.08)
    img = np.clip(img.astype(np.float32) * (0.55 + 0.45 * vignette[..., None]), 0, 255).astype(np.uint8)

    cx, cy = w // 2, h // 2
    card_w, card_h = int(w * 0.74), int(h * 0.58)
    x1, y1 = cx - card_w // 2, cy - card_h // 2
    x2, y2 = x1 + card_w, y1 + card_h

    draw_rounded_rect(img, x1, y1, x2, y2, (46, 42, 58), radius=18)
    cv2.rectangle(img, (x1, y1), (x2, y2), (108, 118, 145), 2, cv2.LINE_AA)

    # Film-strip perforations
    hole_r = max(3, card_w // 80)
    hole_step = max(14, card_h // 14)
    for py in range(y1 + hole_step, y2 - hole_step // 2, hole_step):
        cv2.circle(img, (x1 + 14, py), hole_r, (32, 28, 44), -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - 14, py), hole_r, (32, 28, 44), -1, cv2.LINE_AA)

    # Play icon
    icon_cy = cy - card_h // 7
    icon_r = max(34, min(card_w, card_h) // 7)
    cv2.circle(img, (cx, icon_cy), icon_r, (58, 118, 88), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, icon_cy), icon_r, (130, 195, 155), 2, cv2.LINE_AA)
    tri_half_h = int(icon_r * 0.62)
    tri_half_w = int(icon_r * 0.72)
    play_tri = np.array(
        [[cx - tri_half_w // 2, icon_cy - tri_half_h], [cx - tri_half_w // 2, icon_cy + tri_half_h], [cx + tri_half_w, icon_cy]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(img, play_tri, (235, 245, 240), cv2.LINE_AA)

    # Upward cue toward the Video toolbar button (45° east of north)
    arrow_y = max(36, y1 - 12)
    arrow_len = 32
    arrow_d = int(arrow_len * 0.70710678)
    arrow_x_shift = int(w * 0.30)
    arrow_tail = (cx - arrow_d + arrow_x_shift, arrow_y + arrow_d)
    arrow_head = (cx + arrow_d + arrow_x_shift, arrow_y - arrow_d)
    cv2.arrowedLine(img, arrow_tail, arrow_head, (150, 165, 195), 2, tipLength=0.35, line_type=cv2.LINE_AA)

    put_text_centered(img, "No Video Loaded", cy + card_h // 10, 0.78, (210, 205, 225), 1)
    put_text_centered(img, "Click Video above to browse", cy + card_h // 10 + 38, 0.52, (165, 175, 200), 1)
    put_text_centered(img, "MP4  AVI  TIFF  IMAGE FOLDER", y2 - 28, 0.42, (120, 130, 155), 1)

    return img


def load_sam_models(model_path: str, device_config_dict: dict):
    print("", "Loading model weights...", f"  @ {model_path}", sep="\n", flush=True)
    sam_core = make_sam_from_state_dict(model_path)
    interact_model = sam_core.get_interactive_context()
    track_model = sam_core.get_tracking_context()
    # The tracking context owns every module the interactive path uses.
    # Detector-only modules stay on CPU.
    track_model.to(**device_config_dict)
    return sam_core, interact_model, track_model


def release_sam_runtime() -> None:
    """
    Free cached GPU memory.

    The caller must already have dropped every reference to the previous model,
    its contexts, and encoded tensors. Deleting a local name here would not.
    """

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def write_error_log(message: str) -> str:
    """Write an unexpected-error traceback next to the application."""

    folder = osp.join(osp.dirname(osp.abspath(__file__)), "error_logs")
    os.makedirs(folder, exist_ok=True)
    path = osp.join(folder, f"error_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(message)
    return path


def emergency_save_results(obj_mgr, video_path, history, video_fps: float, source_info=None) -> str | None:
    """
    Write in-memory labels without dialogs.

    Used when the application is exiting because of an error, so a modal
    dialog cannot be the only way to keep the recorded frames.
    """

    if not obj_mgr.results_buffer.has_data():
        return None
    parent = get_default_save_parent_folder(video_path, history)
    fps_value = float(video_fps) if video_fps and video_fps > 0 else 30.0
    pixel_size_um = 1.0
    if history is not None:
        has_val, stored_pixel = history.read("pixel_size_um")
        if has_val and isinstance(stored_pixel, (int, float)) and stored_pixel > 0:
            pixel_size_um = float(stored_pixel)
    save_folder = osp.join(parent, make_mt_results_folder_name(get_video_base_name_for_save(video_path)) + "_AUTOSAVE")
    save_folder, _num_saved = save_tracking_label_tif_sequence(
        obj_mgr.results_buffer.frames_dict, save_folder, fps=fps_value, source_info=source_info
    )
    try:
        export_tracking_analysis(
            save_folder,
            obj_mgr.results_buffer.frames_dict,
            video_path,
            fps_value,
            pixel_size_um,
        )
    except Exception as err:
        print("", f"Warning: emergency metric export failed: {err}", sep="\n", flush=True)
    obj_mgr.results_buffer.clear()
    print("", f"Auto-saved tracking labels @ {save_folder}", sep="\n", flush=True)
    return save_folder


def confirm_before_discarding_results(
    action: str,
    obj_mgr,
    video_path,
    window,
    history,
    toast,
    video_fps,
    source_info=None,
    intensity_range="auto",
) -> bool:
    """Return True when it is safe to continue with action (results saved or discarded)."""

    if not obj_mgr.results_buffer.has_data():
        return True
    choice = ask_save_discard_cancel(
        f"There are unsaved tracking results in memory.\n\nSave them before you {action}?"
    )
    window.refocus()
    if choice == "save":
        return prompt_and_save_tracking_results(
            obj_mgr, video_path, window, history, toast, video_fps, source_info, intensity_range
        )
    if choice == "discard":
        obj_mgr.results_buffer.clear()
        return True
    if choice is None:
        print("", "Warning: save dialog unavailable; the switch was cancelled.", sep="\n", flush=True)
    return False


def run_initial_model_pass(interact_model, sample_frame, imgenc_config_dict):
    print("", "Encoding image data...", sep="\n", flush=True)
    t1 = perf_counter()
    encoded_img = interact_model.encode_image(sample_frame, **imgenc_config_dict)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = perf_counter()
    time_taken_ms = round(1000 * (t2 - t1))
    print(f"  -> Took {time_taken_ms} ms", flush=True)

    prompts = ([], [], [])
    encoded_prompts = interact_model.encode_prompts(*prompts)
    init_mask_preds, iou_preds = interact_model.generate_masks(
        encoded_img, encoded_prompts, blank_promptless_output=False
    )
    init_mask_idx = get_best_mask_index(iou_preds)
    max_side = imgenc_config_dict["max_side_length"]
    use_square = imgenc_config_dict["use_square_sizing"]
    preencode_hw = get_preencoding_hw(interact_model, sample_frame, max_side, use_square)
    token_hw = get_token_hw(encoded_img)
    return encoded_img, init_mask_preds, iou_preds, init_mask_idx, preencode_hw, token_hw


def print_model_config(model_name, device_config_dict, preencode_hw, token_hw, requested_side=None):
    model_device = device_config_dict["device"]
    model_dtype = str(device_config_dict["dtype"]).split(".")[-1]
    image_hw_str = f"{preencode_hw[0]} x {preencode_hw[1]}"
    token_hw_str = f"{token_hw[0]} x {token_hw[1]}"
    lines = [
        "",
        f"Config ({model_name}):",
        f"  Device: {model_device} ({model_dtype})",
        f"  Resolution HW: {image_hw_str}",
        f"  Tokens HW: {token_hw_str}",
    ]
    if requested_side is not None:
        lines.append(f"  Requested side: {int(requested_side)}")
    print(*lines, sep="\n", flush=True)


def apply_encode_side(config: dict, family: str, explicit_side: int | None) -> int:
    """Write the encode side into the shared image-encoder config and warn when needed."""

    side = resolve_encode_side(explicit_side)
    config["max_side_length"] = side
    warning = encode_side_warning(family, side)
    if warning:
        print(f"  Warning: {warning}", flush=True)
    return side


def format_device_label(device_config_dict: dict) -> str:
    device = str(device_config_dict["device"])
    dtype = str(device_config_dict["dtype"]).split(".")[-1]
    return f"{device}/{dtype}"


def clear_hover_preview_mask(obj_mgr, objidx: int, track_idx_keeper) -> None:
    """Remove transient hover preview from display; restore tracker mask if prompts exist."""
    if not obj_mgr.memory_list[objidx].check_has_prompts():
        obj_mgr.maskresults_list[objidx].clear()
    else:
        track_idx_keeper.clear()


def apply_zero_tracking_mask(obj_mgr, objidx: int, frame_idx: int | None = None) -> None:
    """Zero the display mask for an object without running video masking."""
    maskresult = obj_mgr.maskresults_list[objidx]
    zero_preds = maskresult.preds * 0.0
    obj_mgr.maskresults_list[objidx].update(
        zero_preds, maskresult.idx, maskresult.objscore, frame_idx=frame_idx, source="track"
    )


def playback_direction(reverse_video: bool) -> int:
    """+1 when playback moves toward later frames, -1 when it moves backward."""
    return -1 if reverse_video else 1


def track_single_object_at_frame(
    objidx: int,
    frame_idx: int,
    obj_mgr,
    track_model,
    encoded_img,
    object_score_threshold: float,
    keep_tracking_after_loss: bool,
    is_trackhistory_enabled: bool,
    lost_objects_out: list | None = None,
    frame_direction: int = 1,
    mask_select_mode: str = "legacy",
    model_family: str = "v2",
    lost_patience: int = 1,
    max_prompt_attn: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, float | None]:
    """Run video masking for one object slot, or zero its mask if tracking is stopped at this frame."""
    memory = obj_mgr.memory_list[objidx]
    if not memory.check_has_prompts():
        return None, None, None

    memory.reconcile_tracking_stop_frame(frame_idx)
    if not memory.should_run_video_masking(frame_idx, keep_tracking_after_loss):
        apply_zero_tracking_mask(obj_mgr, objidx, frame_idx)
        return None, None, None

    # Drop frame memory before inference when this frame does not continue the chain.
    chain = memory.begin_frame(frame_idx, frame_direction)
    memory_kwargs = memory.to_dict()
    if max_prompt_attn is not None and model_family in ("v3", "v3p1"):
        memory_kwargs["prompt_memory_encodings"] = memory.prompt_encodings_for_frame(frame_idx, max_prompt_attn)
    mask_preds, iou_preds, obj_ptr, obj_score = track_model.step_video_masking(
        encoded_img, **memory_kwargs, return_best_only=False
    )
    obj_score_float = float(obj_score)
    tracked_mask_idx = select_mask_index(iou_preds, model_family, "track", mask_select_mode)

    encode_memory = False
    if obj_score_float < object_score_threshold:
        mask_preds = mask_preds * 0.0
        if not keep_tracking_after_loss:
            if memory.note_score(frame_idx, True, lost_patience) and lost_objects_out is not None:
                lost_objects_out.append(objidx)
        else:
            memory.note_score(frame_idx, False, lost_patience)
    else:
        memory.note_score(frame_idx, False, lost_patience)
        if is_trackhistory_enabled and chain != "same":
            encode_memory = True

    obj_mgr.maskresults_list[objidx].update(
        mask_preds, tracked_mask_idx, obj_score_float, frame_idx=frame_idx, source="track"
    )
    memory.note_playhead(frame_idx, frame_direction)
    if encode_memory:
        obj_mgr.pending_memory.append(
            {
                "objidx": objidx,
                "obj_ptr": obj_ptr,
                "obj_score": obj_score,
                "mask_index": tracked_mask_idx,
                "frame_idx": int(frame_idx),
                "direction": int(frame_direction),
            }
        )
    return mask_preds, iou_preds, obj_score_float


def run_multi_object_tracking(
    obj_mgr,
    track_model,
    encoded_img,
    frame_idx: int,
    object_score_threshold,
    keep_tracking_after_loss,
    is_trackhistory_enabled,
    lost_objects_out: list | None = None,
    frame_direction: int = 1,
    mask_select_mode: str = "legacy",
    model_family: str = "v2",
    lost_patience: int = 1,
    max_prompt_attn: int | None = None,
) -> None:
    """Run video masking for every object that has stored prompts."""
    for objidx in obj_mgr.objiter:
        track_single_object_at_frame(
            objidx,
            frame_idx,
            obj_mgr,
            track_model,
            encoded_img,
            object_score_threshold,
            keep_tracking_after_loss,
            is_trackhistory_enabled,
            lost_objects_out,
            frame_direction,
            mask_select_mode,
            model_family,
            lost_patience,
            max_prompt_attn,
        )


def refresh_exclusive_masks(obj_mgr, uictrl, frame_hw) -> None:
    """Assign overlapping foreground pixels to the higher-scoring object."""

    frame_hw = (int(frame_hw[0]), int(frame_hw[1]))
    originals = {}
    labeled = []
    for objidx in obj_mgr.objiter:
        if not obj_mgr.memory_list[objidx].check_has_prompts():
            continue
        maskresult = obj_mgr.maskresults_list[objidx]
        hires = uictrl.create_hires_mask_uint8(maskresult.preds, maskresult.idx, frame_hw)
        originals[objidx] = hires
        labeled.append((obj_mgr.stable_ids[objidx], hires, float(maskresult.objscore)))

    resolved_by_id = resolve_overlapping_masks(labeled)
    obj_mgr.exclusive_hw = frame_hw
    obj_mgr.mask_before_exclusion = originals
    obj_mgr.exclusive_masks = {
        objidx: resolved_by_id[obj_mgr.stable_ids[objidx]] for objidx in originals
    }


def commit_pending_frame_memory(obj_mgr, track_model, encoded_img) -> None:
    """Encode frame memory from masks after overlap has been removed."""

    for item in obj_mgr.pending_memory:
        objidx = item["objidx"]
        if not (0 <= objidx < len(obj_mgr.memory_list)):
            continue
        maskresult = obj_mgr.maskresults_list[objidx]
        preds = maskresult.preds
        original = obj_mgr.mask_before_exclusion.get(objidx)
        exclusive = obj_mgr.exclusive_masks.get(objidx)
        if original is not None and exclusive is not None:
            removed = (original > 0) & (exclusive == 0)
            preds = suppress_lost_pixels(preds, item["mask_index"], removed)
        mem_enc = track_model.encode_frame_memory(
            encoded_img,
            preds,
            item["obj_ptr"],
            item["obj_score"],
            mask_index=item["mask_index"],
        )
        obj_mgr.memory_list[objidx].store_frame_result(mem_enc)
    obj_mgr.pending_memory.clear()


def finalize_frame_masks(obj_mgr, uictrl, track_model, encoded_img, frame_hw) -> None:
    """Resolve overlaps, then write frame memory for inferences queued on this frame."""

    refresh_exclusive_masks(obj_mgr, uictrl, frame_hw)
    if encoded_img is not None:
        commit_pending_frame_memory(obj_mgr, track_model, encoded_img)
    else:
        obj_mgr.pending_memory.clear()


def notify_lost_objects(toast, obj_mgr, lost_objects) -> None:
    """Emit a transient warning toast when one or more objects are newly lost."""
    if toast is None or not lost_objects:
        return
    stable_ids = []
    for objidx in lost_objects:
        objidx = int(objidx)
        if 0 <= objidx < len(obj_mgr.stable_ids):
            stable_ids.append(int(obj_mgr.stable_ids[objidx]))
    ids = sorted(set(stable_ids))
    if len(ids) == 0:
        return
    if len(ids) == 1:
        toast.notify(f"Object {ids[0]} lost (low score)", level="warning")
    else:
        id_str = ", ".join(str(i) for i in ids)
        toast.notify(f"Objects {id_str} lost (low score)", level="warning")


def mask_for_object(obj_mgr, objidx, uictrl, frame_hw) -> np.ndarray:
    """Return the exclusive display mask when it matches this frame size."""

    frame_hw = (int(frame_hw[0]), int(frame_hw[1]))
    cached = obj_mgr.exclusive_masks.get(objidx)
    if cached is not None and obj_mgr.exclusive_hw == frame_hw and tuple(cached.shape[:2]) == frame_hw:
        return cached
    maskresult = obj_mgr.maskresults_list[objidx]
    return uictrl.create_hires_mask_uint8(maskresult.preds, maskresult.idx, frame_hw)


def collect_mask_contours(obj_mgr, buffer_select_idx, uictrl, frame_hw):
    """Build selected/unselected contour lists from current mask results."""
    selected_mask_contours, selected_mask_uint8 = None, None
    unselected_contours = []
    for objidx, _maskresult in enumerate(obj_mgr.maskresults_list):
        mask_uint8 = mask_for_object(obj_mgr, objidx, uictrl, frame_hw)
        _, mask_contours_norm = get_contours_from_mask(mask_uint8, normalize=True)
        mask_contours_norm = tuple(mask_contours_norm)
        if objidx == buffer_select_idx:
            selected_mask_contours = tuple(mask_contours_norm)
            selected_mask_uint8 = mask_uint8
        else:
            unselected_contours.extend(mask_contours_norm)
    return selected_mask_uint8, selected_mask_contours, unselected_contours


def find_object_index_at_mask_xy(obj_mgr, active_idx: int, xy_norm, uictrl, output_hw) -> int | None:
    """Return the object index under xy_norm inside a stored-prompt mask, or None."""
    hits = []
    for objidx in obj_mgr.objiter:
        if not obj_mgr.memory_list[objidx].check_has_prompts():
            continue
        mask_uint8 = mask_for_object(obj_mgr, objidx, uictrl, output_hw)
        h, w = mask_uint8.shape[0:2]
        x_px = min(max(int(round(xy_norm[0] * (w - 1))), 0), w - 1)
        y_px = min(max(int(round(xy_norm[1] * (h - 1))), 0), h - 1)
        if mask_uint8[y_px, x_px] > 0:
            hits.append(objidx)

    if not hits:
        return None

    non_active_hits = [objidx for objidx in hits if objidx != active_idx]
    if non_active_hits:
        return non_active_hits[-1]
    return active_idx


def apply_display_contours(
    uictrl, unselected_olay, frame, selected_mask_uint8, selected_mask_contours, unselected_contours
):
    uictrl.update_main_display_image(frame, selected_mask_uint8, selected_mask_contours)
    unselected_olay.set_polygons(unselected_contours)


def clear_tracking_ui_state(
    uictrl,
    ui_elems,
    unselected_olay,
    obj_mgr,
    init_mask_preds,
    init_mask_idx,
    track_btn,
    enable_record_btn,
    reversal_btn,
    imgenc_idx_keeper,
    track_idx_keeper,
    num_prompts_text,
    num_history_text,
    vreader=None,
):
    uictrl.read_prompts()
    ui_elems.clear_prompts()
    ui_elems.olays.polygon.set_polygons(())
    unselected_olay.set_polygons([])
    obj_mgr.reset_to_single_object(init_mask_preds, init_mask_idx, clear_prompts=False)
    track_btn.toggle(False, flag_if_changed=False)
    enable_record_btn.toggle(False, flag_if_changed=False)
    reversal_btn.toggle(False, flag_if_changed=False)
    if vreader is not None and hasattr(vreader, "toggle_reverse_state"):
        vreader.toggle_reverse_state(False)
    ui_elems.enable_tools(True, clear_prompt_data_on_disable=False)
    imgenc_idx_keeper.clear(-1)
    track_idx_keeper.clear(-1)
    num_prompts_text.set_value(0)
    num_history_text.set_value(0)
    obj_mgr.results_buffer.clear()


def get_video_base_name_for_save(video_path) -> str:
    if video_path is None:
        return "video"
    return osp.splitext(osp.basename(str(video_path)))[0]


def get_default_save_parent_folder(video_path, history=None) -> str:
    if history is not None:
        has_folder, saved_folder = history.read("save_folder")
        if has_folder and saved_folder and osp.isdir(saved_folder):
            return saved_folder
    if video_path is not None:
        parent = osp.dirname(clean_path_str(str(video_path)))
        if osp.isdir(parent):
            return parent
    return os.getcwd()


def build_object_label_masks(obj_mgr, uictrl, frame_hw) -> list[tuple[int, np.ndarray]]:
    labeled_masks = []
    for objidx in obj_mgr.objiter:
        if not obj_mgr.memory_list[objidx].check_has_prompts():
            continue
        mask_1ch = mask_for_object(obj_mgr, objidx, uictrl, frame_hw)
        labeled_masks.append((obj_mgr.stable_ids[objidx], mask_1ch))
    return labeled_masks


def record_combined_tracking_frame(obj_mgr, uictrl, frame_idx: int, frame_hw):
    labeled_masks = build_object_label_masks(obj_mgr, uictrl, frame_hw)
    if len(labeled_masks) == 0:
        return
    label_img = build_combined_label_image(frame_hw, labeled_masks)
    obj_mgr.results_buffer.record_frame(frame_idx, label_img)


def maybe_record_current_frame(obj_mgr, uictrl, frame_idx: int, frame_hw, enabled: bool) -> None:
    """Record a frame once every prompted object has a tracker mask for that frame."""

    if not enabled:
        return
    prompted = [objidx for objidx in obj_mgr.objiter if obj_mgr.memory_list[objidx].check_has_prompts()]
    if len(prompted) == 0:
        return
    results = [obj_mgr.maskresults_list[objidx] for objidx in prompted]
    if any(result.source != "track" or result.frame_idx != int(frame_idx) for result in results):
        return
    signature = (int(frame_idx), tuple(result.version for result in results))
    if signature == obj_mgr.last_record_signature:
        return
    record_combined_tracking_frame(obj_mgr, uictrl, frame_idx, frame_hw)
    obj_mgr.last_record_signature = signature


class PromptUndoManager:
    """
    Global undo stack for interactive prompts (FG/BG points and boxes).

    The stack records the order in which prompts were added across all three
    overlays so that Ctrl+Z removes the single most-recently-added prompt,
    regardless of which tool created it. The stack is reconciled against the
    actual overlay contents every frame, so external clears (tool switches,
    Store/Clear Prompts, object switches, etc.) stay consistent automatically.
    """

    def __init__(self, ui_elems):
        self._fgpt = ui_elems.olays.fgpt
        self._bgpt = ui_elems.olays.bgpt
        self._box = ui_elems.olays.box
        self._stack: list[str] = []

    def _counts(self) -> dict[str, int]:
        return {"fg": self._fgpt.num_points(), "bg": self._bgpt.num_points(), "box": self._box.num_boxes()}

    def sync(self) -> None:
        """Reconcile the stack so its tag multiset matches current overlay contents."""
        target = self._counts()
        have = {"fg": 0, "bg": 0, "box": 0}

        # Keep existing order for items still present (drops extras of any over-represented tag)
        new_stack = []
        for tag in self._stack:
            if have[tag] < target[tag]:
                new_stack.append(tag)
                have[tag] += 1

        # Append any newly added prompts at the end (most-recent)
        for tag in ("fg", "bg", "box"):
            while have[tag] < target[tag]:
                new_stack.append(tag)
                have[tag] += 1

        self._stack = new_stack

    def undo(self) -> bool:
        """Remove the most-recently-added prompt. Returns True if something was removed."""
        if len(self._stack) == 0:
            return False
        tag = self._stack.pop()
        if tag == "fg":
            self._fgpt.remove_last()
        elif tag == "bg":
            self._bgpt.remove_last()
        else:
            self._box.remove_last()
        return True


def prompt_and_save_tracking_results(
    obj_mgr, video_path, window, history=None, toast=None, video_fps=30.0, source_info=None, intensity_range="auto"
) -> bool:
    if not obj_mgr.results_buffer.has_data():
        return False

    default_parent = get_default_save_parent_folder(video_path, history)
    parent_folder = pick_save_folder(default_parent)
    window.refocus()
    if parent_folder is None:
        return False

    # Ask for physical parameters needed to compute tracking metrics
    default_pixel_size_um = 1.0
    if history is not None:
        has_val, stored_pixel = history.read("pixel_size_um")
        if has_val and isinstance(stored_pixel, (int, float)) and stored_pixel > 0:
            default_pixel_size_um = float(stored_pixel)
    export_params = ask_export_parameters(default_fps=video_fps, default_pixel_size_um=default_pixel_size_um)
    window.refocus()
    if export_params is None:
        return False
    if len(export_params) == 2:
        fps_value, pixel_size_um = export_params
        max_frame_gap = 1
    else:
        fps_value, pixel_size_um, max_frame_gap = export_params

    video_base_name = get_video_base_name_for_save(video_path)
    results_folder_name = make_mt_results_folder_name(video_base_name)
    save_folder = osp.join(parent_folder, results_folder_name)

    # Progress window driven from the main thread while the (blocking) save runs.
    # Stages are mapped onto a single continuous bar: label images -> metrics -> overlay video.
    progress = ProgressWindow("Saving Tracking Results")

    def report_progress(stage, current, total):
        total = max(int(total), 1)
        frac_local = current / total
        if stage == "Saving label images":
            overall, msg = 0.45 * frac_local, f"Saving label images ({current}/{total})"
        elif stage == "Computing metrics":
            overall, msg = 0.45 + 0.05 * frac_local, "Computing metrics..."
        elif stage == "Rendering overlay video":
            overall, msg = 0.50 + 0.50 * frac_local, f"Rendering overlay video ({current}/{total})"
        else:
            overall, msg = frac_local, str(stage)
        progress.update(msg, overall)

    try:
        try:
            save_folder, num_saved = save_tracking_label_tif_sequence(
                obj_mgr.results_buffer.frames_dict,
                save_folder,
                progress_cb=report_progress,
                fps=fps_value,
                source_info=source_info,
            )
        except IOError as err:
            show_message_dialog("Save Failed", f"Could not save tracking results:\n\n{err}", kind="error")
            return False

        # Export metrics (velocity/displacement/orientation/MSD) + overlay video alongside the label sequence
        try:
            summary = export_tracking_analysis(
                save_folder,
                obj_mgr.results_buffer.frames_dict,
                video_path,
                fps_value,
                pixel_size_um,
                progress_cb=report_progress,
                max_frame_gap=max_frame_gap,
                intensity_range=intensity_range,
            )
            missed = int(summary.get("overlay_missed_frames") or 0)
            if missed and toast is not None:
                toast.notify(f"Overlay missed {missed} source frame(s)", level="warning", duration_sec=4.0)
        except Exception as err:
            obj_mgr.pending_analysis = {
                "folder": save_folder,
                "video_path": video_path,
                "fps": float(fps_value),
                "pixel_size_um": float(pixel_size_um),
                "max_frame_gap": int(max_frame_gap),
                "intensity_range": intensity_range,
            }
            show_message_dialog(
                "Export Warning",
                "Tracking label images were saved, but metric/overlay export failed:\n\n"
                f"{err}\n\nUse Retry Metrics to write the CSV files and overlay into the same folder.",
                kind="warning",
            )
            if history is not None:
                history.store(save_folder=parent_folder, pixel_size_um=float(pixel_size_um))
            if toast is not None:
                toast.notify(
                    "Labels saved. Metrics were not written. Use Retry Metrics.",
                    level="warning",
                    duration_sec=4.0,
                )
            return False

        progress.update("Done", 1.0, force=True)
    finally:
        progress.close()
        window.refocus()

    if history is not None:
        history.store(save_folder=parent_folder, pixel_size_um=float(pixel_size_um))

    print("", f"Saved {num_saved} tracking result frame(s)", f"@ {save_folder}", sep="\n", flush=True)
    if toast is not None:
        toast.notify(f"Saved {num_saved} frame(s) + metrics", level="success")
    obj_mgr.pending_analysis = None
    obj_mgr.results_buffer.clear()
    return True


def retry_pending_metrics(obj_mgr, toast=None, window=None) -> bool:
    """Write metrics and the overlay video for a folder whose label images already exist."""

    pending = obj_mgr.pending_analysis
    if not pending:
        if toast is not None:
            toast.notify("No unfinished metrics export", level="warning")
        return False

    progress = ProgressWindow("Retry Metrics")

    def report_progress(stage, current, total):
        total = max(int(total), 1)
        frac_local = current / total
        if stage == "Computing metrics":
            overall, msg = 0.1 * frac_local, "Computing metrics..."
        elif stage == "Rendering overlay video":
            overall, msg = 0.1 + 0.9 * frac_local, f"Rendering overlay video ({current}/{total})"
        else:
            overall, msg = frac_local, str(stage)
        progress.update(msg, overall)

    try:
        frames = load_saved_label_frames(pending["folder"])
        summary = export_tracking_analysis(
            pending["folder"],
            frames,
            pending["video_path"],
            pending["fps"],
            pending["pixel_size_um"],
            progress_cb=report_progress,
            max_frame_gap=int(pending.get("max_frame_gap", 1)),
            intensity_range=pending.get("intensity_range", "auto"),
        )
        missed = int((summary or {}).get("overlay_missed_frames") or 0)
        if missed and toast is not None:
            toast.notify(f"Overlay missed {missed} source frame(s)", level="warning", duration_sec=4.0)
        progress.update("Done", 1.0, force=True)
    except Exception as err:
        show_message_dialog(
            "Export Warning",
            f"Metric/overlay export failed again:\n\n{err}",
            kind="warning",
        )
        return False
    finally:
        progress.close()
        if window is not None:
            window.refocus()

    saved_keys = set(frames.keys())
    live_keys = set(obj_mgr.results_buffer.frames_dict.keys())
    obj_mgr.pending_analysis = None
    if live_keys == saved_keys:
        obj_mgr.results_buffer.clear()
        if toast is not None:
            toast.notify("Metrics saved", level="success")
    elif toast is not None:
        toast.notify("Metrics saved. Newer frames are still in memory.", level="success")
    return True


def handle_close_request(
    obj_mgr, video_path, window, history=None, toast=None, video_fps=30.0, source_info=None, intensity_range="auto"
) -> bool:
    """Return True when the application should exit, False to keep running."""

    if obj_mgr.results_buffer.has_data():
        choice = ask_save_discard_cancel(
            "There are unsaved tracking results in memory.\n\nSave them before closing?"
        )
        window.refocus()
        if choice == "cancel":
            return False
        if choice is None:
            print("", "Warning: save dialog unavailable. Writing an automatic copy.", sep="\n", flush=True)
            emergency_save_results(obj_mgr, video_path, history, video_fps, source_info)
            return True
        if choice == "save":
            if not prompt_and_save_tracking_results(
                obj_mgr, video_path, window, history, toast, video_fps, source_info, intensity_range
            ):
                return False
        else:
            obj_mgr.results_buffer.clear()
        return True

    memories = getattr(obj_mgr, "memory_list", [])
    has_prompts = any(memory.check_has_prompts() for memory in memories)
    if not has_prompts:
        return True
    discard_prompts = ask_yes_no(
        "Quit",
        "Stored prompts have not been saved as a session.\n\nQuit and discard them?",
    )
    window.refocus()
    if discard_prompts is None:
        print("", "Warning: dialog unavailable. Staying open so prompts are not discarded.", sep="\n", flush=True)
        return False
    return bool(discard_prompts)


class ObjectSlotManager:
    """Manage dynamic object slots, sidebar UI, and per-object tracking/save data."""

    MAX_OBJECTS = 255
    OBJECT_GRID_SCROLL_THRESHOLD = 32
    OBJECT_BUTTON_ROW_HEIGHT = 20
    OBJECT_GRID_COLUMNS = 2

    def __init__(
        self,
        init_mask_preds,
        init_mask_idx,
        max_memory_history,
        enable_record_btn,
        buffer_save_btn,
        retry_metrics_btn,
        buffer_clear_btn,
        add_object_btn,
        remove_object_btn,
        ui_elems,
        build_disp_layout_fn,
    ):
        self.init_mask_preds = init_mask_preds
        self.init_mask_idx = init_mask_idx
        self.max_memory_history = max_memory_history
        self.max_prompt_memory = 32

        self.enable_record_btn = enable_record_btn
        self.buffer_save_btn = buffer_save_btn
        self.retry_metrics_btn = retry_metrics_btn
        self.buffer_clear_btn = buffer_clear_btn
        self.add_object_btn = add_object_btn
        self.remove_object_btn = remove_object_btn
        self.ui_elems = ui_elems
        self.build_disp_layout_fn = build_disp_layout_fn

        self.maskresults_list: list[MaskResults] = []
        self.stable_ids: list[int] = []
        self.results_buffer = TrackingResultsBuffer.create()
        self.memory_list: list[SAMVideoMemoryBank] = []
        self.exclusive_masks: dict[int, np.ndarray] = {}
        self.mask_before_exclusion: dict[int, np.ndarray] = {}
        self.exclusive_hw: tuple[int, int] | None = None
        self.pending_memory: list[dict] = []
        self.pending_analysis: dict | None = None
        self.last_record_signature = None
        self.buffer_btns_list: list[ToggleButton] = []
        self.object_grid = None
        self._object_grid_viewport_h = 0
        self.buffer_btn_constraint = RadioConstraint(ToggleButton("Object 1"))
        self.save_sidebar = None
        self.disp_layout = None
        self.window = None

        self.add_object(select_new=True, clear_prompts=False)

    @property
    def objiter(self):
        return list(range(len(self.maskresults_list)))

    def get_select_idx(self) -> int:
        return self.buffer_btn_constraint._select_idx

    def read_selection(self) -> tuple[bool, int, ToggleButton]:
        return self.buffer_btn_constraint.read()

    def ensure_active_object_visible(self):
        """Keep the active object button in view when the object list scrolls."""
        if self.object_grid is None or not hasattr(self.object_grid, "ensure_object_visible"):
            return
        self.object_grid.ensure_object_visible(self.get_select_idx())

    def previous_object(self):
        result = self.buffer_btn_constraint.previous()
        self.ensure_active_object_visible()
        return result

    def next_object(self):
        result = self.buffer_btn_constraint.next()
        self.ensure_active_object_visible()
        return result

    def select_object(self, objidx: int) -> bool:
        """Select an object by index. Returns True if the selection changed."""
        if not (0 <= objidx < len(self.maskresults_list)):
            return False
        changed = objidx != self.get_select_idx()
        if changed:
            self.buffer_btn_constraint.change_to(objidx)
        self.ensure_active_object_visible()
        return changed

    def _allocate_stable_id(self) -> int | None:
        """Smallest unused label id in 1..255. Freed ids can be reused after their pixels are erased."""
        used = set(self.stable_ids)
        for candidate in range(1, self.MAX_OBJECTS + 1):
            if candidate not in used:
                return candidate
        return None

    def _clear_mask_cache(self) -> None:
        self.exclusive_masks.clear()
        self.mask_before_exclusion.clear()
        self.exclusive_hw = None
        self.pending_memory.clear()

    def _append_object_data(self) -> bool:
        stable_id = self._allocate_stable_id()
        if stable_id is None:
            return False
        self.stable_ids.append(stable_id)
        self.maskresults_list.append(MaskResults.create(self.init_mask_preds, self.init_mask_idx))
        self.memory_list.append(SAMVideoMemoryBank(self.max_memory_history, max_prompt_memory=self.max_prompt_memory))
        return True

    def _rebuild_object_rows(self):
        self.buffer_btns_list = []

        for _objidx, stable_id in enumerate(self.stable_ids):
            buffer_btn = ToggleButton(f"Object {stable_id}", button_height=20, text_scale=0.5, on_color=(145, 120, 65))
            self.buffer_btns_list.append(buffer_btn)

        if len(self.buffer_btns_list) > 1:
            force_same_min_width(*self.buffer_btns_list)

    def sync_object_grid_viewport_height(self):
        """Store the fitted object-grid height (used when scrolling starts above the threshold)."""
        if self.object_grid is None:
            return
        ref_h = self.object_grid.get_reference_viewport_h()
        if ref_h > 0:
            self._object_grid_viewport_h = ref_h

    def finalize_layout_callback_regions(self):
        """Sync object-button hit boxes after a full layout render."""
        if self.object_grid is not None and hasattr(self.object_grid, "finalize_callback_regions"):
            self.object_grid.finalize_callback_regions()

    def _rebuild_ui(self, select_idx: int, refresh_window: bool = True):
        select_idx = max(0, min(select_idx, len(self.maskresults_list) - 1))
        if self.object_grid is not None:
            self.sync_object_grid_viewport_height()
        self._rebuild_object_rows()
        self.buffer_btn_constraint = RadioConstraint(*self.buffer_btns_list, initial_selected_index=select_idx)
        inner_grid = GridStack(*self.buffer_btns_list, num_columns=self.OBJECT_GRID_COLUMNS).set_debug_name("ObjectGrid")
        self.object_grid = ScrollableGridViewport(
            inner_grid,
            num_columns=self.OBJECT_GRID_COLUMNS,
            row_height=self.OBJECT_BUTTON_ROW_HEIGHT,
            scroll_when_more_than=self.OBJECT_GRID_SCROLL_THRESHOLD,
            reference_viewport_h=self._object_grid_viewport_h,
        ).set_debug_name("ObjectGridViewport")
        self.save_sidebar = VStack(
            self.enable_record_btn,
            self.object_grid,
            HStack(self.add_object_btn, self.remove_object_btn),
            HStack(self.buffer_save_btn, self.retry_metrics_btn, self.buffer_clear_btn),
        )
        self.disp_layout = self.build_disp_layout_fn(self.save_sidebar)
        self.ensure_active_object_visible()
        if refresh_window and self.window is not None:
            self.window.replace_mouse_callbacks(self.disp_layout)

    def bind_window(self, window: DisplayWindow):
        self.window = window
        self._rebuild_ui(self.get_select_idx(), refresh_window=True)

    def add_object(self, select_new=True, clear_prompts=True) -> bool:
        if len(self.maskresults_list) >= self.MAX_OBJECTS:
            return False

        if not self._append_object_data():
            return False
        self._clear_mask_cache()
        select_idx = len(self.maskresults_list) - 1 if select_new else self.get_select_idx()
        self._rebuild_ui(select_idx)
        if clear_prompts:
            self.ui_elems.clear_prompts()
        return True

    def remove_object(self, objidx: int, clear_prompts=True) -> bool:
        if len(self.maskresults_list) <= 1:
            return False
        if not (0 <= objidx < len(self.maskresults_list)):
            return False

        removed_id = self.stable_ids[objidx]
        self.results_buffer.erase_label(removed_id)
        old_select = self.get_select_idx()
        del self.maskresults_list[objidx]
        del self.memory_list[objidx]
        del self.stable_ids[objidx]
        self._clear_mask_cache()

        if old_select > objidx:
            new_select = old_select - 1
        elif old_select == objidx:
            new_select = min(old_select, len(self.maskresults_list) - 1)
        else:
            new_select = old_select

        self._rebuild_ui(new_select)
        if clear_prompts:
            self.ui_elems.clear_prompts()
        return True

    def remove_selected_object(self, clear_prompts=True) -> bool:
        return self.remove_object(self.get_select_idx(), clear_prompts=clear_prompts)

    def reset_to_single_object(self, init_mask_preds, init_mask_idx, clear_prompts=True):
        """Reset all object slots, tracking memory, and result buffers to a single fresh object."""

        self.init_mask_preds = init_mask_preds
        self.init_mask_idx = init_mask_idx
        self.maskresults_list.clear()
        self.memory_list.clear()
        self.stable_ids.clear()
        self.results_buffer.clear()
        self.last_record_signature = None
        self._clear_mask_cache()
        self._append_object_data()
        self._rebuild_ui(0)
        if clear_prompts:
            self.ui_elems.clear_prompts()
        return self

    def reset_to_stable_ids(self, stable_ids: list[int], init_mask_preds, init_mask_idx):
        """Replace every slot with the given label ids, in that order."""

        if len(stable_ids) == 0 or len(stable_ids) > self.MAX_OBJECTS:
            raise ValueError(f"Session needs between 1 and {self.MAX_OBJECTS} objects.")
        self.init_mask_preds = init_mask_preds
        self.init_mask_idx = init_mask_idx
        self.maskresults_list.clear()
        self.memory_list.clear()
        self.stable_ids.clear()
        self.results_buffer.clear()
        self.last_record_signature = None
        self._clear_mask_cache()
        for stable_id in stable_ids:
            self.stable_ids.append(int(stable_id))
            self.maskresults_list.append(MaskResults.create(self.init_mask_preds, self.init_mask_idx))
            self.memory_list.append(
                SAMVideoMemoryBank(self.max_memory_history, max_prompt_memory=self.max_prompt_memory)
            )
        self._rebuild_ui(0)
        self.ui_elems.clear_prompts()
        return self


# ---------------------------------------------------------------------------------------------------------------------
# %% Main


def main():
    # ---------------------------------------------------------------------------------------------------------------------
    # %% Set up script args

    # Set argparse defaults
    default_device = get_default_device_string()
    default_video_path = None
    default_model_path = None
    default_display_size = 900
    default_max_memory_history = 6
    default_object_score_threshold = 0.0
    default_encode_cache_size = 64
    default_reverse_buffer_size = 120

    # Define script arguments
    parser = argparse.ArgumentParser(
        description=f"{__app_name__} v{__version__} — interactive SAM video segmentation and tracking"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{__app_name__} {__version__}",
    )
    parser.add_argument("-i", "--video_path", default=default_video_path, help="Optional path to input video (otherwise select in GUI)")
    parser.add_argument("-m", "--model_path", default=default_model_path, type=str, help="Optional path to SAM model weights (otherwise use default from model_weights/)")
    parser.add_argument(
        "-s",
        "--display_size",
        default=None,
        type=int,
        help=(
            f"Controls size of displayed results (default: {default_display_size}). "
            "Omit to reuse the saved size; pass this flag to override it, including the default."
        ),
    )
    parser.add_argument(
        "-d",
        "--device",
        default=default_device,
        type=str,
        help=f"Device to use when running model, such as 'cpu' (default: {default_device})",
    )
    parser.add_argument(
        "-f32",
        "--use_float32",
        default=False,
        action="store_true",
        help="Use 32-bit floating point model weights. Note: this doubles VRAM usage",
    )
    parser.add_argument(
        "-ar",
        "--use_aspect_ratio",
        default=False,
        action="store_true",
        help="Process the video at its original aspect ratio. Omit to reuse the saved square/aspect choice.",
    )
    parser.add_argument(
        "--square",
        default=False,
        action="store_true",
        help="Stretch each frame to a square and override the saved aspect setting",
    )
    parser.add_argument(
        "-b",
        "--base_size_px",
        default=DEFAULT_ENCODE_SIDE,
        type=int,
        help=(
            f"Image encoder longest side (default: {DEFAULT_ENCODE_SIDE}). "
            "SAM 2 native is 1024; SAM 3 / 3.1 native is 1008."
        ),
    )
    parser.add_argument(
        "--max_memories",
        default=default_max_memory_history,
        type=int,
        help=f"Maximum number of previous-frame memory encodings to store (default: {default_max_memory_history})",
    )
    parser.add_argument(
        "--keep_bad_objscores",
        default=False,
        action="store_true",
        help=(
            "If set, keep running video masking after a low object-score (lost target); "
            "masks are still zeroed on low-score frames. Default: stop inference from the "
            "first low-score frame onward until the playhead moves before that frame or "
            "new prompts are stored"
        ),
    )
    parser.add_argument(
        "--keep_history_on_new_prompts",
        default=False,
        action="store_true",
        help="If set, existing history data will not be cleared when adding new prompts",
    )
    parser.add_argument(
        "--objscore_threshold",
        default=None,
        type=float,
        help=(
            f"Threshold below which objects are considered to be 'lost' (default: {default_object_score_threshold}). "
            "Omit to reuse the saved value; pass this flag to override it, including 0."
        ),
    )
    parser.add_argument(
        "--encode_cache_size",
        default=default_encode_cache_size,
        type=int,
        help=(
            "Number of recent frame image-encodings to cache on CPU to avoid recomputation when "
            f"scrubbing/stepping/reversing (0 disables, default: {default_encode_cache_size})"
        ),
    )
    parser.add_argument(
        "--mask_select",
        default="legacy",
        choices=("legacy", "official"),
        help="legacy keeps argmax over every mask. official uses SAM's tracking mask rule.",
    )
    parser.add_argument(
        "--lost_patience",
        default=1,
        type=int,
        help="Consecutive low-score frames before a target is lost (1 keeps the historical behavior).",
    )
    parser.add_argument(
        "--max_prompt_attn",
        default=None,
        type=int,
        help="For SAM 3 and 3.1, condition on the first prompt plus the closest others, up to this count.",
    )
    parser.add_argument(
        "--intensity_range",
        default="auto",
        help="16-bit stills: auto (percentile), full (0-65535), or low,high.",
    )
    parser.add_argument(
        "--encode_cache_mb",
        default=2048,
        type=int,
        help="Extra cap on the CPU encode cache, in megabytes (0 disables the byte cap).",
    )
    parser.add_argument(
        "--reverse_buffer_size",
        default=default_reverse_buffer_size,
        type=int,
        help=(
            "Number of recently decoded video frames to buffer in memory for smooth reverse playback "
            f"(0 disables, default: {default_reverse_buffer_size})"
        ),
    )
    # For convenience
    args = parser.parse_args()
    if args.square and args.use_aspect_ratio:
        parser.error("Use only one of --square and --use_aspect_ratio")
    arg_video_path = args.video_path
    arg_model_path = args.model_path
    display_size_px = args.display_size
    device_str = args.device
    use_float32 = args.use_float32
    use_square_sizing = not args.use_aspect_ratio
    active_base_size = resolve_encode_side(args.base_size_px)
    imgenc_base_size = active_base_size
    max_memory_history = args.max_memories
    keep_tracking_after_loss = args.keep_bad_objscores
    clear_history_on_new_prompts = not args.keep_history_on_new_prompts
    object_score_threshold = args.objscore_threshold
    encode_cache_size = max(0, args.encode_cache_size)
    reverse_buffer_size = max(0, args.reverse_buffer_size)
    mask_select_mode = args.mask_select
    lost_patience = max(1, args.lost_patience)
    max_prompt_attn = args.max_prompt_attn
    encode_cache_mb = max(0, args.encode_cache_mb)
    try:
        intensity_range = parse_intensity_range(args.intensity_range)
    except ValueError as err:
        parser.error(str(err))


    def print_startup_banner() -> None:
        """Print project title and author to the terminal before loading resources."""

        line = "=" * 56
        print("", line, sep="\n", flush=True)
        print(f"  {__app_name__}  v{__version__}", flush=True)
        print(f"  by {__author__}  <{__author_email__}>", flush=True)
        print(line, "", sep="\n", flush=True)


    # Set up device config
    device_config_dict = make_device_config(device_str, use_float32)
    compute_device = device_config_dict["device"]

    # Cache for frame image-encodings (stored on CPU to conserve VRAM)
    encode_cache = EncodedImageCache(
        max_items=encode_cache_size,
        store_device="cpu",
        max_bytes=None if encode_cache_mb == 0 else encode_cache_mb * 1024 * 1024,
    )

    # History lives beside this file, not in whatever directory the app was launched from.
    history = HistoryKeeper(__file__)
    _, history_modelpath = history.read("model_path")

    # Restore persisted session settings only for options omitted on the command line.
    # An explicit value, including the built-in default, overrides history.
    history_display = history.read("display_size_px")[1]
    history_objscore = history.read("objscore_threshold")[1]
    history_square = history.read("use_square_sizing")[1]
    try:
        display_size_px, object_score_threshold, use_square_sizing = resolve_startup_settings(
            args.display_size,
            args.objscore_threshold,
            args.use_aspect_ratio,
            args.square,
            default_display_size,
            default_object_score_threshold,
            history_display if isinstance(history_display, int) else None,
            history_objscore if isinstance(history_objscore, (int, float)) else None,
            history_square if isinstance(history_square, bool) else None,
        )
    except ValueError as err:
        parser.error(str(err))
    has_val, stored_history_enabled = history.read("enable_history")
    history_enabled_default = stored_history_enabled if isinstance(stored_history_enabled, bool) else True

    # Persist the resolved settings so the next launch reuses them
    history.store(
        display_size_px=int(display_size_px),
        objscore_threshold=float(object_score_threshold),
        use_square_sizing=bool(use_square_sizing),
        enable_history=bool(history_enabled_default),
    )

    # Resolve model candidates without terminal prompts; video is selected from the GUI unless -i is used.
    # An explicit -m that matches nothing is an error. A history path is only remembered after it loads.
    try:
        model_candidates = resolve_model_candidates(__file__, arg_model_path, history_modelpath)
    except FileNotFoundError as err:
        print("", str(err), sep="\n", flush=True)
        sys.exit(1)

    if arg_video_path and osp.exists(clean_path_str(arg_video_path)):
        video_path = clean_path_str(arg_video_path)
        has_video_source = True
    elif arg_video_path:
        print("", f"Warning: input not found, starting without it: {arg_video_path}", sep="\n", flush=True)
        video_path = None
        has_video_source = False
    else:
        video_path = None
        has_video_source = False

    if has_video_source:
        history.store(video_path=video_path)


    # ---------------------------------------------------------------------------------------------------------------------
    # %% Load resources

    # Set up shared image encoder settings (needs to be consistent across image/video frame encodings)
    imgenc_config_dict = {"max_side_length": imgenc_base_size, "use_square_sizing": use_square_sizing}


    # ---------------------------------------------------------------------------------------------------------------------
    # %% Load model & optional video source

    video_name = osp.basename(video_path) if has_video_source else NO_VIDEO_LABEL
    placeholder_frame = create_placeholder_frame()

    print_startup_banner()
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print(
            "",
            "Warning: tkinter is required for dialogs, saving, and the user guide.",
            "  Install it with your Python distribution before recording results.",
            sep="\n",
            flush=True,
        )
    sam_core = interact_model = track_model = None
    model_path = None
    for candidate in model_candidates:
        try:
            sam_core, interact_model, track_model = load_sam_models(candidate, device_config_dict)
            model_path = candidate
            break
        except Exception as err:
            print("", f"Could not load model weights: {candidate}", f"  {err}", sep="\n", flush=True)
            sam_core = interact_model = track_model = None
            release_sam_runtime()
            if arg_model_path:
                sys.exit(1)
    if model_path is None:
        print("", "No SAM model weights could be loaded.", sep="\n", flush=True)
        sys.exit(1)
    history.store(model_path=model_path)
    model_name = osp.basename(model_path)
    model_family = model_family_name(sam_core)
    print(f"  Model family: {model_family}", flush=True)
    apply_encode_side(imgenc_config_dict, model_family, active_base_size)

    vreader = None
    video_fps = 30.0
    if has_video_source:
        vreader = open_frame_source(video_path, intensity_range=intensity_range).release()
        vreader.set_frame_buffer_size(reverse_buffer_size)
        video_fps = vreader.get_fps()
        sample_frame = vreader.get_sample_frame()
    else:
        sample_frame = placeholder_frame.copy()

    encoded_img, init_mask_preds, iou_preds, init_mask_idx, preencode_hw, token_hw = run_initial_model_pass(
        interact_model, sample_frame, imgenc_config_dict
    )
    encoded_img = strip_detector_branch(encoded_img, model_family)
    print_model_config(
        model_name, device_config_dict, preencode_hw, token_hw, imgenc_config_dict["max_side_length"]
    )


    # ---------------------------------------------------------------------------------------------------------------------
    # %% Set up UI

    # Playback control UI for adjusting video position
    playback_slider = LoopingVideoPlaybackSlider(vreader, stay_paused_on_change=True) if has_video_source else None

    # Set up shared UI elements & control logic
    ui_elems = PromptUI(sample_frame, 2)
    uictrl = PromptUIControl(ui_elems)

    # Transient on-screen feedback + global prompt undo stack
    toast = ToastManager()
    undo_mgr = PromptUndoManager(ui_elems)

    # Add extra polygon drawer for unselected objects
    unselected_olay = DrawPolygonsOverlay((50, 5, 130), bg_color=None)
    ui_elems.overlay_img.add_overlays(unselected_olay)
    middle_pick_olay = MiddleClickCaptureOverlay()
    middle_pick_olay.enable(True)
    ui_elems.overlay_img.add_overlays(middle_pick_olay)

    # Set up text-based reporting UI
    # Bottom status bar sizing (+1/3 vs original 20px / 0.30 scale)
    status_bar_height = 27
    status_bar_text_scale = 0.40

    vram_text = ValueBlock(
        "VRAM: ", "-", "MB", block_height=status_bar_height, text_scale=status_bar_text_scale, max_characters=5
    )
    device_text = ValueBlock(
        "Device: ",
        format_device_label(device_config_dict),
        "",
        block_height=status_bar_height,
        text_scale=status_bar_text_scale,
        max_characters=16,
    )
    shortcuts_btn = ImmediateButton(
        "Shortcuts (F1)", text_scale=0.43, color=(95, 105, 128), button_height=status_bar_height
    )
    num_prompts_text = ValueBlock("Prompts: ", "0", max_characters=2)
    num_history_text = ValueBlock("History: ", "0", max_characters=2)
    force_same_min_width(vram_text, device_text, shortcuts_btn)

    # Set up button controls
    track_btn = ToggleButton("Track", on_color=(30, 140, 30))
    reversal_btn = ToggleButton("Reverse", default_state=False, text_scale=0.35)
    store_prompt_btn = ImmediateButton("Store Prompt (Enter)", text_scale=0.35, color=(145, 160, 40))
    clear_prompts_btn = ImmediateButton("Clear Prompts", text_scale=0.35, color=(80, 110, 230))
    enable_history_btn = ToggleButton(
        "Enable History", default_state=history_enabled_default, text_scale=0.35, on_color=(90, 85, 115)
    )
    clear_history_btn = ImmediateButton("Clear History", text_scale=0.35, color=(130, 60, 90))
    force_same_min_width(store_prompt_btn, clear_prompts_btn, enable_history_btn, clear_history_btn)


    # Create save UI (object rows are managed dynamically by ObjectSlotManager)
    enable_record_btn = ToggleButton("Enable Recording", default_state=False, on_color=(0, 15, 255), button_height=60)
    buffer_save_btn = ImmediateButton("Save Results", button_height=30, text_scale=0.5, color=(110, 145, 65))
    retry_metrics_btn = ImmediateButton("Retry Metrics", button_height=30, text_scale=0.45, color=(90, 125, 145))
    buffer_clear_btn = ImmediateButton("Clear Results", button_height=30, text_scale=0.5, color=(80, 60, 190))
    add_object_btn = ImmediateButton("Add Object", button_height=30, text_scale=0.5, color=(90, 130, 90))
    remove_object_btn = ImmediateButton("Remove Object", button_height=30, text_scale=0.5, color=(130, 70, 70))
    force_same_min_width(buffer_save_btn, retry_metrics_btn, buffer_clear_btn, add_object_btn, remove_object_btn)

    # Set up resource switcher bar (model / video)
    model_btn = ImmediateButton(
        format_resource_button_label("Model", model_name), text_scale=0.4, color=(90, 110, 145)
    )
    video_btn = ImmediateButton(
        format_resource_button_label("Video", video_name), text_scale=0.4, color=(110, 95, 75)
    )
    force_same_min_width(model_btn, video_btn)
    header_bar = HStack(model_btn, video_btn).set_debug_name("HeaderResourceBar")
    save_session_btn = ImmediateButton("Save Session", button_height=30, text_scale=0.45, color=(95, 110, 140))
    load_session_btn = ImmediateButton("Load Session", button_height=30, text_scale=0.45, color=(95, 110, 140))
    force_same_min_width(save_session_btn, load_session_btn)


    def build_disp_layout(save_sidebar):
        return VStack(
            header_bar,
            HStack(ui_elems.layout, save_sidebar),
            playback_slider if has_video_source else None,
            HStack(num_prompts_text, track_btn, num_history_text),
            HStack(store_prompt_btn, clear_prompts_btn, reversal_btn, enable_history_btn, clear_history_btn),
            HStack(save_session_btn, load_session_btn),
            HStack(vram_text, device_text, shortcuts_btn),
        ).set_debug_name("DisplayLayout")


    obj_mgr = ObjectSlotManager(
        init_mask_preds,
        init_mask_idx,
        max_memory_history,
        enable_record_btn,
        buffer_save_btn,
        retry_metrics_btn,
        buffer_clear_btn,
        add_object_btn,
        remove_object_btn,
        ui_elems,
        build_disp_layout,
    )
    disp_layout = obj_mgr.disp_layout

    # Render out an image with a target size, to figure out which side we should limit when rendering
    display_image = disp_layout.render(h=display_size_px, w=display_size_px)
    obj_mgr.finalize_layout_callback_regions()
    render_side = "h" if display_image.shape[1] > display_image.shape[0] else "w"
    render_limit_dict = {render_side: display_size_px}
    min_display_size_px = disp_layout._rdr.limits.min_h if render_side == "h" else disp_layout._rdr.limits.min_w


    # ---------------------------------------------------------------------------------------------------------------------
    # %% Video loop

    # Setup display window
    window = DisplayWindow(f"{__app_name__}  by {__author__}", display_fps=60)
    obj_mgr.bind_window(window)
    shortcuts_help = ShortcutsHelpWindow(offset_xy=(60, 60), version=__version__)
    shortcuts_help.attach_f1_toggle(window)

    user_guide = UserGuideWindow(__app_name__, __version__, __author__, __author_email__, initial_lang="zh")
    user_guide.attach_h_toggle(window)

    # Change tools on Tab / Shift+Tab; change objects on up/down arrow keys.
    # The pause callback reads the current reader, so it stays valid after a video switch.
    uictrl.attach_arrowkey_callbacks(window)

    def _toggle_pause():
        if vreader is not None:
            vreader.toggle_pause()

    window.attach_keypress_callback(" ", _toggle_pause)
    window.attach_arrow_keypress_callback("up", obj_mgr.previous_object)
    window.attach_arrow_keypress_callback("down", obj_mgr.next_object)
    window.attach_keypress_callback("w", obj_mgr.previous_object)
    window.attach_keypress_callback("s", obj_mgr.next_object)
    window.attach_keypress_callback("+", add_object_btn.click)
    window.attach_keypress_callback("=", add_object_btn.click)
    window.attach_keypress_callback("_", remove_object_btn.click)


    def _store_prompt_if_paused():
        if has_video_source and not vreader.get_pause_state():
            return
        store_prompt_btn.click()


    window.attach_keypress_callback(KEY.ENTER, _store_prompt_if_paused)
    window.attach_keypress_callback("r", reversal_btn.toggle)


    def _undo_last_prompt():
        if undo_mgr.undo():
            toast.notify("Undo prompt", level="info", duration_sec=1.2)


    # Ctrl+Z removes the most recently added FG/BG point or box
    window.attach_keypress_callback(26, _undo_last_prompt)

    # For clarity, some additional keypress codes
    KEY_ZOOM_IN = ord("]")
    KEY_ZOOM_OUT = ord("[")
    KEY_STEP_BACK_KEYS = frozenset({ord("a"), KEY.LEFT_ARROW})
    KEY_STEP_FWD_KEYS = frozenset({ord("d"), KEY.RIGHT_ARROW})

    # Set up various value tracking helpers
    imgenc_idx_keeper = ValueChangeTracker(-1)
    track_idx_keeper = ValueChangeTracker(-1)
    pause_keeper = ValueChangeTracker(vreader.get_pause_state() if has_video_source else True)
    vram_report = PeriodicVRAMReport(update_period_ms=2000)

    # Helper used to define/keep track of all states of the UI
    STATES = Enum(
        "state", ["ADJUST_PLAYBACK", "SWITCH_PAUSE_ON", "SWITCH_PAUSE_OFF", "TRACKING", "PAUSED", "NO_TRANSITION"]
    )

    # Set up per-object storage for masking/saving results (managed by obj_mgr)

    if has_video_source:
        vreader.pause()
    curr_state = STATES.PAUSED
    tran_state = STATES.NO_TRANSITION
    video_iter = iter(vreader) if has_video_source else None
    _, _, prev_selected_tool = ui_elems.tools_constraint.read()
    was_playback_adjusting = False
    hover_preview_suppressed = False
    clean_exit = False
    consecutive_errors = 0

    def _restore_prompt_session(session: dict) -> None:
        """Rebuild object slots from raw prompts and encode them with the saved settings."""

        nonlocal model_path, model_name, model_family
        nonlocal sam_core, interact_model, track_model
        nonlocal active_base_size, encoded_img, init_mask_preds, init_mask_idx

        objects = session.get("objects") or []
        if len(objects) == 0:
            raise ValueError("Session file has no prompts.")
        if len(objects) > obj_mgr.MAX_OBJECTS:
            raise ValueError(f"Session has {len(objects)} objects; the limit is {obj_mgr.MAX_OBJECTS}.")
        object_ids = session_object_ids(objects)

        runtime_changed = False
        saved_model = clean_path_str(session.get("model_path"))
        if saved_model:
            if not osp.isfile(saved_model):
                raise FileNotFoundError(f"Session model weights were not found: {saved_model}")
            current_model = osp.normcase(osp.abspath(str(model_path)))
            if osp.normcase(osp.abspath(saved_model)) != current_model:
                new_core, new_interact, new_track = load_sam_models(saved_model, device_config_dict)
                encode_cache.clear()
                sam_core = interact_model = track_model = None
                release_sam_runtime()
                sam_core, interact_model, track_model = new_core, new_interact, new_track
                model_path = saved_model
                model_name = osp.basename(model_path)
                model_family = model_family_name(sam_core)
                history.store(model_path=model_path)
                model_btn.set_label(format_resource_button_label("Model", model_name))
                runtime_changed = True

        saved_side = session.get("encode_side")
        if saved_side is not None and int(saved_side) != int(active_base_size):
            active_base_size = int(saved_side)
            runtime_changed = True
        saved_square = session.get("use_square_sizing")
        if saved_square is not None and bool(saved_square) != bool(imgenc_config_dict["use_square_sizing"]):
            imgenc_config_dict["use_square_sizing"] = bool(saved_square)
            runtime_changed = True
        if runtime_changed:
            apply_encode_side(imgenc_config_dict, model_family, active_base_size)
            encode_cache.clear()
            sample = vreader.get_sample_frame() if vreader is not None else placeholder_frame.copy()
            encoded_img, init_mask_preds, _iou_preds, init_mask_idx, preencode_hw, token_hw = run_initial_model_pass(
                interact_model, sample, imgenc_config_dict
            )
            encoded_img = strip_detector_branch(encoded_img, model_family)
            print_model_config(
                model_name,
                device_config_dict,
                preencode_hw,
                token_hw,
                imgenc_config_dict["max_side_length"],
            )

        vreader.pause(True)
        obj_mgr.reset_to_stable_ids(object_ids, init_mask_preds, init_mask_idx)
        for objidx, obj in enumerate(objects):
            for prompt in obj.get("prompts") or []:
                frame_index = int(prompt["frame_index"])
                vreader.set_playback_position(frame_index)
                frame = vreader.get_current_frame()
                encoded = strip_detector_branch(
                    interact_model.encode_image(frame, **imgenc_config_dict), model_family
                )
                boxes = prompt.get("boxes") or []
                fg_points = prompt.get("fg_points") or []
                bg_points = prompt.get("bg_points") or []
                _best_mask, memory_encoding = track_model.encode_prompt_memory(
                    encoded, boxes, fg_points, bg_points, mask_index=None
                )
                obj_mgr.memory_list[objidx].store_prompt_result(
                    memory_encoding,
                    frame_index,
                    {"boxes": boxes, "fg_points": fg_points, "bg_points": bg_points},
                )
                obj_mgr.memory_list[objidx].clear_tracking_stop_frame()
        track_idx_keeper.clear()
        obj_mgr.last_record_signature = None

    try:

        while True:
            try:
                if has_video_source:
                    is_paused, frame_idx, frame = next(video_iter)
                else:
                    is_paused, frame_idx, frame = True, 0, placeholder_frame.copy()

                pick_changed, pick_xy_norm = middle_pick_olay.read()
                if pick_changed and pick_xy_norm is not None and has_video_source:
                    hit_idx = find_object_index_at_mask_xy(
                        obj_mgr, obj_mgr.get_select_idx(), pick_xy_norm, uictrl, frame.shape[0:2]
                    )
                    if hit_idx is not None:
                        obj_mgr.select_object(hit_idx)

                if model_btn.read():
                    picked_model_path = pick_model_file(__file__, model_path)
                    window.refocus()
                    if picked_model_path and picked_model_path != model_path:
                        if not confirm_before_discarding_results(
                            "switch models",
                            obj_mgr,
                            video_path,
                            window,
                            history,
                            toast,
                            video_fps,
                            getattr(vreader, "source_info", None) if vreader is not None else None,
                            intensity_range,
                        ):
                            continue
                        try:
                            new_core, new_interact, new_track = load_sam_models(picked_model_path, device_config_dict)
                        except Exception as err:
                            show_message_dialog(
                                "Model Load Failed",
                                f"The current model is still loaded.\n\n{err}",
                                kind="error",
                            )
                            window.refocus()
                            continue
                        encode_cache.clear()
                        encoded_img = None
                        sam_core = interact_model = track_model = None
                        release_sam_runtime()
                        sam_core, interact_model, track_model = new_core, new_interact, new_track
                        model_path = picked_model_path
                        model_name = osp.basename(model_path)
                        model_family = model_family_name(sam_core)
                        apply_encode_side(imgenc_config_dict, model_family, active_base_size)
                        history.store(model_path=model_path)
                        if has_video_source:
                            history.store(video_path=video_path)
                        model_btn.set_label(format_resource_button_label("Model", model_name))
                        toast.notify(f"Model loaded: {model_name}", level="success")

                        if has_video_source:
                            vreader.set_playback_position(0)
                            sample_frame = vreader.get_sample_frame()
                        else:
                            sample_frame = placeholder_frame.copy()
                        ui_elems.image.set_image(sample_frame)
                        encoded_img, init_mask_preds, iou_preds, init_mask_idx, preencode_hw, token_hw = run_initial_model_pass(
                            interact_model, sample_frame, imgenc_config_dict
                        )
                        encoded_img = strip_detector_branch(encoded_img, model_family)
                        print_model_config(
                            model_name,
                            device_config_dict,
                            preencode_hw,
                            token_hw,
                            imgenc_config_dict["max_side_length"],
                        )

                        clear_tracking_ui_state(
                            uictrl,
                            ui_elems,
                            unselected_olay,
                            obj_mgr,
                            init_mask_preds,
                            init_mask_idx,
                            track_btn,
                            enable_record_btn,
                            reversal_btn,
                            imgenc_idx_keeper,
                            track_idx_keeper,
                            num_prompts_text,
                            num_history_text,
                            vreader,
                        )
                        if has_video_source and vreader is not None:
                            vreader.pause()
                        pause_keeper.record(True)
                        curr_state = STATES.PAUSED
                        continue

                if video_btn.read():
                    picked_video_path = pick_frame_source(video_path)
                    window.refocus()
                    if picked_video_path and (picked_video_path != video_path):
                        if not confirm_before_discarding_results(
                            "open another video",
                            obj_mgr,
                            video_path,
                            window,
                            history,
                            toast,
                            video_fps,
                            getattr(vreader, "source_info", None) if vreader is not None else None,
                            intensity_range,
                        ):
                            continue
                        try:
                            new_reader = open_frame_source(picked_video_path, intensity_range=intensity_range).release()
                            new_reader.set_frame_buffer_size(reverse_buffer_size)
                            new_sample = new_reader.get_sample_frame()
                        except Exception as err:
                            show_message_dialog(
                                "Video Load Failed",
                                f"The current video is still loaded.\n\n{err}",
                                kind="error",
                            )
                            window.refocus()
                            continue
                        if vreader is not None:
                            vreader.release()
                        encode_cache.clear()
                        encoded_img = None

                        vreader = new_reader
                        video_path = picked_video_path
                        video_name = osp.basename(video_path)
                        history.store(video_path=video_path, model_path=model_path)
                        video_fps = vreader.get_fps()
                        sample_frame = new_sample
                        first_video_load = not has_video_source
                        has_video_source = True

                        if playback_slider is None:
                            playback_slider = LoopingVideoPlaybackSlider(vreader, stay_paused_on_change=True)
                        else:
                            playback_slider.replace_video_reader(vreader)

                        video_iter = iter(vreader)
                        video_btn.set_label(format_resource_button_label("Video", video_name))
                        toast.notify(f"Video loaded: {video_name}", level="success")

                        if first_video_load:
                            obj_mgr._rebuild_ui(obj_mgr.get_select_idx())

                        ui_elems.image.set_image(sample_frame)
                        encoded_img, init_mask_preds, iou_preds, init_mask_idx, preencode_hw, token_hw = run_initial_model_pass(
                            interact_model, sample_frame, imgenc_config_dict
                        )
                        encoded_img = strip_detector_branch(encoded_img, model_family)
                        print_model_config(
                            model_name,
                            device_config_dict,
                            preencode_hw,
                            token_hw,
                            imgenc_config_dict["max_side_length"],
                        )

                        clear_tracking_ui_state(
                            uictrl,
                            ui_elems,
                            unselected_olay,
                            obj_mgr,
                            init_mask_preds,
                            init_mask_idx,
                            track_btn,
                            enable_record_btn,
                            reversal_btn,
                            imgenc_idx_keeper,
                            track_idx_keeper,
                            num_prompts_text,
                            num_history_text,
                            vreader,
                        )
                        vreader.pause()
                        pause_keeper.record(True)
                        curr_state = STATES.PAUSED
                        continue

                # Change playback direction, if needed
                is_reversed_changed, reverse_video = reversal_btn.read()
                frame_direction = playback_direction(reverse_video)
                if has_video_source and is_reversed_changed:
                    vreader.toggle_reverse_state(reverse_video)

                # Read controls
                is_changed_pause_state = pause_keeper.is_changed(is_paused)
                is_history_toggle_changed, is_trackhistory_enabled = enable_history_btn.read()
                if is_history_toggle_changed:
                    history.store(enable_history=bool(is_trackhistory_enabled))
                    toast.notify(
                        f"History {'enabled' if is_trackhistory_enabled else 'disabled'}", level="info", duration_sec=1.5
                    )

                is_changed_buffer, buffer_select_idx, _ = obj_mgr.read_selection()
                if is_changed_buffer:
                    obj_mgr.ensure_active_object_visible()
                is_changed_tool, _, selected_tool = ui_elems.tools_constraint.read()
                selected_has_no_stored_prompts = not obj_mgr.memory_list[buffer_select_idx].check_has_prompts()
                if is_changed_tool:
                    if prev_selected_tool is ui_elems.tools.hover and selected_tool is not ui_elems.tools.hover:
                        clear_hover_preview_mask(obj_mgr, buffer_select_idx, track_idx_keeper)
                    elif selected_has_no_stored_prompts:
                        obj_mgr.maskresults_list[buffer_select_idx].clear()
                    if selected_has_no_stored_prompts:
                        hover_preview_suppressed = True

                is_changed_track_idx = track_idx_keeper.is_changed(frame_idx)
                prev_selected_tool = selected_tool

                if is_changed_buffer:
                    ui_elems.clear_prompts(keep_hover_position=False)
                    clear_hover_preview_mask(obj_mgr, buffer_select_idx, track_idx_keeper)
                    hover_preview_suppressed = True
                    for objidx in obj_mgr.objiter:
                        if not obj_mgr.memory_list[objidx].check_has_prompts():
                            obj_mgr.maskresults_list[objidx].clear()
                    track_idx_keeper.clear()

                # Allow the track button to play/pause the video
                is_trackstate_changed, is_track_on = track_btn.read()
                if has_video_source and is_trackstate_changed:
                    vreader.pause(not is_track_on)

                # Wipe out buffered data
                if clear_prompts_btn.read():
                    obj_mgr.memory_list[buffer_select_idx].clear(clear_frame_memory=False)
                    obj_mgr.maskresults_list[buffer_select_idx].clear()
                    track_idx_keeper.clear()
                    toast.notify(f"Prompts cleared (Object {obj_mgr.stable_ids[buffer_select_idx]})", level="info", duration_sec=1.5)
                if clear_history_btn.read():
                    obj_mgr.memory_list[buffer_select_idx].clear(clear_prompt_memory=False)
                    track_idx_keeper.clear()
                    toast.notify(f"History cleared (Object {obj_mgr.stable_ids[buffer_select_idx]})", level="info", duration_sec=1.5)

                if add_object_btn.read():
                    if obj_mgr.add_object():
                        toast.notify(
                            f"Added Object {obj_mgr.stable_ids[obj_mgr.get_select_idx()]}",
                            level="success",
                            duration_sec=1.5,
                        )
                        layout_image = obj_mgr.disp_layout.render(h=display_size_px, w=display_size_px)
                        obj_mgr.finalize_layout_callback_regions()
                        render_side = "h" if layout_image.shape[1] > layout_image.shape[0] else "w"
                        render_limit_dict = {render_side: display_size_px}
                        min_display_size_px = (
                            obj_mgr.disp_layout._rdr.limits.min_h if render_side == "h" else obj_mgr.disp_layout._rdr.limits.min_w
                        )
                        buffer_select_idx = obj_mgr.get_select_idx()
                        for objidx in obj_mgr.objiter:
                            if objidx != buffer_select_idx and not obj_mgr.memory_list[objidx].check_has_prompts():
                                obj_mgr.maskresults_list[objidx].clear()

                if remove_object_btn.read():
                    remove_idx = obj_mgr.get_select_idx()
                    has_prompts = obj_mgr.memory_list[remove_idx].check_has_prompts()
                    stable_id = obj_mgr.stable_ids[remove_idx]
                    has_labels = obj_mgr.results_buffer.contains_label(stable_id)
                    if has_prompts or has_labels:
                        confirmed = ask_yes_no(
                            "Remove Object",
                            f"Remove Object {stable_id}? Its prompts and recorded labels will be deleted.",
                        )
                        window.refocus()
                        if confirmed is not True:
                            continue
                    if obj_mgr.remove_selected_object():
                        layout_image = obj_mgr.disp_layout.render(h=display_size_px, w=display_size_px)
                        obj_mgr.finalize_layout_callback_regions()
                        render_side = "h" if layout_image.shape[1] > layout_image.shape[0] else "w"
                        render_limit_dict = {render_side: display_size_px}
                        min_display_size_px = (
                            obj_mgr.disp_layout._rdr.limits.min_h if render_side == "h" else obj_mgr.disp_layout._rdr.limits.min_w
                        )
                        buffer_select_idx = obj_mgr.get_select_idx()
                        toast.notify("Object removed", level="info", duration_sec=1.5)

                if shortcuts_btn.read():
                    shortcuts_help.toggle()
                    window.refocus()

                # Update text feedback
                vram_usage_mb = vram_report.get_vram_usage()
                vram_text.set_value(vram_usage_mb)
                num_prompt_mems, num_frame_mems = obj_mgr.memory_list[buffer_select_idx].get_num_memories()
                num_prompts_text.set_value(num_prompt_mems)
                num_history_text.set_value(num_frame_mems)

                # Ugly: Figure out current states
                is_playback_adjusting = playback_slider.is_adjusting() if playback_slider is not None else False
                scrub_just_released = was_playback_adjusting and not is_playback_adjusting and has_video_source
                was_playback_adjusting = is_playback_adjusting
                have_any_stored_prompts = any(mem.check_has_prompts() for mem in obj_mgr.memory_list)
                if is_playback_adjusting:
                    curr_state = STATES.ADJUST_PLAYBACK
                elif is_paused:
                    curr_state = STATES.PAUSED
                else:
                    curr_state = STATES.TRACKING

                # Handle transition states (mostly need to account for playback slider!)
                if is_playback_adjusting:
                    tran_state = STATES.ADJUST_PLAYBACK
                elif is_changed_pause_state and is_paused:
                    tran_state = STATES.SWITCH_PAUSE_ON
                elif is_changed_pause_state and not is_paused:
                    tran_state = STATES.SWITCH_PAUSE_OFF
                else:
                    tran_state = STATES.NO_TRANSITION

                # Encode any 'new' frames as needed (but not while the playback slider is being dragged)
                need_image_encode = has_video_source and imgenc_idx_keeper.is_changed(frame_idx)
                should_encode_frame = (need_image_encode and not is_playback_adjusting) or (
                    scrub_just_released and have_any_stored_prompts
                )
                if should_encode_frame:
                    cached_encoding = encode_cache.get(frame_idx, compute_device)
                    if cached_encoding is not None:
                        encoded_img = cached_encoding
                    else:
                        encoded_img = strip_detector_branch(
                            interact_model.encode_image(frame, **imgenc_config_dict), model_family
                        )
                        encode_cache.store(frame_idx, encoded_img)
                    imgenc_idx_keeper.record(frame_idx)

                if scrub_just_released and have_any_stored_prompts:
                    lost_objects = []
                    run_multi_object_tracking(
                        obj_mgr,
                        track_model,
                        encoded_img,
                        frame_idx,
                        object_score_threshold,
                        keep_tracking_after_loss,
                        is_trackhistory_enabled,
                        lost_objects,
                        frame_direction,
                        mask_select_mode,
                        model_family,
                        lost_patience,
                        max_prompt_attn,
                    )
                    notify_lost_objects(toast, obj_mgr, lost_objects)
                    track_idx_keeper.record(frame_idx)

                # Wipe out masking/contours when jumping around playback (otherwise stays over top of changing video!)
                if is_playback_adjusting:
                    for maskresult in obj_mgr.maskresults_list:
                        maskresult.clear()
                    obj_mgr.exclusive_masks.clear()
                    obj_mgr.mask_before_exclusion.clear()
                    obj_mgr.exclusive_hw = None

                    ui_elems.clear_prompts()
                    if has_video_source:
                        vreader.pause()
                    track_btn.toggle(False, flag_if_changed=False)

                # Universal updates whenever the pause state changes
                if is_changed_pause_state:

                    # Consume user prompt inputs (if we don't do this, inputs queued up during playback can appear!)
                    uictrl.read_prompts()
                    ui_elems.clear_prompts()

                    # Enable/disable prompt UI when playing/pausing
                    ui_elems.enable_tools(is_paused)
                    pause_keeper.record(is_paused)

                # Handle transistion states
                if tran_state == STATES.SWITCH_PAUSE_ON:

                    # Make sure track button is disabled to indicate pause state
                    track_btn.toggle(False, flag_if_changed=False)

                    # For QoL, if user was on FG/BG point, switch back to hover (more intuitive to work with)
                    _, _, selected_tool = ui_elems.tools_constraint.read()
                    need_hover_switch = selected_tool in (ui_elems.tools.fgpt, ui_elems.tools.bgpt)
                    if need_hover_switch:
                        ui_elems.tools_constraint.change_to(ui_elems.tools.hover)

                    # Wipe out segmentation data and any UI interactions that may have queued
                    uictrl.read_prompts()
                    ui_elems.clear_prompts()
                    hover_preview_suppressed = True

                elif tran_state == STATES.SWITCH_PAUSE_OFF:

                    # Make sure track button is enabled to indicate active playback/tracking
                    track_btn.toggle(True, flag_if_changed=False)

                    # If there is no tracking data, clear any on-screen masking (i.e. from user interactions)
                    no_prompt_data = all(mem.check_has_prompts() == 0 for mem in obj_mgr.memory_list)
                    if no_prompt_data:
                        for maskresult in obj_mgr.maskresults_list:
                            maskresult.clear()

                # Handle main steady states (paused or tracking)
                if curr_state == STATES.PAUSED:

                    # Initialize storage for predictions(which may not occur
                    paused_mask_preds = None
                    paused_obj_score = None

                    need_prompt_encode, prompts, hover_input_changed = uictrl.read_prompts()
                    have_user_prompts = check_have_prompts(*prompts)
                    have_track_prompts = any(mem.check_has_prompts() for mem in obj_mgr.memory_list)
                    selected_has_track_prompts = obj_mgr.memory_list[buffer_select_idx].check_has_prompts()
                    _, _, selected_tool = ui_elems.tools_constraint.read()
                    is_hover_tool = selected_tool == ui_elems.tools.hover
                    hover_preview_allowed = is_hover_tool and not selected_has_track_prompts
                    # Hover points are ignored for masking when the selected object already has saved prompts
                    interactive_user_prompts = have_user_prompts and not (is_hover_tool and selected_has_track_prompts)

                    if hover_preview_suppressed:
                        if hover_input_changed:
                            hover_preview_suppressed = False
                        elif hover_preview_allowed:
                            obj_mgr.maskresults_list[buffer_select_idx].clear()

                    can_run_hover_preview = hover_preview_allowed and have_user_prompts and not hover_preview_suppressed

                    # Drop stale hover previews when using non-hover tools on a fresh object
                    if not is_hover_tool and selected_has_no_stored_prompts and not have_user_prompts:
                        obj_mgr.maskresults_list[buffer_select_idx].clear()

                    # Hover preview lifecycle (only when the selected object has no saved prompts)
                    if hover_preview_allowed and not have_user_prompts:
                        obj_mgr.maskresults_list[buffer_select_idx].clear()

                    if need_prompt_encode:
                        if can_run_hover_preview:
                            encoded_prompts = interact_model.encode_prompts(*prompts)
                            paused_mask_preds, iou_preds = interact_model.generate_masks(
                                encoded_img,
                                encoded_prompts,
                                mask_hint=None,
                                blank_promptless_output=True,
                            )
                            track_idx_keeper.clear()
                        elif not is_hover_tool and have_user_prompts:
                            encoded_prompts = interact_model.encode_prompts(*prompts)
                            paused_mask_preds, iou_preds = interact_model.generate_masks(
                                encoded_img,
                                encoded_prompts,
                                mask_hint=None,
                                blank_promptless_output=True,
                            )
                            track_idx_keeper.clear()

                    # If there are no interactive user prompts but the selected object has tracking prompts, run the tracker
                    if (
                        selected_has_track_prompts
                        and not interactive_user_prompts
                        and is_changed_track_idx
                        and not scrub_just_released
                    ):
                        lost_objects = []
                        paused_mask_preds, iou_preds, paused_obj_score = track_single_object_at_frame(
                            buffer_select_idx,
                            frame_idx,
                            obj_mgr,
                            track_model,
                            encoded_img,
                            object_score_threshold,
                            keep_tracking_after_loss,
                            is_trackhistory_enabled,
                            lost_objects,
                            frame_direction,
                            mask_select_mode,
                            model_family,
                            lost_patience,
                            max_prompt_attn,
                        )
                        notify_lost_objects(toast, obj_mgr, lost_objects)
                        track_idx_keeper.record(frame_idx)

                    # Store encoded prompts as needed (requires new FG/BG points or a box, not hover-only on tracked objects)
                    if store_prompt_btn.read():
                        if not interactive_user_prompts:
                            toast.notify(
                                "Store Prompt ignored: add new FG/BG points or a box first",
                                level="warning",
                            )
                        else:
                            stored_mask_idx = obj_mgr.maskresults_list[buffer_select_idx].idx
                            _, init_mem = track_model.encode_prompt_memory(
                                encoded_img,
                                *prompts,
                                mask_index=stored_mask_idx,
                            )
                            selected_memory = obj_mgr.memory_list[buffer_select_idx]
                            selected_memory.store_prompt_result(
                                init_mem,
                                frame_idx,
                                {
                                    "boxes": prompts[0],
                                    "fg_points": prompts[1],
                                    "bg_points": prompts[2],
                                },
                            )
                            selected_memory.clear_tracking_stop_frame()
                            if clear_history_on_new_prompts:
                                selected_memory.discard_frame_memory()
                            obj_mgr.pending_memory = [
                                item for item in obj_mgr.pending_memory if item["objidx"] != buffer_select_idx
                            ]
                            ui_elems.clear_prompts()
                            track_idx_keeper.clear()
                            toast.notify(
                                f"Prompt stored (Object {obj_mgr.stable_ids[buffer_select_idx]})",
                                level="success",
                            )

                    # Store user-interaction results for selected object while paused
                    if paused_mask_preds is not None and not selected_has_track_prompts:
                        num_points = len(prompts[1]) + len(prompts[2])
                        paused_mask_idx = select_mask_index(
                            iou_preds,
                            model_family,
                            "prompt",
                            mask_select_mode,
                            num_points,
                            len(prompts[0]) > 0,
                        )
                        obj_mgr.maskresults_list[buffer_select_idx].update(
                            paused_mask_preds,
                            paused_mask_idx,
                            paused_obj_score,
                            frame_idx=frame_idx,
                            source="preview",
                        )
                    elif paused_mask_preds is not None and interactive_user_prompts:
                        num_points = len(prompts[1]) + len(prompts[2])
                        paused_mask_idx = select_mask_index(
                            iou_preds,
                            model_family,
                            "prompt",
                            mask_select_mode,
                            num_points,
                            len(prompts[0]) > 0,
                        )
                        obj_mgr.maskresults_list[buffer_select_idx].update(
                            paused_mask_preds,
                            paused_mask_idx,
                            paused_obj_score,
                            frame_idx=frame_idx,
                            source="preview",
                        )

                elif curr_state == STATES.TRACKING:

                    # Only run tracking if we're on a new index
                    if is_changed_track_idx:
                        track_idx_keeper.record(frame_idx)
                        lost_objects = []
                        run_multi_object_tracking(
                            obj_mgr,
                            track_model,
                            encoded_img,
                            frame_idx,
                            object_score_threshold,
                            keep_tracking_after_loss,
                            is_trackhistory_enabled,
                            lost_objects,
                            frame_direction,
                            mask_select_mode,
                            model_family,
                            lost_patience,
                            max_prompt_attn,
                        )
                        notify_lost_objects(toast, obj_mgr, lost_objects)

                # Resolve overlaps before drawing or recording, then commit this frame's memory.
                finalize_frame_masks(obj_mgr, uictrl, track_model, encoded_img, frame.shape[0:2])
                _, is_record_enabled = enable_record_btn.read()
                if has_video_source:
                    maybe_record_current_frame(obj_mgr, uictrl, frame_idx, frame.shape[0:2], is_record_enabled)

                # Update the mask indicators
                selected_mask_uint8, selected_mask_contours, unselected_contours = collect_mask_contours(
                    obj_mgr, buffer_select_idx, uictrl, frame.shape[0:2]
                )
                apply_display_contours(
                    uictrl, unselected_olay, frame, selected_mask_uint8, selected_mask_contours, unselected_contours
                )

                # Keep the prompt undo stack consistent with current overlay contents
                undo_mgr.sync()

                # Display final image
                display_image = obj_mgr.disp_layout.render(**render_limit_dict)
                obj_mgr.finalize_layout_callback_regions()
                obj_mgr.sync_object_grid_viewport_height()
                toast.draw(display_image, ui_elems.image.get_region_xyxy())
                req_break, keypress = window.show(display_image, None if is_paused else 1)
                user_guide.process_events(window)
                if req_break:
                    if not handle_close_request(
                        obj_mgr,
                        video_path,
                        window,
                        history,
                        toast,
                        video_fps,
                        getattr(vreader, "source_info", None) if vreader is not None else None,
                        intensity_range,
                    ):
                        continue
                    clean_exit = True
                    break

                # Updates playback indicator & allows for adjusting playback
                if playback_slider is not None:
                    playback_slider.update(frame_idx)

                # Scale display size up when pressing +/- keys
                if keypress == KEY_ZOOM_IN:
                    display_size_px = min(display_size_px + 50, 10000)
                    render_limit_dict = {render_side: display_size_px}
                    history.store(display_size_px=int(display_size_px))
                if keypress == KEY_ZOOM_OUT:
                    display_size_px = max(display_size_px - 50, min_display_size_px)
                    render_limit_dict = {render_side: display_size_px}
                    history.store(display_size_px=int(display_size_px))

                step_delta = 0
                if has_video_source and is_paused:
                    if keypress in KEY_STEP_BACK_KEYS:
                        step_delta = -1
                    elif keypress in KEY_STEP_FWD_KEYS:
                        step_delta = 1

                if step_delta != 0:
                    while step_delta != 0 and not req_break:
                        prev_idx = vreader.get_frame_index()
                        new_idx = vreader.step_frame_by(step_delta)
                        if new_idx == prev_idx:
                            break

                        frame_idx = new_idx
                        frame = vreader.get_current_frame()
                        cached_encoding = encode_cache.get(frame_idx, compute_device)
                        if cached_encoding is not None:
                            encoded_img = cached_encoding
                        else:
                            encoded_img = strip_detector_branch(
                                interact_model.encode_image(frame, **imgenc_config_dict), model_family
                            )
                            encode_cache.store(frame_idx, encoded_img)
                        imgenc_idx_keeper.record(frame_idx)

                        if any(mem.check_has_prompts() for mem in obj_mgr.memory_list):
                            lost_objects = []
                            step_direction = 1 if step_delta > 0 else -1
                            run_multi_object_tracking(
                                obj_mgr,
                                track_model,
                                encoded_img,
                                frame_idx,
                                object_score_threshold,
                                keep_tracking_after_loss,
                                is_trackhistory_enabled,
                                lost_objects,
                                step_direction,
                                mask_select_mode,
                                model_family,
                                lost_patience,
                                max_prompt_attn,
                            )
                            notify_lost_objects(toast, obj_mgr, lost_objects)
                        else:
                            for maskresult in obj_mgr.maskresults_list:
                                maskresult.clear()
                        track_idx_keeper.record(frame_idx)

                        finalize_frame_masks(obj_mgr, uictrl, track_model, encoded_img, frame.shape[0:2])
                        selected_mask_uint8, selected_mask_contours, unselected_contours = collect_mask_contours(
                            obj_mgr, buffer_select_idx, uictrl, frame.shape[0:2]
                        )
                        apply_display_contours(
                            uictrl, unselected_olay, frame, selected_mask_uint8, selected_mask_contours, unselected_contours
                        )
                        display_image = obj_mgr.disp_layout.render(**render_limit_dict)
                        obj_mgr.finalize_layout_callback_regions()
                        toast.draw(display_image, ui_elems.image.get_region_xyxy())
                        req_break, keypress = window.show(display_image, None)
                        user_guide.process_events(window)
                        if playback_slider is not None:
                            playback_slider.update(frame_idx)

                        _, is_record_enabled = enable_record_btn.read()
                        maybe_record_current_frame(obj_mgr, uictrl, frame_idx, frame.shape[0:2], is_record_enabled)

                        step_delta = 0
                        if not req_break and is_paused:
                            if keypress in KEY_STEP_BACK_KEYS:
                                step_delta = -1
                            elif keypress in KEY_STEP_FWD_KEYS:
                                step_delta = 1

                    if req_break:
                        if not handle_close_request(
                        obj_mgr,
                        video_path,
                        window,
                        history,
                        toast,
                        video_fps,
                        getattr(vreader, "source_info", None) if vreader is not None else None,
                        intensity_range,
                    ):
                            req_break = False
                            continue
                        clean_exit = True
                        break
                    continue

                # Save buffered results to disk
                if has_video_source and buffer_save_btn.read():
                    if obj_mgr.results_buffer.has_data():
                        prompt_and_save_tracking_results(
                            obj_mgr,
                            video_path,
                            window,
                            history,
                            toast,
                            video_fps,
                            getattr(vreader, "source_info", None),
                            intensity_range,
                        )
                    else:
                        toast.notify("No tracking results in memory to save", level="warning")

                if retry_metrics_btn.read():
                    retry_pending_metrics(obj_mgr, toast, window)

                # Wipe out all buffered result data if needed
                if buffer_clear_btn.read():
                    obj_mgr.results_buffer.clear()
                    obj_mgr.pending_analysis = None
                    obj_mgr.last_record_signature = None
                    toast.notify("Results buffer cleared", level="info", duration_sec=1.5)

                if save_session_btn.read():
                    session = session_from_manager(
                        obj_mgr,
                        model_path=model_path,
                        encode_side=imgenc_config_dict["max_side_length"],
                        use_square_sizing=imgenc_config_dict["use_square_sizing"],
                        video_path=video_path,
                    )
                    if len(session["objects"]) == 0:
                        toast.notify("No stored prompts to save", level="warning")
                    else:
                        session_path = pick_save_path(
                            "Save prompt session",
                            [("Session JSON", "*.json")],
                            get_default_save_parent_folder(video_path, history),
                            "session.json",
                        )
                        window.refocus()
                        if session_path:
                            write_session(session_path, session)
                            toast.notify("Session saved", level="success")

                if load_session_btn.read():
                    if vreader is None:
                        toast.notify("Load a video before a session", level="warning")
                    else:
                        session_path = pick_file_path(
                            "Load prompt session",
                            [("Session JSON", "*.json"), ("All files", "*.*")],
                            video_path,
                        )
                        window.refocus()
                        if session_path:
                            try:
                                session = read_session(session_path)
                                mismatch = session_video_mismatch(session, video_path)
                                if mismatch:
                                    show_message_dialog("Session Load Failed", mismatch, kind="error")
                                    window.refocus()
                                    continue
                                if not confirm_before_discarding_results(
                                    "load a session",
                                    obj_mgr,
                                    video_path,
                                    window,
                                    history,
                                    toast,
                                    video_fps,
                                    getattr(vreader, "source_info", None) if vreader is not None else None,
                                    intensity_range,
                                ):
                                    continue
                                has_prompts = any(
                                    memory.check_has_prompts() for memory in obj_mgr.memory_list
                                )
                                if has_prompts:
                                    replace_prompts = ask_yes_no(
                                        "Load Session",
                                        "Replace the current stored prompts with this session?",
                                    )
                                    window.refocus()
                                    if replace_prompts is not True:
                                        continue
                                _restore_prompt_session(session)
                                square_label = "square" if imgenc_config_dict["use_square_sizing"] else "aspect"
                                toast.notify(
                                    f"Session loaded ({model_name}, side {imgenc_config_dict['max_side_length']}, {square_label})",
                                    level="success",
                                )
                            except Exception as err:
                                show_message_dialog("Session Load Failed", str(err), kind="error")
                                window.refocus()

                consecutive_errors = 0

            except KeyboardInterrupt:
                if handle_close_request(
                        obj_mgr,
                        video_path,
                        window,
                        history,
                        toast,
                        video_fps,
                        getattr(vreader, "source_info", None) if vreader is not None else None,
                        intensity_range,
                    ):
                    print("", "Closed with Ctrl+C", sep="\n")
                    clean_exit = True
                    break
            except Exception:
                consecutive_errors += 1
                log_path = write_error_log(traceback.format_exc())
                print("", f"Error (see {log_path})", sep="\n", flush=True)
                if vreader is not None:
                    vreader.pause()
                if consecutive_errors >= 3:
                    folder = None
                    try:
                        folder = emergency_save_results(
                    obj_mgr,
                    video_path,
                    history,
                    video_fps,
                    getattr(vreader, "source_info", None) if vreader is not None else None,
                )
                    except Exception:
                        traceback.print_exc()
                    saved_note = f"\n\nLabels were auto-saved to:\n{folder}" if folder else ""
                    show_message_dialog(
                        "Fatal Error",
                        f"Playback stopped after repeated errors.{saved_note}\n\nLog:\n{log_path}",
                        kind="error",
                    )
                    clean_exit = True
                    break
                show_message_dialog(
                    "Error",
                    f"Playback was paused.\n\nDetails were written to:\n{log_path}",
                    kind="error",
                )
                window.refocus()
    finally:
        # A crash or a killed window should not throw away recorded labels.
        if not clean_exit:
            try:
                emergency_save_results(
                    obj_mgr,
                    video_path,
                    history,
                    video_fps,
                    getattr(vreader, "source_info", None) if vreader is not None else None,
                )
            except Exception:
                traceback.print_exc()
        shortcuts_help.close()
        user_guide.close()
        close_tk_root()
        cv2.destroyAllWindows()
        if vreader is not None:
            vreader.release()


if __name__ == "__main__":
    main()
