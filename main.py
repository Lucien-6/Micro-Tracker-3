#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.2.0"
__app_name__ = "Micro Tracker 3"
__author__ = "Lucien"
__author_email__ = "lucien-6@qq.com"


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import argparse
import gc
import os
import os.path as osp
from time import perf_counter
from enum import Enum
from dataclasses import dataclass

import torch
import cv2
import numpy as np

from muggled_sam.make_sam import make_sam_from_state_dict

from muggled_sam.demo_helpers.ui.window import DisplayWindow, KEY
from muggled_sam.demo_helpers.ui.shortcuts_help import ShortcutsHelpWindow
from muggled_sam.demo_helpers.ui.user_guide_window import UserGuideWindow
from muggled_sam.demo_helpers.ui.video import (
    ReversibleLoopingVideoReader,
    LoopingVideoPlaybackSlider,
    ValueChangeTracker,
)
from muggled_sam.demo_helpers.ui.layout import GridStack, HStack, VStack
from muggled_sam.demo_helpers.ui.buttons import ToggleButton, ImmediateButton, RadioConstraint
from muggled_sam.demo_helpers.ui.text import ValueBlock
from muggled_sam.demo_helpers.ui.base import force_same_min_width
from muggled_sam.demo_helpers.ui.overlays import DrawPolygonsOverlay, MiddleClickCaptureOverlay
from muggled_sam.demo_helpers.ui.helpers.images import linear_gradient_image

from muggled_sam.demo_helpers.shared_ui_layout import PromptUIControl, PromptUI

from muggled_sam.demo_helpers.history_keeper import HistoryKeeper
from muggled_sam.demo_helpers.loading import (
    clean_path_str,
    resolve_default_model_path,
    pick_model_file,
    pick_video_file,
    pick_save_folder,
    ask_save_unsaved_results,
)
from muggled_sam.demo_helpers.prompts import check_have_prompts
from muggled_sam.demo_helpers.contours import get_contours_from_mask
from muggled_sam.demo_helpers.video_data_storage import SAMVideoMemoryBank
from muggled_sam.demo_helpers.saving import (
    build_combined_label_image,
    make_mt_results_folder_name,
    save_tracking_label_tif_sequence,
)
from muggled_sam.demo_helpers.misc import PeriodicVRAMReport, make_device_config, get_default_device_string
from muggled_sam.demo_helpers.model_info import get_token_hw, get_preencoding_hw


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_video_path = None
default_model_path = None
default_prompts_path = None
default_display_size = 900
default_base_size = 1344
default_max_memory_history = 6
default_object_score_threshold = 0.0

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
    default=default_display_size,
    type=int,
    help=f"Controls size of displayed results (default: {default_display_size})",
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
    help="Process the video at it's original aspect ratio",
)
parser.add_argument(
    "-b",
    "--base_size_px",
    default=default_base_size,
    type=int,
    help="Set image processing size (will use model default if not set)",
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
    default=default_object_score_threshold,
    type=float,
    help=f"Threshold below which objects are considered to be 'lost' (default: {default_object_score_threshold})",
)
# For convenience
args = parser.parse_args()
arg_video_path = args.video_path
arg_model_path = args.model_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
use_square_sizing = not args.use_aspect_ratio
imgenc_base_size = args.base_size_px
max_memory_history = args.max_memories
keep_tracking_after_loss = args.keep_bad_objscores
clear_history_on_new_prompts = not args.keep_history_on_new_prompts
object_score_threshold = args.objscore_threshold


def print_startup_banner() -> None:
    """Print project title and author to the terminal before loading resources."""

    line = "=" * 56
    print("", line, sep="\n", flush=True)
    print(f"  {__app_name__}  v{__version__}", flush=True)
    print(f"  by {__author__}  <{__author_email__}>", flush=True)
    print(line, "", sep="\n", flush=True)


# Set up device config
device_config_dict = make_device_config(device_str, use_float32)

# Create history to re-use selected inputs
history = HistoryKeeper()
_, history_modelpath = history.read("model_path")

# Resolve model path without terminal prompts; video is selected from the GUI unless -i is used
try:
    model_path = resolve_default_model_path(__file__, arg_model_path, history_modelpath)
except FileNotFoundError as err:
    print("", str(err), sep="\n", flush=True)
    quit()

if arg_video_path and osp.exists(clean_path_str(arg_video_path)):
    video_path = clean_path_str(arg_video_path)
    has_video_source = True
else:
    video_path = None
    has_video_source = False

history.store(model_path=model_path)
if has_video_source:
    history.store(video_path=video_path, model_path=model_path)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Set up shared image encoder settings (needs to be consistent across image/video frame encodings)
imgenc_config_dict = {"max_side_length": imgenc_base_size, "use_square_sizing": use_square_sizing}


# ---------------------------------------------------------------------------------------------------------------------
# %% Helper Data types


@dataclass
class MaskResults:
    """Storage for (per-object) displayable masking results"""

    preds: torch.Tensor
    idx: int = 0
    objscore: float = 0.0

    @classmethod
    def create(cls, mask_predictions, mask_index=1, object_score=0.0):
        """Helper used to create an empty instance of mask results"""
        empty_predictions = torch.full_like(mask_predictions, -7)
        return cls(empty_predictions, mask_index, object_score)

    def clear(self):
        self.preds = torch.zeros_like(self.preds)
        self.objscore = 0.0
        return self

    def update(self, mask_predictions, mask_index, object_score=None):
        if mask_predictions is not None:
            self.preds = mask_predictions
        if mask_index is not None:
            self.idx = mask_index
        if object_score is not None:
            self.objscore = object_score
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
    put_text_centered(img, "MP4  AVI  MOV  MKV  WEBM", y2 - 28, 0.42, (120, 130, 155), 1)

    return img


def load_sam_models(model_path: str, device_config_dict: dict):
    print("", "Loading model weights...", f"  @ {model_path}", sep="\n", flush=True)
    sam_core = make_sam_from_state_dict(model_path)
    sam_core.to(**device_config_dict)
    interact_model = sam_core.get_interactive_context()
    track_model = sam_core.get_tracking_context()
    return sam_core, interact_model, track_model


def unload_sam_model(sam_core):
    del sam_core
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def print_model_config(model_name, device_config_dict, preencode_hw, token_hw):
    model_device = device_config_dict["device"]
    model_dtype = str(device_config_dict["dtype"]).split(".")[-1]
    image_hw_str = f"{preencode_hw[0]} x {preencode_hw[1]}"
    token_hw_str = f"{token_hw[0]} x {token_hw[1]}"
    print(
        "",
        f"Config ({model_name}):",
        f"  Device: {model_device} ({model_dtype})",
        f"  Resolution HW: {image_hw_str}",
        f"  Tokens HW: {token_hw_str}",
        sep="\n",
        flush=True,
    )


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


def apply_zero_tracking_mask(obj_mgr, objidx: int) -> None:
    """Zero the display mask for an object without running video masking."""
    maskresult = obj_mgr.maskresults_list[objidx]
    zero_preds = maskresult.preds * 0.0
    obj_mgr.maskresults_list[objidx].update(zero_preds, maskresult.idx, maskresult.objscore)


def track_single_object_at_frame(
    objidx: int,
    frame_idx: int,
    obj_mgr,
    track_model,
    encoded_img,
    object_score_threshold: float,
    keep_tracking_after_loss: bool,
    is_trackhistory_enabled: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, float | None]:
    """Run video masking for one object slot, or zero its mask if tracking is stopped at this frame."""
    memory = obj_mgr.memory_list[objidx]
    if not memory.check_has_prompts():
        return None, None, None

    memory.reconcile_tracking_stop_frame(frame_idx)
    if not memory.should_run_video_masking(frame_idx, keep_tracking_after_loss):
        apply_zero_tracking_mask(obj_mgr, objidx)
        return None, None, None

    mask_preds, iou_preds, obj_ptr, obj_score = track_model.step_video_masking(
        encoded_img, **memory.to_dict(), return_best_only=False
    )
    best_mask_idx = iou_preds.argmax(dim=-1)
    obj_score_float = float(obj_score)
    tracked_mask_idx = get_best_mask_index(iou_preds)

    if obj_score_float < object_score_threshold:
        mask_preds = mask_preds * 0.0
        if not keep_tracking_after_loss:
            memory.set_tracking_stop_frame(frame_idx)
    elif is_trackhistory_enabled:
        mem_enc = track_model.encode_frame_memory(
            encoded_img, mask_preds, obj_ptr, obj_score, mask_index=best_mask_idx
        )
        memory.store_frame_result(mem_enc)

    obj_mgr.maskresults_list[objidx].update(mask_preds, tracked_mask_idx, obj_score_float)
    return mask_preds, iou_preds, obj_score_float


def run_multi_object_tracking(
    obj_mgr,
    track_model,
    encoded_img,
    frame_idx: int,
    object_score_threshold,
    keep_tracking_after_loss,
    is_trackhistory_enabled,
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
        )


def collect_mask_contours(obj_mgr, buffer_select_idx, uictrl, preencode_hw):
    """Build selected/unselected contour lists from current mask results."""
    selected_mask_contours, selected_mask_uint8 = None, None
    unselected_contours = []
    for objidx, maskresult in enumerate(obj_mgr.maskresults_list):
        mask_preds, mask_idx = maskresult.preds, maskresult.idx
        mask_uint8 = uictrl.create_hires_mask_uint8(mask_preds, mask_idx, preencode_hw)
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
        maskresult = obj_mgr.maskresults_list[objidx]
        mask_uint8 = uictrl.create_hires_mask_uint8(maskresult.preds, maskresult.idx, output_hw)
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


@dataclass
class TrackingResultsBuffer:
    """Storage for combined per-frame label images (8-bit grayscale, object id = gray value)."""

    frames_dict: dict[int, np.ndarray]

    @classmethod
    def create(cls):
        return cls({})

    def clear(self):
        self.frames_dict.clear()
        return self

    def has_data(self) -> bool:
        return len(self.frames_dict) > 0

    def record_frame(self, frame_idx: int, label_img: np.ndarray):
        self.frames_dict[int(frame_idx)] = label_img.copy()
        return self


def get_video_base_name_for_save(video_path) -> str:
    if video_path is None:
        return "video"
    return osp.splitext(osp.basename(str(video_path)))[0]


def get_default_save_parent_folder(video_path) -> str:
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
        mask_preds, mask_idx = obj_mgr.maskresults_list[objidx].preds, obj_mgr.maskresults_list[objidx].idx
        mask_1ch = uictrl.create_hires_mask_uint8(mask_preds, mask_idx, frame_hw)
        labeled_masks.append((objidx + 1, mask_1ch))
    return labeled_masks


def record_combined_tracking_frame(obj_mgr, uictrl, frame_idx: int, frame_hw):
    labeled_masks = build_object_label_masks(obj_mgr, uictrl, frame_hw)
    if len(labeled_masks) == 0:
        return
    label_img = build_combined_label_image(frame_hw, labeled_masks)
    obj_mgr.results_buffer.record_frame(frame_idx, label_img)


def prompt_and_save_tracking_results(obj_mgr, video_path, window) -> bool:
    if not obj_mgr.results_buffer.has_data():
        return False

    default_parent = get_default_save_parent_folder(video_path)
    parent_folder = pick_save_folder(default_parent)
    window.refocus()
    if parent_folder is None:
        return False

    video_base_name = get_video_base_name_for_save(video_path)
    results_folder_name = make_mt_results_folder_name(video_base_name)
    save_folder = osp.join(parent_folder, results_folder_name)

    try:
        save_folder, num_saved = save_tracking_label_tif_sequence(obj_mgr.results_buffer.frames_dict, save_folder)
    except IOError as err:
        print("", f"Error saving tracking results: {err}", sep="\n", flush=True)
        return False

    print("", f"Saved {num_saved} tracking result frame(s)", f"@ {save_folder}", sep="\n", flush=True)
    obj_mgr.results_buffer.clear()
    return True


def handle_close_request(obj_mgr, video_path, window) -> bool:
    """Return True when the application should exit, False to keep running."""

    if not obj_mgr.results_buffer.has_data():
        return True

    should_save = ask_save_unsaved_results()
    window.refocus()
    if should_save is None:
        print("", "Warning: save dialog unavailable, discarding unsaved results.", sep="\n", flush=True)
        obj_mgr.results_buffer.clear()
        return True

    if should_save:
        if not prompt_and_save_tracking_results(obj_mgr, video_path, window):
            return False
    else:
        obj_mgr.results_buffer.clear()

    return True


class ObjectSlotManager:
    """Manage dynamic object slots, sidebar UI, and per-object tracking/save data."""

    MAX_OBJECTS = 32

    def __init__(
        self,
        init_mask_preds,
        init_mask_idx,
        max_memory_history,
        enable_record_btn,
        buffer_save_btn,
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
        self.buffer_clear_btn = buffer_clear_btn
        self.add_object_btn = add_object_btn
        self.remove_object_btn = remove_object_btn
        self.ui_elems = ui_elems
        self.build_disp_layout_fn = build_disp_layout_fn

        self.maskresults_list: list[MaskResults] = []
        self.results_buffer = TrackingResultsBuffer.create()
        self.memory_list: list[SAMVideoMemoryBank] = []
        self.buffer_btns_list: list[ToggleButton] = []
        self.object_grid = None
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

    def previous_object(self):
        return self.buffer_btn_constraint.previous()

    def next_object(self):
        return self.buffer_btn_constraint.next()

    def select_object(self, objidx: int) -> bool:
        """Select an object by index. Returns True if the selection changed."""
        if not (0 <= objidx < len(self.maskresults_list)):
            return False
        if objidx == self.get_select_idx():
            return False
        self.buffer_btn_constraint.change_to(objidx)
        return True

    def _append_object_data(self):
        self.maskresults_list.append(MaskResults.create(self.init_mask_preds, self.init_mask_idx))
        self.memory_list.append(SAMVideoMemoryBank(self.max_memory_history, max_prompt_memory=self.max_prompt_memory))

    def _rebuild_object_rows(self):
        self.buffer_btns_list = []

        for objidx in range(len(self.maskresults_list)):
            buffer_btn = ToggleButton(f"Object {1 + objidx}", button_height=20, text_scale=0.5, on_color=(145, 120, 65))
            self.buffer_btns_list.append(buffer_btn)

        if len(self.buffer_btns_list) > 1:
            force_same_min_width(*self.buffer_btns_list)

    def _rebuild_ui(self, select_idx: int, refresh_window: bool = True):
        select_idx = max(0, min(select_idx, len(self.maskresults_list) - 1))
        self._rebuild_object_rows()
        self.buffer_btn_constraint = RadioConstraint(*self.buffer_btns_list, initial_selected_index=select_idx)
        self.object_grid = GridStack(*self.buffer_btns_list, num_columns=2).set_debug_name("ObjectGrid")
        self.save_sidebar = VStack(
            self.enable_record_btn,
            self.object_grid,
            HStack(self.add_object_btn, self.remove_object_btn),
            HStack(self.buffer_save_btn, self.buffer_clear_btn),
        )
        self.disp_layout = self.build_disp_layout_fn(self.save_sidebar)
        if refresh_window and self.window is not None:
            self.window.replace_mouse_callbacks(self.disp_layout)

    def bind_window(self, window: DisplayWindow):
        self.window = window
        self._rebuild_ui(self.get_select_idx(), refresh_window=True)

    def add_object(self, select_new=True, clear_prompts=True) -> bool:
        if len(self.maskresults_list) >= self.MAX_OBJECTS:
            return False

        self._append_object_data()
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

        old_select = self.get_select_idx()
        del self.maskresults_list[objidx]
        del self.memory_list[objidx]

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
        self.results_buffer.clear()
        self._append_object_data()
        self._rebuild_ui(0)
        if clear_prompts:
            self.ui_elems.clear_prompts()
        return self


# ---------------------------------------------------------------------------------------------------------------------
# %% Load model & optional video source

model_name = osp.basename(model_path)
video_name = osp.basename(video_path) if has_video_source else NO_VIDEO_LABEL
placeholder_frame = create_placeholder_frame()

print_startup_banner()
sam_core, interact_model, track_model = load_sam_models(model_path, device_config_dict)

vreader = None
video_fps = 30.0
if has_video_source:
    vreader = ReversibleLoopingVideoReader(video_path).release()
    video_fps = vreader.get_fps()
    sample_frame = vreader.get_sample_frame()
else:
    sample_frame = placeholder_frame.copy()

encoded_img, init_mask_preds, iou_preds, init_mask_idx, preencode_hw, token_hw = run_initial_model_pass(
    interact_model, sample_frame, imgenc_config_dict
)
prediction_hw = init_mask_preds.shape[2:]
print_model_config(model_name, device_config_dict, preencode_hw, token_hw)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up UI

# Playback control UI for adjusting video position
playback_slider = LoopingVideoPlaybackSlider(vreader, stay_paused_on_change=True) if has_video_source else None

# Set up shared UI elements & control logic
ui_elems = PromptUI(sample_frame, 2)
uictrl = PromptUIControl(ui_elems)

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
store_prompt_btn = ImmediateButton("Store Prompt", text_scale=0.35, color=(145, 160, 40))
clear_prompts_btn = ImmediateButton("Clear Prompts", text_scale=0.35, color=(80, 110, 230))
enable_history_btn = ToggleButton("Enable History", default_state=True, text_scale=0.35, on_color=(90, 85, 115))
clear_history_btn = ImmediateButton("Clear History", text_scale=0.35, color=(130, 60, 90))
force_same_min_width(store_prompt_btn, clear_prompts_btn, enable_history_btn, clear_history_btn)


# Create save UI (object rows are managed dynamically by ObjectSlotManager)
enable_record_btn = ToggleButton("Enable Recording", default_state=False, on_color=(0, 15, 255), button_height=60)
buffer_save_btn = ImmediateButton("Save Results", button_height=30, text_scale=0.5, color=(110, 145, 65))
buffer_clear_btn = ImmediateButton("Clear Results", button_height=30, text_scale=0.5, color=(80, 60, 190))
add_object_btn = ImmediateButton("Add Object", button_height=30, text_scale=0.5, color=(90, 130, 90))
remove_object_btn = ImmediateButton("Remove Object", button_height=30, text_scale=0.5, color=(130, 70, 70))
force_same_min_width(buffer_save_btn, buffer_clear_btn, add_object_btn, remove_object_btn)

# Set up resource switcher bar (model / video)
model_btn = ImmediateButton(
    format_resource_button_label("Model", model_name), text_scale=0.4, color=(90, 110, 145)
)
video_btn = ImmediateButton(
    format_resource_button_label("Video", video_name), text_scale=0.4, color=(110, 95, 75)
)
force_same_min_width(model_btn, video_btn)
header_bar = HStack(model_btn, video_btn).set_debug_name("HeaderResourceBar")


def build_disp_layout(save_sidebar):
    return VStack(
        header_bar,
        HStack(ui_elems.layout, save_sidebar),
        playback_slider if has_video_source else None,
        HStack(num_prompts_text, track_btn, num_history_text),
        HStack(store_prompt_btn, clear_prompts_btn, reversal_btn, enable_history_btn, clear_history_btn),
        HStack(vram_text, device_text, shortcuts_btn),
    ).set_debug_name("DisplayLayout")


obj_mgr = ObjectSlotManager(
    init_mask_preds,
    init_mask_idx,
    max_memory_history,
    enable_record_btn,
    buffer_save_btn,
    buffer_clear_btn,
    add_object_btn,
    remove_object_btn,
    ui_elems,
    build_disp_layout,
)
disp_layout = obj_mgr.disp_layout

# Render out an image with a target size, to figure out which side we should limit when rendering
display_image = disp_layout.render(h=display_size_px, w=display_size_px)
render_side = "h" if display_image.shape[1] > display_image.shape[0] else "w"
render_limit_dict = {render_side: display_size_px}
min_display_size_px = disp_layout._rdr.limits.min_h if render_side == "h" else disp_layout._rdr.limits.min_w


# ---------------------------------------------------------------------------------------------------------------------
# %% Video loop

# Setup display window
window = DisplayWindow(f"{__app_name__}  by {__author__}", display_fps=60)
obj_mgr.bind_window(window)
shortcuts_help = ShortcutsHelpWindow(offset_xy=(60, 60))
shortcuts_help.attach_f1_toggle(window)

user_guide = UserGuideWindow(__app_name__, __version__, __author__, __author_email__, initial_lang="zh")
user_guide.attach_h_toggle(window)

# Change tools on left/right arrow keys; change objects on up/down arrow keys
uictrl.attach_arrowkey_callbacks(window)
if has_video_source:
    window.attach_keypress_callback(" ", vreader.toggle_pause)
window.attach_arrow_keypress_callback("up", obj_mgr.previous_object)
window.attach_arrow_keypress_callback("down", obj_mgr.next_object)
window.attach_keypress_callback("w", obj_mgr.previous_object)
window.attach_keypress_callback("s", obj_mgr.next_object)
window.attach_keypress_callback("+", add_object_btn.click)
window.attach_keypress_callback("=", add_object_btn.click)
window.attach_keypress_callback("-", remove_object_btn.click)
window.attach_keypress_callback(KEY.TAB, store_prompt_btn.click)
window.attach_keypress_callback("r", reversal_btn.toggle)

# For clarity, some additional keypress codes
KEY_ZOOM_IN = ord("]")
KEY_ZOOM_OUT = ord("[")
KEY_STEP_BACK = ord("a")
KEY_STEP_FWD = ord("d")

# Set up various value tracking helpers
imgenc_idx_keeper = ValueChangeTracker(-1)
track_idx_keeper = ValueChangeTracker(-1)
pause_keeper = ValueChangeTracker(vreader.get_pause_state() if has_video_source else True)
vram_report = PeriodicVRAMReport(update_period_ms=2000)

# Helper used to define/keep track of all states of the UI
STATES = Enum(
    "state", ["ADJUST_PLAYBACK", "SWITCH_PAUSE_ON", "SWITCH_PAUSE_OFF", "TRACKING", "PAUSED", "NO_TRANSISTION"]
)

# Set up per-object storage for masking/saving results (managed by obj_mgr)

if has_video_source:
    vreader.pause()
curr_state = STATES.PAUSED
tran_state = STATES.NO_TRANSISTION
video_iter = iter(vreader) if has_video_source else None
_, _, prev_selected_tool = ui_elems.tools_constraint.read()
was_playback_adjusting = False
hover_preview_suppressed = False
try:

    while True:
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
                unload_sam_model(sam_core)
                model_path = picked_model_path
                model_name = osp.basename(model_path)
                if has_video_source:
                    history.store(video_path=video_path, model_path=model_path)
                else:
                    history.store(model_path=model_path)

                sam_core, interact_model, track_model = load_sam_models(model_path, device_config_dict)
                model_btn.set_label(format_resource_button_label("Model", model_name))

                if has_video_source:
                    vreader.set_playback_position(0)
                    sample_frame = vreader.get_sample_frame()
                else:
                    sample_frame = placeholder_frame.copy()
                ui_elems.image.set_image(sample_frame)
                encoded_img, init_mask_preds, iou_preds, init_mask_idx, preencode_hw, token_hw = run_initial_model_pass(
                    interact_model, sample_frame, imgenc_config_dict
                )
                print_model_config(model_name, device_config_dict, preencode_hw, token_hw)

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

        if video_btn.read():
            picked_video_path = pick_video_file(video_path)
            window.refocus()
            if picked_video_path and (picked_video_path != video_path):
                if vreader is not None:
                    vreader.release()

                video_path = picked_video_path
                video_name = osp.basename(video_path)
                history.store(video_path=video_path, model_path=model_path)

                vreader = ReversibleLoopingVideoReader(video_path).release()
                video_fps = vreader.get_fps()
                sample_frame = vreader.get_sample_frame()
                first_video_load = not has_video_source
                has_video_source = True

                if playback_slider is None:
                    playback_slider = LoopingVideoPlaybackSlider(vreader, stay_paused_on_change=True)
                else:
                    playback_slider.replace_video_reader(vreader)

                video_iter = iter(vreader)
                video_btn.set_label(format_resource_button_label("Video", video_name))

                if first_video_load:
                    window.attach_keypress_callback(" ", vreader.toggle_pause)
                    obj_mgr._rebuild_ui(obj_mgr.get_select_idx())

                ui_elems.image.set_image(sample_frame)
                encoded_img, init_mask_preds, iou_preds, init_mask_idx, preencode_hw, token_hw = run_initial_model_pass(
                    interact_model, sample_frame, imgenc_config_dict
                )
                print_model_config(model_name, device_config_dict, preencode_hw, token_hw)

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
        if has_video_source and is_reversed_changed:
            vreader.toggle_reverse_state(reverse_video)

        # Read controls
        is_changed_pause_state = pause_keeper.is_changed(is_paused)
        _, is_trackhistory_enabled = enable_history_btn.read()

        is_changed_buffer, buffer_select_idx, _ = obj_mgr.read_selection()
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
            pass

        # Wipe out buffered data
        if clear_prompts_btn.read():
            obj_mgr.memory_list[buffer_select_idx].clear(clear_frame_memory=False)
            obj_mgr.maskresults_list[buffer_select_idx].clear()
            track_idx_keeper.clear()
        if clear_history_btn.read():
            obj_mgr.memory_list[buffer_select_idx].clear(clear_prompt_memory=False)
            track_idx_keeper.clear()

        if add_object_btn.read():
            if obj_mgr.add_object():
                layout_image = obj_mgr.disp_layout.render(h=display_size_px, w=display_size_px)
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
            if obj_mgr.remove_selected_object():
                layout_image = obj_mgr.disp_layout.render(h=display_size_px, w=display_size_px)
                render_side = "h" if layout_image.shape[1] > layout_image.shape[0] else "w"
                render_limit_dict = {render_side: display_size_px}
                min_display_size_px = (
                    obj_mgr.disp_layout._rdr.limits.min_h if render_side == "h" else obj_mgr.disp_layout._rdr.limits.min_w
                )
                buffer_select_idx = obj_mgr.get_select_idx()

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
            tran_state = STATES.NO_TRANSISTION

        # Encode any 'new' frames as needed (but not while the playback slider is being dragged)
        need_image_encode = has_video_source and imgenc_idx_keeper.is_changed(frame_idx)
        should_encode_frame = (need_image_encode and not is_playback_adjusting) or (
            scrub_just_released and have_any_stored_prompts
        )
        if should_encode_frame:
            encoded_img = interact_model.encode_image(frame, **imgenc_config_dict)
            imgenc_idx_keeper.record(frame_idx)

        if scrub_just_released and have_any_stored_prompts:
            run_multi_object_tracking(
                obj_mgr,
                track_model,
                encoded_img,
                frame_idx,
                object_score_threshold,
                keep_tracking_after_loss,
                is_trackhistory_enabled,
            )
            track_idx_keeper.record(frame_idx)

        # Wipe out masking/contours when jumping around playback (otherwise stays over top of changing video!)
        if is_playback_adjusting:
            for maskresult in obj_mgr.maskresults_list:
                maskresult.clear()

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

            pass

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
                paused_mask_preds, iou_preds, paused_obj_score = track_single_object_at_frame(
                    buffer_select_idx,
                    frame_idx,
                    obj_mgr,
                    track_model,
                    encoded_img,
                    object_score_threshold,
                    keep_tracking_after_loss,
                    is_trackhistory_enabled,
                )
                track_idx_keeper.record(frame_idx)

            # Store encoded prompts as needed (requires new FG/BG points or a box, not hover-only on tracked objects)
            if store_prompt_btn.read():
                if not interactive_user_prompts:
                    print(
                        "",
                        "Store Prompt ignored: add new foreground/background points or a box on the current frame first.",
                        sep="\n",
                        flush=True,
                    )
                else:
                    _, init_mem = track_model.encode_prompt_memory(
                        encoded_img,
                        *prompts,
                        mask_index=None,
                    )
                    selected_memory = obj_mgr.memory_list[buffer_select_idx]
                    selected_memory.store_prompt_result(init_mem)
                    selected_memory.clear_tracking_stop_frame()
                    ui_elems.clear_prompts()
                    track_idx_keeper.clear()

            # Store user-interaction results for selected object while paused
            if paused_mask_preds is not None:
                paused_mask_idx = get_best_mask_index(iou_preds)
                obj_mgr.maskresults_list[buffer_select_idx].update(paused_mask_preds, paused_mask_idx, paused_obj_score)

        elif curr_state == STATES.TRACKING:

            # Only run tracking if we're on a new index
            if is_changed_track_idx:
                track_idx_keeper.record(frame_idx)
                run_multi_object_tracking(
                    obj_mgr,
                    track_model,
                    encoded_img,
                    frame_idx,
                    object_score_threshold,
                    keep_tracking_after_loss,
                    is_trackhistory_enabled,
                )

        # Update the mask indicators
        selected_mask_uint8, selected_mask_contours, unselected_contours = collect_mask_contours(
            obj_mgr, buffer_select_idx, uictrl, preencode_hw
        )
        apply_display_contours(
            uictrl, unselected_olay, frame, selected_mask_uint8, selected_mask_contours, unselected_contours
        )

        # Display final image
        display_image = obj_mgr.disp_layout.render(**render_limit_dict)
        req_break, keypress = window.show(display_image, None if is_paused else 1)
        user_guide.process_events(window)
        if req_break:
            if not handle_close_request(obj_mgr, video_path, window):
                continue
            break

        # Updates playback indicator & allows for adjusting playback
        if playback_slider is not None:
            playback_slider.update(frame_idx)

        # Scale display size up when pressing +/- keys
        if keypress == KEY_ZOOM_IN:
            display_size_px = min(display_size_px + 50, 10000)
            render_limit_dict = {render_side: display_size_px}
        if keypress == KEY_ZOOM_OUT:
            display_size_px = max(display_size_px - 50, min_display_size_px)
            render_limit_dict = {render_side: display_size_px}

        step_delta = 0
        if has_video_source and is_paused:
            if keypress == KEY_STEP_BACK:
                step_delta = -1
            elif keypress == KEY_STEP_FWD:
                step_delta = 1

        if step_delta != 0:
            while step_delta != 0 and not req_break:
                prev_idx = vreader.get_frame_index()
                new_idx = vreader.step_frame_by(step_delta)
                if new_idx == prev_idx:
                    break

                frame_idx = new_idx
                frame = vreader.get_current_frame()
                encoded_img = interact_model.encode_image(frame, **imgenc_config_dict)
                imgenc_idx_keeper.record(frame_idx)

                if any(mem.check_has_prompts() for mem in obj_mgr.memory_list):
                    run_multi_object_tracking(
                        obj_mgr,
                        track_model,
                        encoded_img,
                        frame_idx,
                        object_score_threshold,
                        keep_tracking_after_loss,
                        is_trackhistory_enabled,
                    )
                else:
                    for maskresult in obj_mgr.maskresults_list:
                        maskresult.clear()
                track_idx_keeper.record(frame_idx)

                selected_mask_uint8, selected_mask_contours, unselected_contours = collect_mask_contours(
                    obj_mgr, buffer_select_idx, uictrl, preencode_hw
                )
                apply_display_contours(
                    uictrl, unselected_olay, frame, selected_mask_uint8, selected_mask_contours, unselected_contours
                )
                display_image = obj_mgr.disp_layout.render(**render_limit_dict)
                req_break, keypress = window.show(display_image, None)
                user_guide.process_events(window)
                if playback_slider is not None:
                    playback_slider.update(frame_idx)

                _, is_record_enabled = enable_record_btn.read()
                if is_record_enabled:
                    record_combined_tracking_frame(obj_mgr, uictrl, frame_idx, frame.shape[0:2])

                step_delta = 0
                if not req_break and is_paused:
                    if keypress == KEY_STEP_BACK:
                        step_delta = -1
                    elif keypress == KEY_STEP_FWD:
                        step_delta = 1

            if req_break:
                if not handle_close_request(obj_mgr, video_path, window):
                    req_break = False
                    continue
                break
            continue

        # Handle recording of combined label images
        _, is_record_enabled = enable_record_btn.read()
        if has_video_source and is_record_enabled and curr_state == STATES.TRACKING:
            record_combined_tracking_frame(obj_mgr, uictrl, frame_idx, frame.shape[0:2])

        # Save buffered results to disk
        if has_video_source and buffer_save_btn.read():
            if obj_mgr.results_buffer.has_data():
                prompt_and_save_tracking_results(obj_mgr, video_path, window)
            else:
                print("", "No tracking results in memory to save.", sep="\n", flush=True)

        # Wipe out all buffered result data if needed
        if buffer_clear_btn.read():
            obj_mgr.results_buffer.clear()

except KeyboardInterrupt:
    print("", "Closed with Ctrl+C", sep="\n")

finally:
    # Clean up resources
    shortcuts_help.close()
    user_guide.close()
    cv2.destroyAllWindows()
    if vreader is not None:
        vreader.release()
