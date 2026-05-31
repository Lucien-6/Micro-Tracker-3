#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from dataclasses import dataclass

from .ui.window import KEY
from .ui.layout import HStack, VStack, OverlayStack
from .ui.buttons import ToggleButton, ImmediateButton, RadioConstraint
from .ui.overlays import HoverOverlay, BoxSelectOverlay, PointSelectOverlay, DrawPolygonsOverlay
from .ui.images import ExpandingImage
from .ui.base import force_same_max_height, force_same_min_height

import cv2
import numpy as np
import torch.nn as nn

# For type hints
from numpy import ndarray
from torch import Tensor
from .ui.window import DisplayWindow


# ---------------------------------------------------------------------------------------------------------------------
# %% Data types


@dataclass
class ToolButtonsGroup:
    hover: ToggleButton
    box: ToggleButton
    fgpt: ToggleButton
    bgpt: ToggleButton
    clear: ImmediateButton

    def totuple(self) -> tuple:
        """Helper which returns a tuple of tool items"""
        return (self.hover, self.box, self.fgpt, self.bgpt, self.clear)

    def enable(self, enabled=True):
        for item in self.totuple():
            item.enable(enabled)
        return self


@dataclass
class OverlayGroup:
    polygon: DrawPolygonsOverlay | None
    hover: HoverOverlay | None
    box: BoxSelectOverlay | None
    fgpt: PointSelectOverlay | None
    bgpt: PointSelectOverlay | None

    def totuple(self) -> tuple:
        """Helper which returns a tuple of all overlays"""
        return (self.polygon, self.hover, self.box, self.fgpt, self.bgpt)

    def enable(self, enabled=True):
        for item in self.totuple():
            item.enable(enabled)
        return self

    def clear_all(self, flag_is_changed=True):
        for item in self.totuple():
            try:
                item.clear(flag_is_changed)
            except TypeError:
                item.clear()
        return self


# ---------------------------------------------------------------------------------------------------------------------
# %% Builder functions


def build_tool_overlays() -> OverlayGroup:
    """Helper used to build overlay UI (draws polygon outlines, bounding-boxes, points etc.)"""

    # Set up prompt UI interactions
    hover_olay = HoverOverlay()
    box_olay = BoxSelectOverlay(thickness=2, bg_color=None)
    fgpt_olay = PointSelectOverlay((0, 255, 0), point_radius=3)
    bgpt_olay = PointSelectOverlay((0, 0, 255), point_radius=3, bg_color=None)
    polygon_olay = DrawPolygonsOverlay((100, 10, 255), bg_color=None)

    return OverlayGroup(polygon_olay, hover_olay, box_olay, fgpt_olay, bgpt_olay)


def build_tool_buttons(text_scale=0.75) -> tuple[ToolButtonsGroup, RadioConstraint]:
    """Helper used to build tool group UI (tool select buttons + prompt clear button)"""

    # Set up tool selection UI
    hover_btn, box_btn, fgpt_btn, bgpt_btn = ToggleButton.many(
        "Hover", "Box", "FG Point", "BG Point", text_scale=text_scale
    )
    clear_all_prompts_btn = ImmediateButton("Clear", color=(0, 0, 150), text_scale=text_scale)

    # Set up constraint so only 1 tool can be active (excluding clear button, which isn't toggled)
    tools_group = ToolButtonsGroup(hover_btn, box_btn, fgpt_btn, bgpt_btn, clear_all_prompts_btn)
    tool_constraint = RadioConstraint(hover_btn, box_btn, fgpt_btn, bgpt_btn)

    return tools_group, tool_constraint


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class PromptUI:
    """
    Hacky-ish class used to bundle all 'base' UI elements which are shared
    for (most?) SAM model use cases. This consists of a regular image stacked
    alongside multiple mask images (which also act as radio buttons) for showing
    SAM mask results, along with support for rendering segmentation outlines
    as an overlay. This is stored as a 'display_block' component

    Additionally this class includes a tool-select bar for providing model prompts
    through hover/box/fg-point/bg-point buttons, along with the corresponding
    overlay graphics (i.e. rendering boxes or fg points)
    """

    # .................................................................................................................

    def __init__(
        self, full_image_bgr: ndarray, target_aspect_ratio=2.0, tool_button_text_scale=0.75
    ):

        # Build out UI component pieces
        self.olays = build_tool_overlays()
        self.tools, self.tools_constraint = build_tool_buttons(tool_button_text_scale)

        # Build main layout!
        self.image, self.overlay_img, self.display_block = self._build_display_block(full_image_bgr)
        self.layout = self._build_ui_layout()

    # .................................................................................................................

    def _build_display_block(self, image_bgr: ndarray) -> tuple[ExpandingImage, HStack]:
        """Function used to build out the central display block (main image only)."""

        # Set up display image
        main_display_image = ExpandingImage(image_bgr).set_debug_name("MainDisplayImage")
        imgoverlay_cb = OverlayStack(main_display_image, *self.olays.totuple())
        imgoverlay_cb._rdr.pad.color = (35, 25, 30)

        main_display_block = imgoverlay_cb
        main_display_block.set_debug_name("MainDisplayBlock")

        return main_display_image, imgoverlay_cb, main_display_block

    # .................................................................................................................

    def _build_ui_layout(self) -> VStack:
        """Function used to build out the final UI block which includes prompt tool buttons"""

        # Tie tool overlays to toggle buttons (so only 1 overlay responds to user input, based on tool selection)
        self.tools.hover.add_on_change_listeners(self.olays.hover.enable)
        self.tools.box.add_on_change_listeners(self.olays.box.enable)
        self.tools.fgpt.add_on_change_listeners(self.olays.fgpt.enable)
        self.tools.bgpt.add_on_change_listeners(self.olays.bgpt.enable)

        # Set up full display layout
        toolselect_bar = HStack(*self.tools.totuple())
        display_layout = VStack(toolselect_bar, self.display_block)
        display_layout.set_debug_name("DisplayLayout")

        return display_layout

    # .................................................................................................................

    def enable_tools(self, enable=True, clear_prompt_data_on_disable=True):
        """Helper used to enable/disable the tool components of the UI"""

        self.tools.enable(enable)
        if enable:
            # Only the overlay of the selected tool should be enabled
            _, tool_select_idx, _ = self.tools_constraint.read()
            olays_ordered_by_tool = [self.olays.hover, self.olays.box, self.olays.fgpt, self.olays.bgpt]
            olay_select = olays_ordered_by_tool[tool_select_idx]
            olay_select.enable(enable)

        # Wipe out prompt data if needed (to clear display of prompts)
        if not enable and clear_prompt_data_on_disable:
            self.clear_prompts()

        return self

    # .................................................................................................................

    def clear_prompts(self, keep_hover_position=False):
        """Helper used to wipe out prompt data from overlays"""

        self.olays.hover.clear(keep_position=keep_hover_position)
        self.olays.box.clear()
        self.olays.fgpt.clear()
        self.olays.bgpt.clear()

        return self

    # .................................................................................................................


class BaseUIControl:
    """
    Helper class used to manage access to the base UI implementation.
    Includes functionality for rendering masked images & final 'hi-res' mask results.
    """

    # .................................................................................................................

    def __init__(self, ui_elements: PromptUI):
        self.elems = ui_elements

    # .................................................................................................................

    def attach_arrowkey_callbacks(self, window: DisplayWindow):
        """Helper used to attach keypress callbacks for UI arrow-key controls"""
        return self

    # .................................................................................................................

    def update_main_display_image(
        self, image_bgr: ndarray, mask_uint8: ndarray, mask_contours_norm
    ) -> None:
        """Helper used to update the displayed image + mask outlines"""

        self.elems.olays.polygon.set_polygons(mask_contours_norm)
        self.elems.image.set_image(image_bgr)

        return

    # .................................................................................................................

    @staticmethod
    def create_hires_mask_uint8(mask_predictions, mask_select_index, output_hw, mask_threshold=0.0) -> ndarray:
        """Draws binary mask matching the given output height & width. Returns: mask_uint8_1ch"""
        mask_select = mask_predictions[:, mask_select_index].float()
        return make_hires_mask_uint8(mask_select, output_hw, mask_threshold).squeeze(0)

    # .................................................................................................................


class PromptUIControl(BaseUIControl):
    """
    An extension to the BaseUIControl class, with additional
    functions for dealing with prompts
    """

    # .................................................................................................................

    def __init__(self, prompt_ui: PromptUI):
        super().__init__(prompt_ui)

    # .................................................................................................................

    def load_initial_prompts(self, initial_prompts_dict: dict):
        """
        Can be used to initialize prompts based on a given dictionary
        The dictionary should contain keys: "boxes", "fg_points", "bg_points"
        """

        # Bail if we aren't given the right data type (e.g. given 'None')
        if not isinstance(initial_prompts_dict, dict):
            return

        # Exhaust the clear button, in case it's been triggered (don't want next read to clear our init prompts!)
        self.elems.tools.clear.read()

        # Initialize overlays with starting prompts
        self.elems.olays.box.add_boxes(*initial_prompts_dict.get("boxes", []))
        self.elems.olays.fgpt.add_points(*initial_prompts_dict.get("fg_points", []))
        self.elems.olays.bgpt.add_points(*initial_prompts_dict.get("bg_points", []))

        return

    # .................................................................................................................

    def attach_arrowkey_callbacks(self, window: DisplayWindow):
        """Helper used to attach keypress callbacks so that tools can be switched with Tab / Shift+Tab"""

        tool_const = self.elems.tools_constraint
        window.attach_keypress_callback(KEY.TAB, tool_const.next)
        window.attach_shift_tab_callback(tool_const.previous)
        window.attach_keypress_callback("c", self.elems.tools.clear.click)

        return self

    # .................................................................................................................

    def read_prompts(self) -> tuple[bool, tuple[list, list, list], bool]:
        """
        Helper used to manage prompt reading, as well as quality-of-life behaviors when interpretting prompts
        Returns:
            need_prompt_reencoding, (boxes_xy1xy2_norm_list, fg_xy_norm_list, bg_xy_norm_list), hover_input_changed
        """

        need_prompt_clear = self.elems.tools.clear.read()
        if need_prompt_clear:
            self.elems.olays.clear_all(flag_is_changed=True)

        return read_prompts(self.elems.olays, self.elems.tools, self.elems.tools_constraint)

    # .................................................................................................................


class ReusableBaseImage:
    """
    Convenience class, used to manage a re-usable (static) image
    that is re-sized to a target display size from an original
    image that is assumed to be much larger.
    This can help reduce cpu load since we avoid using
    (and repeatedly downscaling) the original image which may
    be much larger/heavier to work with!
    """

    # .................................................................................................................

    def __init__(self, full_image_bgr: ndarray):

        # Initialize state values
        self._full_img = self._disp_img = self._prev_h = self._prev_w = None
        self.set_new_image(full_image_bgr)

    def set_new_image(self, new_image_bgr: ndarray):
        """
        Store a new image to be cached. This isn't expected to happen often!
        (setting the image frequently defeats the purpose of caching)
        """

        self._full_img = new_image_bgr
        self._disp_img = new_image_bgr.copy()
        self._prev_h, self._prev_w = new_image_bgr.shape[0:2]

        return self

    def regenerate(self, new_display_hw):
        """Resizes the original input image to the given display size or re-uses a cached copy at the given size"""

        # Resize original image to given display size and store for re-use
        disp_h, disp_w = new_display_hw
        if disp_h != self._prev_h or disp_w != self._prev_w:
            self._disp_img = cv2.resize(self._full_img, dsize=(disp_w, disp_h))
            prev_disp_h, prev_disp_w = self._disp_img.shape[0:2]

        return self._disp_img

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def read_prompts(
    overlays_group: OverlayGroup,
    tools_group: ToolButtonsGroup,
    tools_constraint: RadioConstraint,
    debug_name="",
) -> tuple[bool, tuple[list, list, list], bool]:
    """
    Helper used to standardize prompt reading, assuming standard
    hover/box/fg/bg UI.

    Returns:
        is_prompt_changed, (boxes_xy1xy2_norm_list, fg_xy_norm_list, bg_xy_norm_list), hover_input_changed
    """

    # Read all overlay states (which is where prompts are actually held!)
    box_prompt_changed, boxes_xy1xy2_norm_list = overlays_group.box.read()
    fg_prompt_changed, fg_xy_norm_list = overlays_group.fgpt.read()
    bg_prompt_changed, bg_xy_norm_list = overlays_group.bgpt.read()
    _, _, selected_tool_elem = tools_constraint.read()

    # Only add hover point when the tool is active
    hover_changed = False
    if selected_tool_elem is tools_group.hover:

        # Add hover points (if any) as foreground prompts
        hover_changed, clicked_left, clicked_right, hover_xy_event = overlays_group.hover.read()
        if hover_xy_event.is_in_region and not clicked_right:
            fg_xy_norm_list = tuple([*fg_xy_norm_list, hover_xy_event.xy_norm])

        # Treat hover clicks as foreground/background points (and switch to FG/BG tool)
        if clicked_left:
            tools_constraint.change_to(tools_group.fgpt)
            overlays_group.fgpt.add_points(hover_xy_event.xy_norm)
        elif clicked_right and hover_xy_event.is_in_region:
            tools_constraint.change_to(tools_group.bgpt)
            overlays_group.bgpt.add_points(hover_xy_event.xy_norm)

    # Toggle back to hover in case where an fg/bg point is removed and no other points remain
    if fg_prompt_changed or bg_prompt_changed:
        no_prompts = sum(len(pts) for pts in (fg_xy_norm_list, boxes_xy1xy2_norm_list, bg_xy_norm_list)) == 0
        if no_prompts:
            tools_constraint.change_to(tools_group.hover)
            overlays_group.hover.clear(keep_position=False)

    # Bundle all prompt changes into single check for convenience
    is_prompt_changed = any((hover_changed, box_prompt_changed, fg_prompt_changed, bg_prompt_changed))

    return is_prompt_changed, (boxes_xy1xy2_norm_list, fg_xy_norm_list, bg_xy_norm_list), hover_changed


# .....................................................................................................................


def make_hires_mask_uint8(mask_prediction: Tensor, output_hw=(1024, 1024), mask_threshold=0.0) -> ndarray:
    """
    Helper used to draw a high-resolution binary uint8 (numpy)
    version of a given mask prediction. Expects a single mask
    prediction, but will work with a variety of input shapes.
    Supports inputs of shape:
        BxNxHxW, NxHxW, HxW

    Will return result with matching shape dimensions,
    but with output H/W sizing in numpy uint8 format
    """

    # Set up dimension padding (for interpolation func) & post-indexing to remove padded dimensions
    preds, squeeze_idx = mask_prediction, slice(None)
    if preds.ndim == 3:
        preds = mask_prediction[None]
        squeeze_idx = (0,)
    elif preds.ndim == 2:
        preds = mask_prediction[None, None]
        squeeze_idx = (0, 0)

    mask_upscale = nn.functional.interpolate(preds, size=output_hw, mode="bilinear", align_corners=False)
    return ((mask_upscale[squeeze_idx] > mask_threshold) * 255).byte().cpu().numpy()
