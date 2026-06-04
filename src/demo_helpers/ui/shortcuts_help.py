#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from .window import KEY
from .helpers.images import linear_gradient_image, draw_box_outline
from .helpers.text import TextDrawer


# ---------------------------------------------------------------------------------------------------------------------
# %% Data

# OpenCV Hershey fonts only render ASCII reliably; avoid Unicode punctuation in all panel strings.

F1_KEYCODES = (KEY.WIN_F1, KEY.GTK_F1)

SHORTCUT_SECTIONS = (
    (
        "Playback",
        (95, 145, 210),
        (
            ("Space", "Play / pause video"),
            ("Left / Right", "Step backward / forward (while paused)"),
            ("A / D", "Step backward / forward (while paused, alternate)"),
            ("R", "Toggle reverse playback direction"),
            ("Timeline", "Drag slider to scrub frames"),
        ),
    ),
    (
        "Prompt Tools",
        (145, 160, 90),
        (
            ("Tab / Shift+Tab", "Switch Hover / Box / FG / BG tool (forward / back)"),
            ("Ctrl+Z", "Undo last added prompt (FG/BG point or box)"),
            ("C", "Clear on-screen prompts (not stored memory)"),
            ("Enter", "Store current prompts to selected object (while paused)"),
            ("Hover + move", "Live mask preview (no stored prompts)"),
        ),
    ),
    (
        "Mouse Prompts",
        (110, 175, 145),
        (
            ("Hover L-click", "Add foreground point, switch to FG tool"),
            ("Hover R-click", "Add background point, switch to BG tool"),
            ("Box tool", "Drag to draw a box prompt"),
            ("FG / BG tool", "Click to place prompt points"),
            ("Middle-click", "Select object under cursor (tracked masks)"),
        ),
    ),
    (
        "Objects",
        (145, 120, 70),
        (
            ("Up / Down", "Previous / next object slot"),
            ("W / S", "Previous / next object slot"),
            ("+ / =", "Add object slot"),
            ("-", "Remove selected object slot"),
            ("Mouse wheel", "Scroll object list (33+ objects, over grid)"),
        ),
    ),
    (
        "View & System",
        (130, 95, 165),
        (
            ("[ / ]", "Zoom display out / in"),
            ("H", "Toggle user guide (English / 中文)"),
            ("F1", "Toggle this shortcuts window"),
            ("Q / Esc", "Quit application"),
        ),
    ),
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Rendering helpers


def _draw_rounded_rect(img, x1, y1, x2, y2, color, radius=6):
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r <= 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
        return
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
    cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)


def _measure_key_badge_width(key_text: str, key_drawer: TextDrawer, min_w: int = 54, pad_x: int = 10) -> int:
    txt_w, _, _ = key_drawer.get_text_size(key_text)
    return max(min_w, txt_w + 2 * pad_x)


def _draw_key_badge(img, x, y, key_text: str, key_drawer: TextDrawer, badge_w: int, badge_h: int = 22):
    x2 = x + badge_w
    y2 = y + badge_h
    _draw_rounded_rect(img, x, y, x2, y2, (48, 44, 58), radius=5)
    cv2.rectangle(img, (x, y), (x2, y2), (95, 100, 120), 1, cv2.LINE_AA)
    key_drawer.xy_centered(img[y:y2, x:x2], key_text)


def _render_shortcuts_panel() -> np.ndarray:
    """Build the static shortcuts reference image."""

    panel_w = 560
    margin_x = 22
    key_col_w = 138
    row_h = 28
    section_title_h = 28
    section_gap = 12
    header_pad_x = 12
    header_top = 10
    header_h = 86
    footer_h = 44

    key_drawer = TextDrawer(0.40, thickness=1, color=(235, 238, 245), font=cv2.FONT_HERSHEY_DUPLEX)
    desc_drawer = TextDrawer(0.40, thickness=1, color=(195, 198, 210))
    title_drawer = TextDrawer(0.68, thickness=1, color=(245, 246, 250), font=cv2.FONT_HERSHEY_DUPLEX)
    subtitle_drawer = TextDrawer(0.36, thickness=1, color=(150, 155, 175))
    section_drawer = TextDrawer(0.46, thickness=1, color=(230, 232, 240), font=cv2.FONT_HERSHEY_DUPLEX)
    footer_drawer = TextDrawer(0.34, thickness=1, color=(125, 130, 150))

    body_h = 0
    for _, _, rows in SHORTCUT_SECTIONS:
        body_h += section_title_h + len(rows) * row_h + section_gap

    content_h = header_top + header_h + 8 + body_h + footer_h
    img = linear_gradient_image(content_h, panel_w, start_color=(34, 32, 46), end_color=(52, 46, 62), vertical=True)

    # Header band (text drawn inside header ROI so layout stays aligned)
    header_bottom = header_top + header_h
    _draw_rounded_rect(img, header_pad_x, header_top, panel_w - header_pad_x, header_bottom, (42, 40, 56), radius=10)
    cv2.rectangle(img, (header_pad_x, header_top), (panel_w - header_pad_x, header_bottom), (88, 92, 118), 1, cv2.LINE_AA)

    header_roi = img[header_top:header_bottom, header_pad_x : panel_w - header_pad_x]
    title_drawer.xy_norm(header_roi, "Keyboard Shortcuts", (0.5, 0.36))
    subtitle_drawer.xy_norm(
        header_roi, "Micro Tracker 3 v1.5.0  |  H: user guide  |  F1: shortcuts", (0.5, 0.78)
    )

    y = header_bottom + 10
    desc_x = margin_x + key_col_w + 10

    for section_title, accent_color, rows in SHORTCUT_SECTIONS:
        cv2.rectangle(img, (margin_x, y + 3), (margin_x + 4, y + 21), accent_color, -1, cv2.LINE_AA)
        section_drawer.xy_px(img, section_title, (margin_x + 12, y + 18))
        y += section_title_h

        section_badge_w = max(_measure_key_badge_width(key_text, key_drawer) for key_text, _ in rows)
        section_badge_w = min(section_badge_w, key_col_w)

        for key_text, description in rows:
            _draw_key_badge(img, margin_x, y, key_text, key_drawer, section_badge_w)
            desc_drawer.xy_px(img, description, (desc_x, y + 15))
            y += row_h

        y += section_gap

    footer_top = content_h - footer_h + 4
    cv2.line(img, (margin_x, footer_top), (panel_w - margin_x, footer_top), (70, 68, 82), 1, cv2.LINE_AA)
    footer_roi = img[footer_top + 6 : content_h - 8, margin_x : panel_w - margin_x]
    footer_drawer.xy_norm(footer_roi, "Non-modal window: place beside the main UI while working.", (0.5, 0.5))

    return draw_box_outline(img, color=(72, 76, 96), thickness=1)


# ---------------------------------------------------------------------------------------------------------------------
# %% Window class


class ShortcutsHelpWindow:
    """
    Non-modal OpenCV reference window listing keyboard and mouse shortcuts.
    Toggled via F1 or a GUI button; does not capture input from the main window.
    """

    WINDOW_TITLE = "Micro Tracker 3 - Shortcuts"
    F1_KEYCODES = F1_KEYCODES

    def __init__(self, offset_xy: tuple[int, int] = (40, 40)):
        self._offset_xy = offset_xy
        self._is_visible = False
        self._panel_bgr = _render_shortcuts_panel()

        try:
            cv2.destroyWindow(self.WINDOW_TITLE)
        except cv2.error:
            pass

    def is_visible(self) -> bool:
        return self._is_visible

    def show(self):
        if self._is_visible:
            return self

        cv2.namedWindow(self.WINDOW_TITLE, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.WINDOW_TITLE, self._panel_bgr)
        x, y = self._offset_xy
        try:
            cv2.moveWindow(self.WINDOW_TITLE, x, y)
        except cv2.error:
            pass
        self._is_visible = True
        return self

    def hide(self):
        if not self._is_visible:
            return self
        try:
            cv2.destroyWindow(self.WINDOW_TITLE)
        except cv2.error:
            pass
        self._is_visible = False
        return self

    def toggle(self):
        self.hide() if self._is_visible else self.show()
        return self

    def refresh(self):
        """Re-blit the panel if the window is open (keeps it visible on some platforms)."""
        if self._is_visible:
            cv2.imshow(self.WINDOW_TITLE, self._panel_bgr)
        return self

    def close(self):
        return self.hide()

    def attach_f1_toggle(self, display_window) -> None:
        """Register F1 (all known platform key codes) on the main display window."""

        def _toggle_and_refocus():
            self.toggle()
            display_window.refocus()

        for keycode in self.F1_KEYCODES:
            display_window.attach_keypress_callback(keycode, _toggle_and_refocus)
