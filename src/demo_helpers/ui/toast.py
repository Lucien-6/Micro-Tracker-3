#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from time import perf_counter

import cv2


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class ToastManager:
    """
    Lightweight, non-blocking on-screen notification ('toast').

    A single most-recent message is drawn near the top-center of the display
    for a short duration and then fades out. This is meant for transient,
    non-critical feedback (e.g. "Prompt stored", "Target lost"). Critical
    errors/failures should use a modal dialog instead.

    Usage:
        toast = ToastManager()
        toast.notify("Prompt stored", level="success")
        ...
        toast.draw(display_image)   # call right before window.show(...)
    """

    # Styling per level: (background BGR, accent BGR)
    _LEVEL_STYLES = {
        "info": ((48, 42, 38), (210, 180, 90)),
        "success": ((30, 58, 32), (90, 205, 110)),
        "warning": ((38, 48, 66), (60, 175, 240)),
    }
    _TEXT_COLOR = (240, 240, 240)

    # .................................................................................................................

    def __init__(self, text_scale=0.55, thickness=1, top_margin_frac=0.045, fade_sec=0.45):
        self._message = None
        self._expiry = 0.0
        self._duration_sec = 2.5
        self._bg_color = self._LEVEL_STYLES["info"][0]
        self._accent_color = self._LEVEL_STYLES["info"][1]

        self._scale = text_scale
        self._thick = thickness
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._top_margin_frac = top_margin_frac
        self._fade_sec = max(fade_sec, 1e-3)

    # .................................................................................................................

    def notify(self, message: str, level: str = "info", duration_sec: float = 2.5):
        """Queue a transient message. Replaces any currently displayed message."""
        self._message = str(message)
        self._duration_sec = float(duration_sec)
        self._expiry = perf_counter() + self._duration_sec
        self._bg_color, self._accent_color = self._LEVEL_STYLES.get(level, self._LEVEL_STYLES["info"])
        return self

    # .................................................................................................................

    def clear(self):
        self._message = None
        return self

    # .................................................................................................................

    def draw(self, frame, region_xyxy=None):
        """
        Draw the active toast (if any) onto the given BGR frame, in place.

        If 'region_xyxy' (x1, y1, x2, y2) is provided, the toast is centered
        within that sub-region (e.g. the video display area); otherwise it is
        centered near the top of the full frame.
        """

        if self._message is None:
            return frame

        remaining = self._expiry - perf_counter()
        if remaining <= 0.0:
            self._message = None
            return frame

        # Fade out near the end of the display window
        alpha = 1.0 if remaining > self._fade_sec else max(0.0, remaining / self._fade_sec)
        blend = 0.9 * alpha

        frame_h, frame_w = frame.shape[0:2]

        # Resolve the target region (defaults to the full frame)
        if region_xyxy is not None:
            rx1, ry1, rx2, ry2 = (int(v) for v in region_xyxy)
            rx1 = max(0, min(rx1, frame_w - 1))
            ry1 = max(0, min(ry1, frame_h - 1))
            rx2 = max(rx1 + 1, min(rx2, frame_w))
            ry2 = max(ry1 + 1, min(ry2, frame_h))
            center_vertically = True
        else:
            rx1, ry1, rx2, ry2 = 0, 0, frame_w, frame_h
            center_vertically = False
        region_w = rx2 - rx1
        region_h = ry2 - ry1

        (text_w, text_h), baseline = cv2.getTextSize(self._message, self._font, self._scale, self._thick)
        pad_x, pad_y = 16, 11
        box_w = min(text_w + 2 * pad_x, region_w - 4)
        box_h = text_h + baseline + 2 * pad_y

        x1 = rx1 + max((region_w - box_w) // 2, 2)
        if center_vertically:
            y1 = ry1 + max((region_h - box_h) // 2, 2)
        else:
            y1 = ry1 + max(int(self._top_margin_frac * region_h), 2)
        x2 = min(x1 + box_w, frame_w - 1)
        y2 = min(y1 + box_h, frame_h - 1)
        if x2 <= x1 or y2 <= y1:
            return frame

        # Build the panel separately so we only alpha-blend the affected region
        sub = frame[y1:y2, x1:x2]
        panel = sub.copy()
        ph, pw = panel.shape[0:2]
        cv2.rectangle(panel, (0, 0), (pw - 1, ph - 1), self._bg_color, -1)
        cv2.rectangle(panel, (0, 0), (4, ph - 1), self._accent_color, -1)
        cv2.putText(
            panel,
            self._message,
            (pad_x, pad_y + text_h),
            self._font,
            self._scale,
            self._TEXT_COLOR,
            self._thick,
            cv2.LINE_AA,
        )
        cv2.addWeighted(panel, blend, sub, 1.0 - blend, 0.0, dst=sub)

        return frame

    # .................................................................................................................
