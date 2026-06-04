#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from time import perf_counter

import cv2
import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class ProgressWindow:
    """
    Lightweight modal-style progress window built on OpenCV HighGUI.

    It shows a single labelled progress bar and is meant to be driven from the
    main (GUI) thread while a blocking task runs. Call 'update(...)' as work
    proceeds and 'close()' when finished. Redraws are throttled so frequent
    updates do not slow down the underlying task.

    Usage:
        progress = ProgressWindow("Saving Results")
        try:
            progress.update("Saving label images", 0.0)
            ...
            progress.update("Rendering overlay video", 0.5)
        finally:
            progress.close()
    """

    _BG_COLOR = (32, 28, 30)
    _BAR_BG_COLOR = (64, 58, 62)
    _BAR_FG_COLOR = (110, 190, 120)
    _BORDER_COLOR = (120, 120, 130)
    _TEXT_COLOR = (235, 235, 235)
    _SUBTEXT_COLOR = (190, 190, 195)

    # .................................................................................................................

    def __init__(self, title: str = "Working...", width: int = 540, height: int = 150, min_redraw_sec: float = 0.04):
        self._title = str(title)
        self._w = int(width)
        self._h = int(height)
        self._min_redraw_sec = float(min_redraw_sec)
        self._last_draw = 0.0
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._closed = False

        cv2.namedWindow(self._title, cv2.WINDOW_AUTOSIZE)
        self.update("Preparing...", 0.0, force=True)

    # .................................................................................................................

    def update(self, message: str, fraction: float, force: bool = False):
        """Redraw the progress bar with the given message and fraction (0..1)."""

        if self._closed:
            return self

        now = perf_counter()
        frac = max(0.0, min(1.0, float(fraction)))
        # Always honor completion / forced redraws; otherwise throttle to keep the task fast
        if not force and frac < 1.0 and (now - self._last_draw) < self._min_redraw_sec:
            return self
        self._last_draw = now

        img = np.full((self._h, self._w, 3), self._BG_COLOR, dtype=np.uint8)

        # Status message
        cv2.putText(img, str(message), (20, 42), self._font, 0.6, self._TEXT_COLOR, 1, cv2.LINE_AA)

        # Progress bar background
        bx0, by0, bx1, by1 = 20, 66, self._w - 20, 98
        cv2.rectangle(img, (bx0, by0), (bx1, by1), self._BAR_BG_COLOR, -1)
        fill_x = bx0 + int(round((bx1 - bx0) * frac))
        if fill_x > bx0:
            cv2.rectangle(img, (bx0, by0), (fill_x, by1), self._BAR_FG_COLOR, -1)
        cv2.rectangle(img, (bx0, by0), (bx1, by1), self._BORDER_COLOR, 1)

        # Percentage label
        pct_text = f"{frac * 100.0:.0f}%"
        (tw, _), _ = cv2.getTextSize(pct_text, self._font, 0.55, 1)
        cv2.putText(img, pct_text, ((bx0 + bx1) // 2 - tw // 2, by1 + 30), self._font, 0.55, self._SUBTEXT_COLOR, 1, cv2.LINE_AA)

        cv2.imshow(self._title, img)
        cv2.waitKey(1)
        return self

    # .................................................................................................................

    def close(self):
        if self._closed:
            return self
        self._closed = True
        try:
            cv2.destroyWindow(self._title)
            cv2.waitKey(1)
        except cv2.error:
            pass
        return self

    # .................................................................................................................

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    # .................................................................................................................
