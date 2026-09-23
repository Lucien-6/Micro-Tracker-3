#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""One hidden Tk root for the user guide and every dialog. Author: Lucien, 2026-09-23."""


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import tkinter as tk


# ---------------------------------------------------------------------------------------------------------------------
# %% Root


_ROOT: tk.Tk | None = None


def get_tk_root() -> tk.Tk:
    """
    Return the process-wide hidden Tk root.

    A second Tk() becomes the default root and steals StringVar values that
    were created without an explicit master. Dialogs and the user guide must
    share this root.
    """

    global _ROOT
    if _ROOT is None:
        _ROOT = tk.Tk()
        _ROOT.withdraw()
    return _ROOT


def close_tk_root() -> None:
    """Destroy the shared root when the application exits."""

    global _ROOT
    if _ROOT is None:
        return
    try:
        _ROOT.destroy()
    except tk.TclError:
        pass
    _ROOT = None
