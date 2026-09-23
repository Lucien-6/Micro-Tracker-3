#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os
import os.path as osp


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions

# .....................................................................................................................


def clean_path_str(path=None):
    """
    Helper used to interpret user-given paths correctly.
    Important for Windows, since 'copy path' in File Explorer wraps the path in quotes.
    Only a matching pair of quotes around the whole path is removed, so names
    such as "Lucien's data" stay intact.
    """

    path_str = "" if path is None else str(path).strip()
    if len(path_str) >= 2 and path_str[0] == path_str[-1] and path_str[0] in "\"'":
        path_str = path_str[1:-1].strip()
    return osp.expanduser(path_str)


# .....................................................................................................................


def resolve_model_candidates(file_dunder, model_path=None, history_path=None) -> list[str]:
    """
    Ordered SAM weight candidates, without interactive prompts.

    An explicit -m value is used on its own: an existing file, or the unique
    file in model_weights/ whose name contains that text. It is an error when
    that value matches nothing, instead of silently falling back to history.

    With no -m, the history path comes first and the other weight files follow
    in name order, so a bad history entry can be skipped.
    """

    explicit_path = clean_path_str(model_path)
    model_file_paths = get_model_weights_paths(file_dunder)
    names = ", ".join(osp.basename(path) for path in sorted(model_file_paths)) or "(none)"

    if explicit_path:
        if osp.isfile(explicit_path):
            return [explicit_path]
        matched = [path for path in model_file_paths if explicit_path in osp.basename(path)]
        if len(matched) == 1:
            return matched
        if len(matched) > 1:
            matched_names = ", ".join(osp.basename(path) for path in sorted(matched))
            raise FileNotFoundError(f"Model selector {explicit_path!r} matches more than one file: {matched_names}")
        raise FileNotFoundError(f"No SAM weights match {explicit_path!r}. Available: {names}")

    candidates = []
    stored_path = clean_path_str(history_path) if history_path else ""
    if stored_path and osp.isfile(stored_path):
        candidates.append(stored_path)
    for path in sorted(model_file_paths):
        if path not in candidates:
            candidates.append(path)
    if len(candidates) == 0:
        raise FileNotFoundError(
            "No SAM model weights found. Place a .pt/.pth file in the model_weights folder "
            "or pass -m / --model_path, then restart."
        )
    return candidates


def resolve_default_model_path(file_dunder, model_path=None, history_path=None) -> str:
    """First entry from resolve_model_candidates."""

    return resolve_model_candidates(file_dunder, model_path, history_path)[0]


# .....................................................................................................................


def get_model_weights_paths(file_dunder, model_weights_folder_name="model_weights"):

    # Build path to model weight folder (and create if missing)
    script_caller_folder_path = osp.dirname(file_dunder) if osp.isfile(file_dunder) else file_dunder
    model_weights_path = osp.join(script_caller_folder_path, model_weights_folder_name)
    os.makedirs(model_weights_path, exist_ok=True)

    # Get only the paths to files with specific extensions
    valid_exts = {".pt", ".pth"}
    all_files_list = os.listdir(model_weights_path)
    model_files_list = [file for file in all_files_list if osp.splitext(file)[1].lower() in valid_exts]
    model_file_paths = [osp.join(model_weights_path, file) for file in model_files_list]

    return model_file_paths


# .....................................................................................................................


def pick_file_path(
    title: str,
    filetypes: list[tuple[str, str]],
    initial_path: str | None = None,
) -> str | None:
    """
    Open a native file-picker dialog and return the selected path.
    Returns None if the user cancels or no valid path is chosen.
    """

    try:
        from tkinter import filedialog

        from .tk_host import get_tk_root
    except ImportError:
        print("", "Warning: tkinter unavailable, cannot open file picker.", sep="\n", flush=True)
        return None

    initialdir = None
    if initial_path is not None:
        initial_path = clean_path_str(initial_path)
        if osp.isdir(initial_path):
            initialdir = initial_path
        elif osp.isfile(initial_path):
            initialdir = osp.dirname(initial_path)

    root = get_tk_root()
    root.attributes("-topmost", True)
    selected_path = filedialog.askopenfilename(title=title, filetypes=filetypes, initialdir=initialdir, parent=root)
    root.attributes("-topmost", False)

    selected_path = clean_path_str(selected_path)
    if selected_path == "" or not osp.exists(selected_path):
        return None

    return selected_path


# .....................................................................................................................


def pick_save_path(
    title: str,
    filetypes: list[tuple[str, str]],
    initial_path: str | None = None,
    default_name: str = "",
) -> str | None:
    """Ask for a new file path. Returns None when the user cancels."""

    try:
        from tkinter import filedialog

        from .tk_host import get_tk_root
    except ImportError:
        print("", "Warning: tkinter unavailable, cannot open save dialog.", sep="\n", flush=True)
        return None

    initialdir = None
    if initial_path is not None:
        initial_path = clean_path_str(initial_path)
        if osp.isdir(initial_path):
            initialdir = initial_path
        elif osp.isfile(initial_path):
            initialdir = osp.dirname(initial_path)
    root = get_tk_root()
    root.attributes("-topmost", True)
    selected = filedialog.asksaveasfilename(
        title=title,
        filetypes=filetypes,
        initialdir=initialdir,
        initialfile=default_name,
        defaultextension=".json",
        parent=root,
    )
    root.attributes("-topmost", False)
    selected = clean_path_str(selected)
    return selected or None


def pick_model_file(file_dunder, current_path: str | None = None) -> str | None:
    """Browse for a SAM model weights file (.pt / .pth)."""

    model_file_paths = get_model_weights_paths(file_dunder)
    initial_path = current_path
    if initial_path is None and len(model_file_paths) > 0:
        initial_path = osp.dirname(model_file_paths[0])

    return pick_file_path(
        "Select model weights",
        [("PyTorch weights", "*.pt *.pth"), ("All files", "*.*")],
        initial_path,
    )


# .....................................................................................................................


def pick_video_file(current_path: str | None = None) -> str | None:
    """Browse for a video file, TIFF stack, or still image."""

    return pick_file_path(
        "Select video or image",
        [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("Image stacks and stills", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.webp"),
            ("All files", "*.*"),
        ],
        current_path,
    )


def pick_image_folder(current_path: str | None = None) -> str | None:
    """Browse for a folder of still images, in filename order."""

    try:
        from tkinter import filedialog

        from .tk_host import get_tk_root
    except ImportError:
        print("", "Warning: tkinter unavailable, cannot open folder picker.", sep="\n", flush=True)
        return None

    initialdir = None
    if current_path is not None:
        current_path = clean_path_str(current_path)
        if osp.isdir(current_path):
            initialdir = current_path
        elif osp.isfile(current_path):
            initialdir = osp.dirname(current_path)

    root = get_tk_root()
    root.attributes("-topmost", True)
    selected_path = filedialog.askdirectory(title="Select image sequence folder", initialdir=initialdir, parent=root)
    root.attributes("-topmost", False)

    selected_path = clean_path_str(selected_path)
    if selected_path == "" or not osp.isdir(selected_path):
        return None
    return selected_path


def pick_frame_source(current_path: str | None = None) -> str | None:
    """
    Ask whether to open a video/TIFF file or a folder of stills, then show that picker.

    Returns None if the user cancels.
    """

    try:
        import tkinter as tk

        from .tk_host import get_tk_root
    except ImportError:
        return pick_video_file(current_path)

    choice: dict[str, str] = {}
    root = get_tk_root()
    dialog = tk.Toplevel(root)
    dialog.title("Open frames")
    dialog.attributes("-topmost", True)
    dialog.resizable(False, False)
    tk.Label(dialog, text="Open a video, a TIFF stack, or a folder of still images.").grid(
        row=0, column=0, columnspan=2, padx=12, pady=(12, 8)
    )

    def choose(kind: str):
        choice["kind"] = kind
        dialog.destroy()

    button_row = tk.Frame(dialog)
    button_row.grid(row=1, column=0, columnspan=2, pady=(0, 12))
    tk.Button(button_row, text="Video or TIFF file", width=18, command=lambda: choose("file")).pack(side="left", padx=6)
    tk.Button(button_row, text="Image folder", width=18, command=lambda: choose("folder")).pack(side="left", padx=6)
    dialog.bind("<Escape>", lambda _event: choose("cancel"))
    dialog.protocol("WM_DELETE_WINDOW", lambda: choose("cancel"))
    dialog.grab_set()
    root.wait_window(dialog)

    kind = choice.get("kind")
    if kind == "folder":
        return pick_image_folder(current_path)
    if kind == "file":
        return pick_video_file(current_path)
    return None


def resolve_startup_settings(
    display_size_arg: int | None,
    objscore_arg: float | None,
    use_aspect_ratio: bool,
    force_square: bool,
    default_display_size: int,
    default_objscore: float,
    history_display: int | None,
    history_objscore: float | None,
    history_square: bool | None,
) -> tuple[int, float, bool]:
    """
    Resolve display size, object-score threshold, and square sizing.

    A value passed on the command line is used as given, including the built-in
    default (0.0, 900, or an explicit square/aspect choice). History is used
    only when that setting was omitted.
    """

    if force_square and use_aspect_ratio:
        raise ValueError("Use only one of --square and --use_aspect_ratio.")

    if display_size_arg is None:
        display_size = default_display_size
        if isinstance(history_display, int) and history_display > 0:
            display_size = history_display
    else:
        display_size = int(display_size_arg)

    if objscore_arg is None:
        objscore = float(default_objscore)
        if isinstance(history_objscore, (int, float)):
            objscore = float(history_objscore)
    else:
        objscore = float(objscore_arg)

    if force_square:
        use_square = True
    elif use_aspect_ratio:
        use_square = False
    elif isinstance(history_square, bool):
        use_square = history_square
    else:
        use_square = True

    return display_size, objscore, use_square


# .....................................................................................................................


def pick_save_folder(initial_path: str | None = None) -> str | None:
    """
    Open a native folder-picker dialog and return the selected directory.
    Returns None if the user cancels or no valid path is chosen.
    """

    try:
        from tkinter import filedialog

        from .tk_host import get_tk_root
    except ImportError:
        print("", "Warning: tkinter unavailable, cannot open folder picker.", sep="\n", flush=True)
        return None

    initialdir = None
    if initial_path is not None:
        initial_path = clean_path_str(initial_path)
        if osp.isdir(initial_path):
            initialdir = initial_path
        elif osp.isfile(initial_path):
            initialdir = osp.dirname(initial_path)

    root = get_tk_root()
    root.attributes("-topmost", True)
    selected_path = filedialog.askdirectory(title="Select results output folder", initialdir=initialdir, parent=root)
    root.attributes("-topmost", False)

    selected_path = clean_path_str(selected_path)
    if selected_path == "" or not osp.isdir(selected_path):
        return None

    return selected_path


# .....................................................................................................................


def ask_export_parameters(
    default_fps: float = 30.0,
    default_pixel_size_um: float = 1.0,
) -> tuple[float, float] | None:
    """
    Prompt the user for the physical parameters needed to compute tracking metrics:
    the video frame rate (FPS) and the pixel size (micrometers per pixel).

    Returns (fps, pixel_size_um) or None if the dialog is cancelled/unavailable.
    """

    try:
        import tkinter as tk
        from tkinter import messagebox

        from .tk_host import get_tk_root
    except ImportError:
        print("", "Warning: tkinter unavailable, using default export parameters.", sep="\n", flush=True)
        return float(default_fps), float(default_pixel_size_um)

    result: dict[str, float] = {}
    root = get_tk_root()
    dialog = tk.Toplevel(root)
    dialog.title("Export Parameters")
    dialog.attributes("-topmost", True)
    dialog.resizable(False, False)

    tk.Label(dialog, text="Frame rate (FPS):").grid(row=0, column=0, padx=10, pady=(12, 4), sticky="w")
    fps_var = tk.StringVar(master=dialog, value=f"{float(default_fps):g}")
    fps_entry = tk.Entry(dialog, textvariable=fps_var, width=16)
    fps_entry.grid(row=0, column=1, padx=10, pady=(12, 4))

    tk.Label(dialog, text="Pixel size (um/pixel):").grid(row=1, column=0, padx=10, pady=4, sticky="w")
    pixel_var = tk.StringVar(master=dialog, value=f"{float(default_pixel_size_um):g}")
    pixel_entry = tk.Entry(dialog, textvariable=pixel_var, width=16)
    pixel_entry.grid(row=1, column=1, padx=10, pady=4)

    tk.Label(dialog, text="Max frame gap for velocity:").grid(row=2, column=0, padx=10, pady=4, sticky="w")
    gap_var = tk.StringVar(master=dialog, value="1")
    gap_entry = tk.Entry(dialog, textvariable=gap_var, width=16)
    gap_entry.grid(row=2, column=1, padx=10, pady=4)

    def on_ok():
        try:
            fps_val = float(fps_var.get())
            pixel_val = float(pixel_var.get())
            gap_val = int(float(gap_var.get()))
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Enter numeric values for FPS, pixel size, and frame gap.",
                parent=dialog,
            )
            return
        if fps_val <= 0 or pixel_val <= 0 or gap_val < 1:
            messagebox.showerror(
                "Invalid Input",
                "FPS and pixel size must be greater than 0, and the frame gap must be at least 1.",
                parent=dialog,
            )
            return
        result["fps"] = fps_val
        result["pixel_size_um"] = pixel_val
        result["max_frame_gap"] = gap_val
        dialog.destroy()

    def on_cancel():
        dialog.destroy()

    button_row = tk.Frame(dialog)
    button_row.grid(row=3, column=0, columnspan=2, pady=(8, 12))
    tk.Button(button_row, text="OK", width=10, command=on_ok).pack(side="left", padx=6)
    tk.Button(button_row, text="Cancel", width=10, command=on_cancel).pack(side="left", padx=6)

    dialog.bind("<Return>", lambda _event: on_ok())
    dialog.bind("<Escape>", lambda _event: on_cancel())
    dialog.protocol("WM_DELETE_WINDOW", on_cancel)
    fps_entry.focus_set()
    dialog.grab_set()
    root.wait_window(dialog)

    if "fps" not in result:
        return None
    return result["fps"], result["pixel_size_um"], result["max_frame_gap"]


# .....................................................................................................................


def show_message_dialog(title: str, message: str, kind: str = "error") -> None:
    """
    Show a modal message dialog for important feedback (errors/failures).

    'kind' may be one of: 'error', 'warning', 'info'. Falls back to a terminal
    message if tkinter is unavailable.
    """

    try:
        from tkinter import messagebox

        from .tk_host import get_tk_root
    except ImportError:
        print("", f"{title}: {message}", sep="\n", flush=True)
        return

    root = get_tk_root()
    root.attributes("-topmost", True)
    if kind == "warning":
        messagebox.showwarning(title, message, parent=root)
    elif kind == "info":
        messagebox.showinfo(title, message, parent=root)
    else:
        messagebox.showerror(title, message, parent=root)
    root.attributes("-topmost", False)
    return


# .....................................................................................................................


def ask_save_unsaved_results() -> bool | None:
    """
    Ask the user whether unsaved tracking results should be saved before closing.
    Returns True to save, False to discard, or None if the dialog is unavailable.
    """

    try:
        from tkinter import messagebox

        from .tk_host import get_tk_root
    except ImportError:
        print("", "Warning: tkinter unavailable, cannot show save confirmation dialog.", sep="\n", flush=True)
        return None

    root = get_tk_root()
    root.attributes("-topmost", True)
    should_save = messagebox.askyesno(
        "Unsaved Results",
        "There are unsaved tracking results in memory.\n\nDo you want to save them before closing?",
        parent=root,
    )
    root.attributes("-topmost", False)
    return should_save


def ask_yes_no(title: str, message: str) -> bool | None:
    """Return True for yes, False for no, or None when the dialog is unavailable or cancelled."""

    try:
        from tkinter import messagebox

        from .tk_host import get_tk_root
    except ImportError:
        print("", f"{title}: {message}", sep="\n", flush=True)
        return None

    root = get_tk_root()
    root.attributes("-topmost", True)
    answer = messagebox.askyesno(title, message, parent=root)
    root.attributes("-topmost", False)
    return bool(answer)


def parse_intensity_range(text: str) -> str | tuple[float, float]:
    """Parse auto, full, or low,high for 16-bit stills."""

    cleaned = str(text).strip().lower()
    if cleaned in ("auto", "full"):
        return cleaned
    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) != 2:
        raise ValueError("intensity range must be auto, full, or low,high")
    low, high = float(parts[0]), float(parts[1])
    if high <= low:
        raise ValueError("intensity range high must be greater than low")
    return low, high


def ask_save_discard_cancel(message: str) -> str | None:
    """
    Ask whether to save, discard, or cancel.

    Returns "save", "discard", "cancel", or None when tkinter is unavailable.
    """

    try:
        from tkinter import messagebox

        from .tk_host import get_tk_root
    except ImportError:
        print("", "Warning: tkinter unavailable, cannot show save confirmation dialog.", sep="\n", flush=True)
        return None

    root = get_tk_root()
    root.attributes("-topmost", True)
    answer = messagebox.askyesnocancel("Unsaved Results", message, parent=root)
    root.attributes("-topmost", False)
    if answer is None:
        return "cancel"
    return "save" if answer else "discard"
