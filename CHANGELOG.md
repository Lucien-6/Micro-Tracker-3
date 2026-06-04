# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2026-06-04

### Added

- **On-screen toast notifications** (`src/demo_helpers/ui/toast.py`) — Transient, non-blocking feedback for model/video load, prompt store/clear, object add/remove, history toggle, and **lost-target warnings**. Toasts are drawn **centered in the video display area** and fade out automatically. Errors and failures still use modal dialogs.
- **Modal message dialogs** (`show_message_dialog` in `src/demo_helpers/loading.py`) — Error/warning popups for save and metric/overlay export failures.
- **Ctrl+Z prompt undo** — Removes the most recently added foreground/background point or box for the active object (`PromptUndoManager`).
- **Tracking analysis export** (`src/demo_helpers/analysis.py`) — Saved alongside the label sequence:
  - `tracking_metrics.csv` — per-frame, per-object centroid, area, orientation angle (long axis of the fitted ellipse vs. +X, from image moments), velocity, and displacement.
  - `tracking_msd.csv` — time-averaged Mean Squared Displacement per object.
  - `tracking_overlay.mp4` — single overlay video with a uniquely colored contour per object and a fading ~20-frame trajectory. Trajectories are hidden once an object leaves the field of view.
- **Export parameter dialog** (`ask_export_parameters`) — Prompts for **frame rate (fps)** and **pixel size (µm/pixel)** before export; the pixel size is remembered between sessions.
- **Save progress window** (`src/demo_helpers/ui/progress.py`) — Live progress bar shown while saving label images and rendering metrics/overlay video.
- **Expanded session persistence** (`.history`) — Now also restores display size, last save folder, **Enable History** state, `--objscore_threshold`, square/aspect sizing, and pixel size, when not overridden on the command line.
- **CPU image-encoding LRU cache** (`src/demo_helpers/encode_cache.py`, `--encode_cache_size`, default 64) — Reuses image encodings to speed up scrubbing, frame stepping, and reverse playback.
- **Reverse-playback frame buffer** (`--reverse_buffer_size`, default 120) — Buffers decoded frames for smoother reverse playback and stepping.

### Fixed

- **Model switch without a loaded video** — No longer crashes with `AttributeError: 'NoneType' object has no attribute 'pause'`; the playback reader is only paused when a video is loaded.

### Notes

- `tracking_overlay.mp4` and the metric CSVs are written only when a source video is available; metric units depend on the fps and µm/pixel values entered in the export dialog.

## [1.4.2] - 2026-05-31

### Changed

- **Keyboard shortcuts (annotation workflow)** — Remapped for faster hands-on-video use:
  - **Enter** — Store Prompt while paused (replaces Tab).
  - **Tab / Shift+Tab** — Switch prompt tool forward / backward (Hover / Box / FG / BG; replaces ← / →).
  - **← / →** — Step one frame backward / forward while paused; **A / D** remain as alternate step keys.
- **Store Prompt button** — Label shows `(Enter)`; Enter is ignored during playback.
- **Documentation** — README, `docs/USER_GUIDE.md`, F1 shortcuts panel, and in-app user guide (H) updated to match.

## [1.4.1] - 2026-05-29

### Fixed

- **Object sidebar hit targets** — Button click and hover regions now stay aligned with on-screen `Object N` labels after layout render and when scrolling (fixes spurious hover while the pointer is over the video, and missed clicks in the sidebar).
- **`ScrollableGridViewport.finalize_callback_regions()`** — Re-syncs grid button regions from content-relative coordinates using the parent sidebar’s final screen position; off-screen slots are disabled instead of sharing a single corner hit box.

## [1.4.0] - 2026-05-29

### Added

- **Up to 255 object slots** (was 32), each with its own prompt memory bank and tracking state.
- **Scrollable object sidebar** — When more than 32 objects are defined, the right-hand two-column object list keeps the same button row height as at 32 objects; use the **mouse wheel** over the object grid to scroll. Recording, Add/Remove, and Save/Clear controls stay fixed outside the scroll area.
- **Active object visibility** — Switching the selected object (sidebar, W/S, ↑/↓, or middle-click on a mask) scrolls the list so the active slot stays in view.

### Changed

- **`ScrollableGridViewport`** (`src/demo_helpers/ui/layout.py`) — Wraps the object `GridStack` for large slot counts.
- **`HStack` layout** — Row height uses the tallest child instead of vertically cropping taller sidebars (fixes compressed object buttons after enabling scroll mode).
- **`CBEventFlags.wheel_delta`** — Mouse wheel delta is passed to UI callbacks for object-list scrolling.

### Notes

- Combined label export still uses 8-bit gray values 1–255 per object index; very large slot counts increase per-frame UI render cost and multi-object inference time.

## [1.3.0] - 2026-05-28

### Changed

- **Package layout** — Renamed directory `muggled_sam/` to `src/`. Application imports are now `from src...` (e.g. `from src.make_sam import make_sam_from_state_dict`). Forks or scripts that imported `muggled_sam` must update paths accordingly.
- Added `src/__init__.py` to mark the inference/UI package root.

## [1.2.0] - 2026-05-28

### Added

- **In-app user guide (tkinter)** — Press **H** to open a scrollable, non-modal guide with **English / 中文** switching (`src/demo_helpers/ui/user_guide_window.py` after 1.3.0; was `muggled_sam/...` in 1.2.0).
- **F1 shortcuts panel** — Subtitle references **H** (user guide) and **F1** (keyboard shortcuts).

### Fixed

- **Windows GIL crash** when pressing **H** during `cv2.waitKeyEx` — Guide toggle is deferred to the main loop (`request_toggle` + `process_events` after `DisplayWindow.show()`).

## [1.1.0] - 2026-05-28

### Changed

- **`--keep_bad_objscores`** — Now controls whether video masking continues after a low object score (lost target). Default behavior stops `step_video_masking` for that object from the first loss frame onward (mask still zeroed on low-score frames). With the flag set, inference continues for possible auto-recovery while masks remain zeroed during loss.
- **Per-object loss handling** — Each object slot records a `tracking_stop_frame_idx` when lost (default mode). Inference resumes if the playhead moves to an earlier frame; stopped objects are skipped at or after the loss frame until cleared.
- **Store Prompt** — Requires interactive prompts (foreground/background points or a box; hover-only on already-tracked objects is not accepted). Successful store clears the loss stop marker for that object (revival after loss).

### Added

- `docs/USER_GUIDE.md` — Markdown user guide (workflow, lost-target behavior, CLI reference)

## [1.0.0] - 2026-05-28

First stable release of **Micro Tracker 3**.

### Added

- Interactive video segmentation and tracking GUI built on SAM 2 / SAM 3 / SAM 3.1 (`muggled_sam` inference bundle)
- Multi-object tracking with up to 32 independent object slots
- Prompt tools: hover preview, bounding box, foreground/background points
- Per-object prompt memory and temporal frame memory banks
- Forward and reverse video playback with timeline scrubbing
- Combined 8-bit grayscale label export as a TIF image sequence (`*_MT-Results_*`)
- GUI resource switcher for model and video files
- Keyboard shortcuts reference panel (F1)
- VRAM usage display and configurable inference settings
- Automatic SAM weight version detection (v2, v3, v3.1)
- Session history file (`.history`) for last-used model and video paths

### Notes

- Model weights are **not** bundled; place `.pt` / `.pth` files in `model_weights/`
- Inference backend: PyTorch with CUDA, Apple MPS, or CPU

[1.5.0]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.5.0
[1.4.2]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.4.2
[1.4.1]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.4.1
[1.4.0]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.4.0
[1.3.0]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.3.0
[1.2.0]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.2.0
[1.1.0]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.1.0
[1.0.0]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.0.0
