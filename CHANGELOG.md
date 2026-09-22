# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] - 2026-09-22

### Added

- TIFF stacks and folders of still images can be opened as frame sources. The export dialog still supplies the frame rate.
- Label TIFF files are named with the source frame index, and `frame_index.csv` records filename, frame index, and time.
- **Retry Metrics** writes the metrics CSV, MSD CSV, and overlay video into a folder whose label images were already saved.
- `--square` forces square image padding and overrides the saved aspect setting.

### Changed

- Overlapping label pixels go to the object with the higher score. Equal scores keep the smaller stable id.
- **Store Prompt** clears that object's frame memory unless `--keep_history_on_new_prompts` is set.
- Display size, object-score threshold, and square/aspect sizing use `.history` only when the command line omits them. An explicit value, including the built-in default, overrides the saved one.

### Fixed

- Object label ids stay fixed when a slot is removed. The removed id is erased from frames already in memory; surviving ids are not renumbered. Lost-target warnings use the same stable id.
- Frame memory is appended only for the next adjacent frame in the same playback direction. A jump or a direction change clears that chain and leaves prompt memory in place.
- A failed metrics or overlay export keeps the in-memory labels and does not report the save as fully successful.
- Releasing the timeline while recording writes that frame.
- Ctrl+C uses the same unsaved-results prompt as closing the window. Cancelling the save leaves the app open.

## [1.5.1] - 2026-06-27

### Changed

- **Mouse interaction documentation** — Accurately documented the per-tool effect of left/right clicks, Shift modifiers, and drag for all three prompt tools (Hover / FG-BG / Box). The previous wording (e.g. "Click to place prompt points") was ambiguous and did not reflect the actual overlay behavior.
  - **Hover** — Left/right clicks always **append** an FG/BG point and switch to the corresponding tool. Shift has no effect in Hover.
  - **FG / BG** — Plain left-click **replaces the last point** (or places the first one); **Shift + left-click appends** a new point (required for multi-point prompts). Right-click deletes the point nearest to the click (Shift-independent).
  - **Box** — Plain left-drag **replaces the last box**; **Shift + left-drag appends** a new box (required for multi-box prompts). Right-click deletes the box whose corner is nearest. A left-click without dragging discards the last box without adding a new one (avoid).
- **Updated in three places** for consistency:
  - `src/demo_helpers/ui/shortcuts_help.py` — F1 shortcuts panel "Mouse Prompts" section expanded from 5 to 11 entries.
  - `src/demo_helpers/ui/user_guide_window.py` — In-app user guide (H) section 3, both English and 中文, now lists per-tool mouse behavior.
  - `docs/USER_GUIDE.md` — Section 3 now includes a full "Mouse interactions per tool" table plus notes on right-click drag semantics, FG/BG left-drag (no intermediate preview), and the auto-switch-back-to-Hover rule (applies to FG/BG only, not Box).

### Fixed

- **Auto-switch-to-Hover note corrected for Box tool** — The previous draft of the new docs over-generalized the auto-switch behavior. Code only auto-switches back to Hover when FG/BG points are removed (`if fg_prompt_changed or bg_prompt_changed:` in `read_prompts`); deleting the last box in the Box tool leaves you on the Box tool. The Markdown note now states this explicitly.

### Notes

- No code logic changed in this release; only documentation strings and Markdown content were updated to match the existing overlay implementations in `src/demo_helpers/ui/overlays.py` and `src/demo_helpers/shared_ui_layout.py`.

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

[1.6.0]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.6.0
[1.5.1]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.5.1
[1.5.0]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.5.0
[1.4.2]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.4.2
[1.4.1]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.4.1
[1.4.0]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.4.0
[1.3.0]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.3.0
[1.2.0]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.2.0
[1.1.0]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.1.0
[1.0.0]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.0.0
