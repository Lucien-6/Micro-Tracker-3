# Micro Tracker 3 — User Guide

**Version:** 1.8.2  
**Last updated:** 2026-09-23  
**Author:** Lucien · [lucien-6@qq.com](mailto:lucien-6@qq.com)

This document mirrors the in-app guide (press **H** while the main window is focused; switch **English / 中文** in the guide window). For installation and repository layout, see [README.md](../README.md).

---

## 1. Overview

Micro Tracker 3 is an interactive desktop tool for:

1. Annotating objects in video frames with SAM (points, boxes).
2. Propagating instance masks through time (tracking).
3. Exporting combined **8-bit label TIF** sequences for downstream analysis (ImageJ, TrackMate, custom pipelines).

The app supports up to **255 object slots**, each with its own prompt memory and optional frame history.

**Code layout (v1.3.0+):** SAM inference and UI helpers live under `src/` (formerly `muggled_sam/`). `python main.py` resolves `src` from the folder that contains `main.py`. Extend the app with `from src...` imports.

---

## 2. Typical workflow

| Step | Action |
| --- | --- |
| 1 | Load **model** and **video** (GUI buttons or `-m` / `-i`). |
| 2 | Select an **object slot** (Object 1, 2, …). |
| 3 | **Pause** the video (Space). |
| 4 | Choose a prompt tool (Hover / Box / FG / BG; **Tab / Shift+Tab** to switch) and annotate the target. |
| 5 | **Store Prompt** (Enter or button, while paused) — writes prompts into the tracker memory bank. |
| 6 | **Track** — play forward (Space / Track); masks update each new frame. |
| 7 | Optional: **Enable Recording**, then **Save Results** for TIF export. |

Repeat steps 2–5 for additional objects before tracking.

**Important:** Prompt tools are disabled during playback. Pause before annotating or storing prompts.

---

## 3. Prompt tools

| Tool | Use |
| --- | --- |
| **Hover** | Live mask preview when the slot has **no** stored prompts yet. |
| **Box** | Drag a rectangle around the target. |
| **FG Point** | Foreground clicks (include region). |
| **BG Point** | Background clicks (exclude region). |

### Mouse interactions per tool

The effect of left/right clicks and Shift depends on the active tool. Only the selected tool's overlay responds; other tools ignore mouse input.

| Tool | Input | Effect |
| --- | --- | --- |
| **Hover** | Move | Live mask preview at the cursor (only when the slot has no stored prompts). |
| **Hover** | Left-click | **Append** one FG point and switch to the FG tool. Shift has no effect. |
| **Hover** | Right-click | **Append** one BG point and switch to the BG tool. Shift has no effect. |
| **FG / BG** | Left-click | If no points exist, place the first one. Otherwise **replace the last point** (moves it to the click position). |
| **FG / BG** | Shift + Left-click | **Append** a new point. Use this to build multi-point prompts (e.g. 1 FG + N BG). |
| **FG / BG** | Right-click | **Delete** the point nearest to the click (Euclidean distance, in pixels). Shift has no effect. |
| **Box** | Left-drag | **Replace** the last box with the newly drawn box. |
| **Box** | Shift + Left-drag | **Append** a new box. Use this for multi-box prompts. |
| **Box** | Right-click | **Delete** the box whose corner is nearest to the click. Shift has no effect. |
| **Box** | Left-click without dragging | Discards the last box and adds nothing (the new box is too small to be kept). Avoid this input. |
| Any | Middle-click | Select the object under the cursor (operates on tracked masks, independent of the prompt tool). |

Notes:

- Right-click drag is treated as a right-click (release inside the region); there is no separate drag semantic for right-button actions.
- For FG/BG tools, left-drag has no intermediate preview — the point is applied on release using the rules above.
- When all FG/BG points are removed (e.g. by right-click in the FG/BG tool), the tool auto-switches back to Hover. Removing the last box in the Box tool does **not** auto-switch — you stay on the Box tool.

- **Tab / Shift+Tab** — Switch prompt tool forward / backward (Hover → Box → FG → BG).
- **Ctrl+Z** — Undo the most recently added prompt (one FG/BG point or box) for the active object, before it is stored.
- **C** — Clears on-screen prompts only (does not remove stored tracker memory).
- **Enter / Store Prompt** — Saves current interactive prompts to the **selected** object slot (while paused only).

### Store Prompt requirements

Store Prompt only runs when you have **interactive** prompts on the current frame:

- At least one foreground/background point **or** a box, and  
- Not **hover-only** prompts on an object that already has stored prompts.

If Store is ignored, add FG/BG points or a box (switch off Hover for tracked objects), then try again.

---

## 4. Tracking and temporal memory

- **Enable History** (default on) — Appends a frame-memory encoding only when playback steps exactly one frame in the same direction. A jump or a direction change clears that chain and leaves the stored prompts in place.
- **Clear History** — Removes frame memory for the selected object; prompt memory remains.
- **Store Prompt** clears that object's frame memory unless `--keep_history_on_new_prompts` is set.
- **Clear Prompts** — Clears stored prompt memory for the selected object (on-screen prompts cleared separately with **C**).

Tracking runs for every object slot that has **stored prompts**, on each new frame during playback, keyboard step, or timeline scrub release.

---

## 5. Lost targets (out of frame, defocus, occlusion)

SAM outputs an **object score** per frame. Low scores mean the model is unsure the target is visible.

### Default behavior (no `--keep_bad_objscores`)

1. On the **first frame** where `object score` falls below `--objscore_threshold` (default `0.0`):
   - The app runs masking **once** on that frame, then **zeros** the mask.
   - It records a **stop frame index** for that object.
2. On that frame (when revisited) and **all later frames**, **`step_video_masking` is not called** for that object; the mask stays zero.
3. **Prompt memory is kept** — the slot is still “active” but not inferenced until you recover.

**Recovery options:**

| Method | Effect |
| --- | --- |
| Move playhead **before** the stop frame | Stop marker clears; tracking can run again from earlier frames. |
| Pause, add **new** FG/BG points or a box, **Store Prompt** | Clears stop marker and appends prompt memory (revival). |

### With `--keep_bad_objscores`

- Masking **continues every frame** after loss (auto-recovery attempt).
- Masks are still **zeroed** on low-score frames.
- Frame history is not updated on low-score frames (same as default on those frames).

Adjust sensitivity with `--objscore_threshold` (higher = stricter “lost” detection). `--lost_patience N` waits for N consecutive low-score frames before stopping (default 1). `--mask_select official` chooses the mask the way SAM 2/3 tracking does. `--max_prompt_attn 4` limits SAM 3 / 3.1 to four prompt memories per frame.

---

## 6. Multi-object notes

- Loss and stop frames are **per object** — one target leaving the field does not stop others.
- Combined export assigns gray levels: 0 = background, and each live object keeps the id it was given when created (1–255). Removing a slot erases that id from frames already recorded; other ids stay put.
- Overlapping pixels go to the object with the higher score. Equal scores keep the smaller id.

### Object sidebar (right panel)

| Count | Behavior |
| --- | --- |
| **1–32** | Two-column **Object N** buttons share the available sidebar height (may compress with window size). |
| **33–255** | The object grid keeps the **same row height** as when 32 objects are shown; **mouse wheel** over the grid scrolls extra rows. Enable Recording, Add/Remove, and Save/Clear stay fixed above and below the list. |
| **Selection** | The **active** object scrolls into view when you pick a slot (sidebar click, **W** / **S**, **↑** / **↓**, or middle-click on a tracked mask). |

Add slots with **+** or **Add Object**; remove with **Shift+`-`** or **Remove Object** (at least one slot always remains).

Click targets are kept in sync with the visible buttons after each UI redraw and when you scroll the list (v1.4.1+).

---

## 7. Feedback, undo, and session memory

- **Toast notifications** — Short, non-blocking messages appear **centered in the video area** for actions such as model/video load, store/clear prompts, add/remove object, history toggle, and **lost-target warnings**. They fade out automatically.
- **Error dialogs** — Failures (e.g. could not save, metric/overlay export failed) appear as modal popups so they are not missed.
- **Undo (Ctrl+Z)** — Removes the most recently added FG/BG point or box for the active object before it is stored. Repeat to step back through the prompts.
- **Session memory (`.history`)** — Restores display size, last save folder, Enable History, object-score threshold, square/aspect sizing, and pixel size when those options are **omitted** on the command line. Passing a value, including the built-in default, overrides the saved one. `--square` stretches each frame to a square; `-ar` forces the original aspect ratio.

---

## 8. Performance (scrubbing and reverse playback)

- **Encode cache** (`--encode_cache_size`, default 64) — Image encodings are cached on the CPU and reused, so scrubbing, frame stepping, and reverse playback avoid re-encoding visited frames.
- **Reverse frame buffer** (`--reverse_buffer_size`, default 120) — Decoded frames are buffered for smoother reverse playback and stepping.
- Set either option to `0` to disable it (lower memory use, slower revisits).

---

## 9. Playback and timeline

| Control | Action |
| --- | --- |
| Space | Play / pause |
| ← / → | Step one frame backward / forward (paused) |
| A / D | Step one frame backward / forward (paused, alternate) |
| R | Reverse playback |
| Timeline slider | Scrub; masks clear while dragging, tracking refreshes on release (if prompts exist) |

While scrubbing, on-screen masks are cleared temporarily; after release, tracking respects each object’s stop frame (objects at or after their loss frame stay inactive until you move earlier or re-store prompts).

---

## 10. Keyboard and mouse

Press **F1** in the main window for the full shortcut panel. Common bindings:

| Input | Action |
| --- | --- |
| H | User guide (English / 中文) |
| F1 | Keyboard shortcuts panel |
| Space | Play / pause |
| Enter | Store Prompt (while paused) |
| Ctrl+Z | Undo last added prompt (FG/BG point or box) |
| Tab / Shift+Tab | Switch prompt tool forward / backward |
| C | Clear on-screen prompts |
| ← / → or A / D | Step one frame (paused) |
| W / S or ↑ / ↓ | Previous / next object |
| + / Shift+− | Add object / remove object (asks when the slot has prompts or labels) |
| [ / ] | Zoom display out / in |
| Mouse wheel (object grid, 33+ objects) | Scroll object list |
| Middle-click | Select object under mask |
| Q / Esc | Quit |

---

## 11. Export

1. Enable **Enable Recording** during tracking.
2. Click **Save Results** — choose a folder.
3. In the **export parameter dialog**, confirm the **frame rate (fps)** and **pixel size (µm/pixel)**. The pixel size is remembered for the next session.
4. A **progress window** shows live status while label images are written and metrics/overlay are rendered.

Output folder: `{video_basename}_MT-Results_{YYYYMMDD-HHMMSS}/`

| File | Contents |
| --- | --- |
| `00010.tif`, `00240.tif`, … | 8-bit grayscale label images named by source frame index (0 = background, N = object id N). |
| `frame_index.csv` | Maps each TIFF filename to `frame_index` and `time_s`. |
| `tracking_metrics.csv` | Per frame and object: centroid (px and µm), area, orientation angle (long axis of the fitted ellipse vs. +X, from image moments), velocity, displacement. |
| `tracking_msd.csv` | Time-averaged Mean Squared Displacement per object. |
| `tracking_overlay.mp4` | One overlay video: each object drawn with a unique contour color and a fading ~20-frame trajectory. Sixteen colors then repeat by object id. Written for video, TIFF stacks, and image folders; a missing source frame is black. |

Only frames visited while recording are saved. Lost objects contribute **background (0)** on frames where their mask is zero. In the overlay video, an object's trajectory is shown only while it is in view; once the object leaves the field of view, its trajectory disappears.

Physical units (µm, µm/s, µm²) depend on the fps and µm/pixel values you enter; centroid is also reported in pixels. The first sample of each object has no velocity. A frame gap larger than the export dialog's max gap also leaves velocity blank. Orientation is measured in image coordinates, so a positive angle is clockwise because y increases downward.

16-bit TIFF stacks and image folders are scaled with one intensity mapping for the whole sequence (`--intensity_range auto` by default). Save Session stores the raw prompts plus the model path, encode side, square setting, and video path. Load Session keeps the saved object ids, applies those encode settings, then re-encodes the prompts on that same video. A different open video is refused. Unsaved recordings are offered save, discard, or cancel first; stored prompts are replaced only after confirmation. Older session files that omit the new fields keep the current video, model, and encode settings. Retry Metrics repeats the failed analysis with the same frame-gap limit and intensity range. Shift+`-` removes an object that already has prompts or labels, after a confirmation.

---

## 12. Command-line reference

```text
python main.py [OPTIONS]

  -i, --video_path PATH       Input video, TIFF stack, or image folder (optional; GUI picker otherwise)
  -m, --model_path PATH       SAM weights (default: model_weights/)
  -d, --device DEVICE         cuda | mps | cpu
  -s, --display_size PX       UI size. Omit to reuse the saved size; pass 900 to force the default.
  -b, --base_size_px PX       Encoder longest side (default 1344). Native: 1024 (SAM 2), 1008 (SAM 3 / 3.1)
  -ar, --use_aspect_ratio     Keep video aspect ratio (overrides the saved choice)
  --square                    Stretch each frame to a square (overrides the saved choice)
  -f32, --use_float32         Float32 on GPU (more VRAM). CPU already uses float32
  --max_memories N            Frame history depth (default: 6)
  --objscore_threshold F      Score below = lost. Omit to reuse the saved value; pass 0 to force the default.
  --keep_bad_objscores        Keep inferencing after loss (masks still zeroed)
  --keep_history_on_new_prompts
                              Retain frame history when adding new prompts
  --encode_cache_size N       CPU encode cache size (0 disables, default 64)
  --reverse_buffer_size N     Reverse-playback frame buffer (0 disables, default 120)
  --mask_select MODE          legacy (default) or official
  --lost_patience N           Consecutive low scores before a target is lost (default 1)
  --max_prompt_attn N         SAM 3 / 3.1 prompt memories per frame (default: all)
  --intensity_range SPEC      16-bit stills: auto, full, or low,high
  --encode_cache_mb N         CPU encode-cache megabyte cap (default 2048)
```

**In-app (recommended):** Press **H** for the tkinter user guide (English / 中文, non-modal).  
**This file:** Markdown copy for offline reading or printing.  
**Shortcuts:** Press **F1** for the keyboard and mouse reference panel (OpenCV window).

---

## 13. Microscopy tips

- Store prompts on a **sharp, in-focus** frame when possible.
- Use **FG/BG** for irregular cells; **boxes** for round cells.
- If drift builds up, disable **History**, clear frame memory, and re-store prompts on a good frame.
- Use a smaller `-b` if VRAM is limited (faster, less detail).

---

## 14. Version history

See [CHANGELOG.md](../CHANGELOG.md).
