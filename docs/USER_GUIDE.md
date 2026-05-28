# Micro Tracker 3 — User Guide

**Version:** 1.2.0  
**Last updated:** 2026-05-28  
**Author:** Lucien · [lucien-6@qq.com](mailto:lucien-6@qq.com)

This document mirrors the in-app guide (press **H** while the main window is focused; switch **English / 中文** in the guide window). For installation and repository layout, see [README.md](../README.md).

---

## 1. Overview

Micro Tracker 3 is an interactive desktop tool for:

1. Annotating objects in video frames with SAM (points, boxes).
2. Propagating instance masks through time (tracking).
3. Exporting combined **8-bit label TIF** sequences for downstream analysis (ImageJ, TrackMate, custom pipelines).

The app supports up to **32 object slots**, each with its own prompt memory and optional frame history.

---

## 2. Typical workflow

| Step | Action |
|------|--------|
| 1 | Load **model** and **video** (GUI buttons or `-m` / `-i`). |
| 2 | Select an **object slot** (Object 1, 2, …). |
| 3 | **Pause** the video (Space). |
| 4 | Choose a prompt tool (Hover / Box / FG / BG) and annotate the target. |
| 5 | **Store Prompt** (Tab or button) — writes prompts into the tracker memory bank. |
| 6 | **Track** — play forward (Space / Track); masks update each new frame. |
| 7 | Optional: **Enable Recording**, then **Save Results** for TIF export. |

Repeat steps 2–5 for additional objects before tracking.

**Important:** Prompt tools are disabled during playback. Pause before annotating or storing prompts.

---

## 3. Prompt tools

| Tool | Use |
|------|-----|
| **Hover** | Live mask preview when the slot has **no** stored prompts yet. |
| **Box** | Drag a rectangle around the target. |
| **FG Point** | Foreground clicks (include region). |
| **BG Point** | Background clicks (exclude region). |

- **C** — Clears on-screen prompts only (does not remove stored tracker memory).
- **Tab / Store Prompt** — Saves current interactive prompts to the **selected** object slot.

### Store Prompt requirements

Store Prompt only runs when you have **interactive** prompts on the current frame:

- At least one foreground/background point **or** a box, and  
- Not **hover-only** prompts on an object that already has stored prompts.

If Store is ignored, add FG/BG points or a box (switch off Hover for tracked objects), then try again.

---

## 4. Tracking and temporal memory

- **Enable History** (default on) — Appends per-frame memory encodings (depth set by `--max_memories`, default 6) to help long runs.
- **Clear History** — Removes frame memory for the selected object; prompt memory remains.
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
|--------|--------|
| Move playhead **before** the stop frame | Stop marker clears; tracking can run again from earlier frames. |
| Pause, add **new** FG/BG points or a box, **Store Prompt** | Clears stop marker and appends prompt memory (revival). |

### With `--keep_bad_objscores`

- Masking **continues every frame** after loss (auto-recovery attempt).
- Masks are still **zeroed** on low-score frames.
- Frame history is not updated on low-score frames (same as default on those frames).

Adjust sensitivity with `--objscore_threshold` (higher = stricter “lost” detection).

---

## 6. Multi-object notes

- Loss and stop frames are **per object** — one target leaving the field does not stop others.
- Combined export assigns gray levels: 0 = background, 1 = Object 1, 2 = Object 2, …
- Later objects overwrite overlapping pixels in the combined label image.

---

## 7. Playback and timeline

| Control | Action |
|---------|--------|
| Space | Play / pause |
| A / D | Step one frame (paused) |
| R | Reverse playback |
| Timeline slider | Scrub; masks clear while dragging, tracking refreshes on release (if prompts exist) |

While scrubbing, on-screen masks are cleared temporarily; after release, tracking respects each object’s stop frame (objects at or after their loss frame stay inactive until you move earlier or re-store prompts).

---

## 8. Export

1. Enable **Enable Recording** during tracking.
2. Click **Save Results** — choose a folder.
3. Output folder: `{video_basename}_MT-Results_{YYYYMMDD-HHMMSS}/` with `00001.tif`, `00002.tif`, …

Only frames visited while recording are saved. Lost objects contribute **background (0)** on frames where their mask is zero.

---

## 9. Command-line reference

```text
python main.py [OPTIONS]

  -i, --video_path PATH       Input video (optional; GUI picker otherwise)
  -m, --model_path PATH       SAM weights (default: model_weights/)
  -d, --device DEVICE         cuda | mps | cpu
  -s, --display_size PX       UI size (default: 900)
  -b, --base_size_px PX       Encoder longest side (default: 1344)
  -ar, --use_aspect_ratio     Keep video aspect ratio (default: square pad)
  -f32, --use_float32         Float32 weights (more VRAM)
  --max_memories N            Frame history depth (default: 6)
  --objscore_threshold F      Score below = lost (default: 0.0)
  --keep_bad_objscores        Keep inferencing after loss (masks still zeroed)
  --keep_history_on_new_prompts
                              Retain frame history when adding new prompts
```

**In-app (recommended):** Press **H** for the tkinter user guide (English / 中文, non-modal).  
**This file:** Markdown copy for offline reading or printing.  
**Shortcuts:** Press **F1** for the keyboard and mouse reference panel (OpenCV window).

---

## 10. Microscopy tips

- Store prompts on a **sharp, in-focus** frame when possible.
- Use **FG/BG** for irregular cells; **boxes** for round cells.
- If drift builds up, disable **History**, clear frame memory, and re-store prompts on a good frame.
- Use a smaller `-b` if VRAM is limited (faster, less detail).

---

## 11. Version history

See [CHANGELOG.md](../CHANGELOG.md).
