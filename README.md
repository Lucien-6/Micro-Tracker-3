# Micro Tracker 3

**Interactive multi-object video segmentation and tracking for microscopy and beyond.**

Micro Tracker 3 is a desktop application for annotating targets in video frames and propagating instance masks through time using Meta's **Segment Anything Model (SAM)** video-tracking pipeline. It is designed for workflows where classical thresholding is unreliable—such as **microbial motility**, particle tracking, or any scene requiring prompt-based segmentation—while exporting analysis-ready **8-bit label image sequences**.

Built on a bundled SAM inference stack in [`src/`](src/) (derived from [muggled_sam](https://github.com/heyoeyo/muggled_sam); SAM 2 / SAM 3 / SAM 3.1, pure PyTorch), Micro Tracker 3 wraps model loading, an OpenCV-based GUI, multi-object memory management, and TIF export into a single interactive tool.

**Current version:** [1.8.2](CHANGELOG.md) (2026-09-23) · **Author:** Lucien · **License:** [MIT](LICENSE) for this application; Apache-2.0 for the vendored SAM code ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)) · **User guide:** press **H** in-app, or [docs/USER_GUIDE.md](docs/USER_GUIDE.md)

---

## Features

| Category | Capability |
| --- | --- |
| **Input** | Video files, multipage TIFF stacks, and folders of still images |
| **Segmentation** | Point, box, and hover-based prompts with live mask preview |
| **Tracking** | Temporal propagation via SAM memory encoder (prompt + frame history) |
| **Lost-target policy** | Per-object stop on low object score (default); optional continued inference via `--keep_bad_objscores` |
| **Multi-object** | Up to 255 slots with stable label ids; scrollable two-column list (mouse wheel) when more than 32 |
| **Playback** | Pause, play, reverse, frame stepping, and timeline scrubbing; CPU encode cache + reverse frame buffer for fast scrubbing/stepping |
| **Export** | Frame-index label TIFFs (`00240.tif`), `frame_index.csv`, metrics CSV, MSD CSV, and an overlay video; **Retry Metrics** repeats a failed analysis with the same frame gap and intensity range |
| **Analysis** | Per-frame centroid, area, orientation (fitted-ellipse long axis), velocity, displacement, and MSD |
| **Feedback** | On-screen **toast** messages centered in the video; modal dialogs for errors; live **save progress** window |
| **Editing** | **Ctrl+Z** undo for the last prompt; **Save / Load Session** keeps prompts, object ids, model path, encode side, square setting, and video path; `.history` restores display options |
| **Models** | Auto-detects SAM 2, SAM 3, or SAM 3.1 weights (`.pt` / `.pth`) |
| **Hardware** | CUDA, Apple MPS, or CPU. GPU default is bfloat16. CPU stays float32 |
| **Help** | **H** — tkinter user guide (English / 中文); **F1** — keyboard shortcuts panel |

---

## Requirements

- **Python** 3.10 or newer
- **PyTorch** ≥ 2.1 (with CUDA or MPS if available)
- **OpenCV** ≥ 4.5, &lt; 4.14
- **NumPy** ≥ 1.21
- **tkinter** (required for dialogs, saving, and the in-app user guide; included with Python on Windows)

### Hardware notes

- A **GPU with sufficient VRAM** is strongly recommended for real-time tracking.
- GPU inference uses **bfloat16** unless you pass `-f32` (roughly doubles VRAM). CPU inference stays **float32**.
- The default encoding side is **1344** px. Pass `-b 1024` for SAM 2's native side, or `-b 1008` for SAM 3 / 3.1.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Lucien-6/Micro-Tracker-3.git
cd Micro-Tracker-3
```

### 2. Create a virtual environment (recommended)

**Conda:**

```bash
conda create -n micro-tracker-3 python=3.11 -y
conda activate micro-tracker-3
```

**venv:**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Install PyTorch

Install the build matching your platform from [pytorch.org](https://pytorch.org/get-started/locally/), then:

```bash
pip install -r requirements.txt
```

### 4. Download SAM model weights

Place one or more weight files in the `model_weights/` folder:

```text
micro-tracker-3/
└── model_weights/
    └── sam2_hiera_large.pt   # example filename
```

Supported formats: official Meta checkpoints and muggled-sam converted weights for **SAM 2**, **SAM 3**, and **SAM 3.1**. The loader picks the correct architecture automatically from the checkpoint keys.

For download links and conversion notes, see the [muggled_sam model weights guide](https://github.com/heyoeyo/muggled_sam/blob/main/README.md#model-weights).

> **Note:** Model weights are not included in this repository due to size and licensing.

---

## Quick start

Launch the application (video can be selected inside the GUI):

```bash
python main.py
```

`python main.py` resolves `src` from the folder that contains `main.py`, so the shell does not have to start in that folder. Import modules as `from src.make_sam import ...` (not `muggled_sam`).

Or specify resources on the command line:

```bash
python main.py -i path/to/video.mp4 -m model_weights/sam2_hiera_large.pt
```

On first run, the app resolves the model from (in order):

1. `-m / --model_path` when it is an existing file, or the unique file in `model_weights/` whose name contains that text. A selector that matches nothing is an error.
2. With no `-m`, the path stored in `.history` when that file still exists.
3. The other files in `model_weights/`, in name order.  

---

## Workflow

```text
  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
  │ Load video  │ ──► │ Pause & annotate │ ──► │ Store Prompt    │
  │  (+ model)  │     │ (box / FG / BG)  │     │ (Enter / btn)   │
  └─────────────┘     └──────────────────┘     └────────┬────────┘
                                                        │
                        ┌──────────────────┐            ▼
                        │ Save TIF labels  │ ◄── ┌──────────────┐
                        │  (Save Results)  │     │ Track (play) │
                        └──────────────────┘     └──────────────┘
```

### Step-by-step

1. **Load resources** — Use the **Model** and **Video** buttons in the header bar, or pass `-m` / `-i`.
2. **Select an object slot** — `Object 1`, `Object 2`, … (Up/Down or W/S). Add slots with **+**.
3. **Pause the video** — Space bar.
4. **Annotate** — Choose Hover, Box, FG Point, or BG Point (**Tab / Shift+Tab** to switch tools); place prompts on the target.
5. **Store Prompt** — Click **Store Prompt (Enter)** or press **Enter** while paused (requires FG/BG points or a box on the current frame; see [user guide](docs/USER_GUIDE.md)).
6. **Track** — Press **Track** or Space to play forward; masks propagate automatically each frame. If a target is lost (low object score), tracking for that object stops by default until you move to an earlier frame or store new prompts.
7. **Record** — Enable **Enable Recording** to buffer label frames in memory.
8. **Export** — Click **Save Results**, confirm **frame rate** and **pixel size (µm/pixel)** in the export dialog, then the app writes the TIF label sequence plus the metrics CSV, MSD CSV, and overlay video. A progress window shows live status.
9. **Save Session** — Stores the prompts, object ids, model file, encode side, square setting, and video path. **Load Session** rebuilds the masks on that same video after asking before it replaces unsaved recordings or stored prompts. A different open video is refused.

Repeat steps 3–5 for additional objects before tracking.

---

## Keyboard shortcuts

Press **F1** inside the app for the full in-GUI reference. Summary:

| Keys | Action |
| --- | --- |
| `Space` | Play / pause |
| `←` / `→` | Step backward / forward (while paused) |
| `A` / `D` | Step backward / forward (while paused, alternate) |
| `R` | Toggle reverse playback |
| `Tab` / `Shift+Tab` | Switch prompt tool forward / backward (Hover / Box / FG / BG) |
| `Enter` | Store current prompts to selected object (while paused) |
| `Ctrl+Z` | Undo the last added prompt (FG/BG point or box) |
| `C` | Clear on-screen prompts (not stored memory) |
| `↑` / `↓` or `W` / `S` | Previous / next object slot |
| `+` / Shift+- | Add object / remove object |
| `[` / `]` | Zoom display out / in |
| Middle-click | Select object under cursor (tracked masks) |
| `H` | Toggle user guide (tkinter, English / 中文) |
| `F1` | Toggle shortcuts panel |
| `Q` / `Esc` | Quit (prompts to save unsaved results) |

---

## Command-line options

```text
python main.py [OPTIONS]

  -i, --video_path PATH       Input video, TIFF stack, or image folder (optional; otherwise pick in GUI)
  -m, --model_path PATH       SAM weights (optional; default: model_weights/)
  -d, --device DEVICE         cuda | mps | cpu (default: auto-detect)
  -s, --display_size PX       UI display size (default: 900)
  -b, --base_size_px PX       Encoder longest side (default: 1344). Native: 1024 (SAM 2), 1008 (SAM 3 / 3.1)
  -ar, --use_aspect_ratio     Keep original aspect ratio (overrides the saved choice)
  --square                    Stretch each frame to a square (overrides the saved choice)
  -f32, --use_float32         Float32 on GPU. CPU already uses float32
  --max_memories N            Frame memory history length (default: 6)
  --objscore_threshold F      Object score below which target is treated as lost.
                              Omit to reuse the saved value; pass 0 to force the default.
  --keep_bad_objscores        Keep inferencing after loss; masks still zeroed on low-score frames
  --keep_history_on_new_prompts
                              Retain frame history when adding new prompts
  --encode_cache_size N       CPU LRU cache of image encodings for fast scrubbing/stepping/reverse
                              (0 disables, default: 64)
  --reverse_buffer_size N     Decoded-frame buffer for reverse playback / stepping
                              (0 disables, default: 120)
  --mask_select MODE          legacy (default) or official
  --lost_patience N           Consecutive low scores before a target is lost (default: 1)
  --max_prompt_attn N         SAM 3 / 3.1 prompt memories used per frame (default: all)
  --intensity_range SPEC      16-bit stills: auto, full, or low,high
  --encode_cache_mb N         CPU encode-cache byte cap (default: 2048, 0 disables the cap)
```

**Lost target (default):** On the first low-score frame, the app masks once, zeros the mask, and stops further inference for that object on that frame and later frames. Move the playhead before the loss frame or add new points/box and **Store Prompt** to resume. See [docs/USER_GUIDE.md](docs/USER_GUIDE.md#5-lost-targets-out-of-frame-defocus-occlusion).

Example — CPU inference with aspect-ratio-preserving encoding:

```bash
python main.py -i sample.mp4 -d cpu -ar -b 1024
```

---

## Output format

When recording is enabled and results are saved, the app writes a folder named:

```text
{video_basename}_MT-Results_{YYYYMMDD-HHMMSS}/
├── 00000.tif               # label for video frame 0
├── 00240.tif               # label for video frame 240
├── frame_index.csv         # filename, frame_index, time_s
├── tracking_metrics.csv    # per-frame, per-object metrics
├── tracking_msd.csv        # time-averaged MSD per object
└── tracking_overlay.mp4    # colored contours + fading trajectories (if a video is loaded)
```

Each `*.tif` is an **8-bit grayscale label image** named with the source frame index:

- Pixel value **0** = background
- Pixel value **1** = the object whose stable id is 1
- Pixel value **2** = the object whose stable id is 2
- …

Object ids stay fixed when a slot is removed. Removing a slot erases that id from frames already in memory; the other ids do not move. Overlapping pixels are given to the object with the higher score on that frame. Equal scores keep the smaller id.

The label sequence is compatible with common downstream tools (ImageJ, TrackMate, custom Python/MATLAB pipelines). In addition, Micro Tracker 3 computes analysis-ready outputs directly:

- **`tracking_metrics.csv`** — per frame and object: centroid (px and µm), area, orientation angle (long axis of the fitted ellipse vs. +X, derived from image moments), velocity, and displacement. Physical units use the **fps** and **µm/pixel** values entered in the export dialog.
- **`tracking_msd.csv`** — time-averaged Mean Squared Displacement per object.
- **`tracking_overlay.mp4`** — every object as a uniquely colored contour with a fading ~20-frame trajectory; a trajectory disappears once its object leaves the field of view.

---

## Project structure

```text
micro-tracker-3/
├── main.py                   # Application entry point
├── requirements.txt
├── VERSION                   # Current release (1.8.2)
├── CHANGELOG.md
├── docs/
│   └── USER_GUIDE.md         # Markdown user guide (same topics as H-key window)
├── LICENSE
├── model_weights/            # Place SAM checkpoints here
└── src/
    ├── make_sam.py           # Weight loader & version detection
    ├── v2_sam/               # SAM 2 implementation
    ├── v3_sam/               # SAM 3 implementation
    ├── v3p1_sam/             # SAM 3.1 implementation
    └── demo_helpers/         # UI, I/O, memory banks, saving, analysis
        ├── shared_ui_layout.py
        ├── video_data_storage.py
        ├── saving.py
        ├── analysis.py       # metrics / MSD CSV + overlay video export
        ├── encode_cache.py   # CPU LRU cache of image encodings
        └── ui/               # OpenCV widget toolkit, toast, progress, F1 shortcuts, H-key user guide (tkinter)
```

---

## Architecture overview

Micro Tracker 3 has three layers:

1. **Application (`main.py`)** — Main loop, state machine (paused / tracking / scrubbing), multi-object scheduling, recording, and resource switching.
2. **Demo helpers (`src/demo_helpers/`)** — Reusable OpenCV UI, `SAMVideoMemoryBank`, file dialogs, and TIF export.
3. **Model stack (`src/`)** — Pure-PyTorch SAM with two runtime contexts:
   - **Interactive context** — `encode_image`, `encode_prompts`, `generate_masks` (single-frame segmentation)
   - **Tracking context** — `encode_prompt_memory`, `step_video_masking`, `encode_frame_memory` (video propagation)

Each object slot maintains its own prompt memory (up to 32 entries per slot), frame memory deque (default depth 6, configurable via `--max_memories`), and an optional **tracking stop frame index** when a target is lost in default mode. With more than 32 slots, the sidebar object grid scrolls with the mouse wheel while keeping the same row height as at 32 objects; the active slot scrolls into view when selected.

Press **H** for the in-app user guide (English / 中文) or **F1** for keyboard shortcuts.

---

## Tips for microscopy video

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for full detail. Summary:

- **Pause before annotating** — Prompt tools are disabled during playback.
- **Use FG/BG points** on irregular cell shapes; boxes work well for round cells.
- **Store Prompt on a clear frame** — Requires interactive points or a box; avoid hover-only on tracked objects.
- **Enable History** — Keeps temporal context; disable if drift accumulates, then re-store prompts.
- **Lost target** — Default: stop inference from the loss frame onward (mask zeroed). Scrub earlier or re-store prompts to resume; use `--keep_bad_objscores` for continuous inference.
- **Smaller `-b`** values reduce VRAM and speed up encoding but may lose fine detail.

---

## Acknowledgments

- **[muggled_sam](https://github.com/heyoeyo/muggled_sam)** — Pure PyTorch SAM 2 / 3 / 3.1 inference used as the segmentation backend.
- **[Segment Anything (Meta AI)](https://segment-anything.com/)** — Original SAM models and research.

Micro Tracker 3 adds the interactive video-tracking GUI, multi-object workflow, and label-sequence export on top of the bundled inference code.

---

## License

This project is released under the [MIT License](LICENSE).

SAM model weights remain subject to their respective upstream licenses (Meta / muggled_sam terms). Users are responsible for obtaining and complying with those licenses.

---

## Contact

**Lucien** — [lucien-6@qq.com](mailto:lucien-6@qq.com)

For bugs and feature requests, please open a [GitHub Issue](https://github.com/Lucien-6/Micro-Tracker-3/issues).
