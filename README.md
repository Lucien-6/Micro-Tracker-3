# Micro Tracker 3

**Interactive multi-object video segmentation and tracking for microscopy and beyond.**

Micro Tracker 3 is a desktop application for annotating targets in video frames and propagating instance masks through time using Meta's **Segment Anything Model (SAM)** video-tracking pipeline. It is designed for workflows where classical thresholding is unreliable—such as **microbial motility**, particle tracking, or any scene requiring prompt-based segmentation—while exporting analysis-ready **8-bit label image sequences**.

Built on a bundled [muggled_sam](https://github.com/heyoeyo/muggled_sam) inference stack (SAM 2 / SAM 3 / SAM 3.1, pure PyTorch), Micro Tracker 3 wraps model loading, an OpenCV-based GUI, multi-object memory management, and TIF export into a single interactive tool.

**Current version:** [1.2.0](CHANGELOG.md) (2026-05-28) · **Author:** Lucien · **License:** [MIT](LICENSE) · **User guide:** press **H** in-app, or [docs/USER_GUIDE.md](docs/USER_GUIDE.md)

---

## Features

| Category | Capability |
|----------|------------|
| **Segmentation** | Point, box, and hover-based prompts with live mask preview |
| **Tracking** | Temporal propagation via SAM memory encoder (prompt + frame history) |
| **Lost-target policy** | Per-object stop on low object score (default); optional continued inference via `--keep_bad_objscores` |
| **Multi-object** | Up to 32 independent object slots, each with its own memory bank |
| **Playback** | Pause, play, reverse, frame stepping, and timeline scrubbing |
| **Export** | Combined per-frame label masks as `00001.tif`, `00002.tif`, … |
| **Models** | Auto-detects SAM 2, SAM 3, or SAM 3.1 weights (`.pt` / `.pth`) |
| **Hardware** | CUDA, Apple MPS, or CPU; default bfloat16 for lower VRAM use |
| **Help** | **H** — tkinter user guide (English / 中文); **F1** — keyboard shortcuts panel |

---

## Requirements

- **Python** 3.10+ recommended
- **PyTorch** ≥ 2.1 (with CUDA or MPS if available)
- **OpenCV** ≥ 4.5, &lt; 4.14
- **NumPy** ≥ 1.21
- **tkinter** (optional, for native file/folder dialogs; usually included with Python on Windows)

### Hardware notes

- A **GPU with sufficient VRAM** is strongly recommended for real-time tracking.
- Default inference uses **bfloat16**; pass `-f32` to force float32 (roughly doubles VRAM).
- Default encoding resolution uses a **1344 px** longest side (`-b` to change).

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/micro-tracker-3.git
cd micro-tracker-3
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

```
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

Or specify resources on the command line:

```bash
python main.py -i path/to/video.mp4 -m model_weights/sam2_hiera_large.pt
```

On first run, the app resolves the model from (in order):

1. `-m / --model_path` if the file exists  
2. Last path stored in `.history`  
3. The only file in `model_weights/`, or the newest match  

---

## Workflow

```text
  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
  │ Load video  │ ──► │ Pause & annotate │ ──► │ Store Prompt    │
  │  (+ model)  │     │ (box / FG / BG)  │     │  (Tab / button) │
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
4. **Annotate** — Choose Hover, Box, FG Point, or BG Point; place prompts on the target.
5. **Store Prompt** — Click **Store Prompt** or press **Tab** (requires FG/BG points or a box on the current frame; see [user guide](docs/USER_GUIDE.md)).
6. **Track** — Press **Track** or Space to play forward; masks propagate automatically each frame. If a target is lost (low object score), tracking for that object stops by default until you move to an earlier frame or store new prompts.
7. **Record** — Enable **Enable Recording** to buffer label frames in memory.
8. **Export** — Click **Save Results** to write a TIF sequence to disk.

Repeat steps 3–5 for additional objects before tracking.

---

## Keyboard shortcuts

Press **F1** inside the app for the full in-GUI reference. Summary:

| Keys | Action |
|------|--------|
| `Space` | Play / pause |
| `A` / `D` | Step backward / forward (while paused) |
| `R` | Toggle reverse playback |
| `←` / `→` | Switch prompt tool (Hover / Box / FG / BG) |
| `Tab` | Store current prompts to selected object |
| `C` | Clear on-screen prompts (not stored memory) |
| `↑` / `↓` or `W` / `S` | Previous / next object slot |
| `+` / `-` | Add / remove object slot |
| `[` / `]` | Zoom display out / in |
| Middle-click | Select object under cursor (tracked masks) |
| `H` | Toggle user guide (tkinter, English / 中文) |
| `F1` | Toggle shortcuts panel |
| `Q` / `Esc` | Quit (prompts to save unsaved results) |

---

## Command-line options

```text
python main.py [OPTIONS]

  -i, --video_path PATH       Input video (optional; otherwise pick in GUI)
  -m, --model_path PATH       SAM weights (optional; default: model_weights/)
  -d, --device DEVICE         cuda | mps | cpu (default: auto-detect)
  -s, --display_size PX       UI display size (default: 900)
  -b, --base_size_px PX       Image encoder longest side (default: 1344)
  -ar, --use_aspect_ratio     Keep original aspect ratio (default: square pad)
  -f32, --use_float32         Use float32 instead of bfloat16
  --max_memories N            Frame memory history length (default: 6)
  --objscore_threshold F      Object score below which target is treated as lost (default: 0.0)
  --keep_bad_objscores        Keep inferencing after loss; masks still zeroed on low-score frames
  --keep_history_on_new_prompts
                              Retain frame history when adding new prompts
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
├── 00001.tif
├── 00002.tif
└── ...
```

Each file is an **8-bit grayscale label image**:

- Pixel value **0** = background  
- Pixel value **1** = Object 1  
- Pixel value **2** = Object 2  
- …  

Overlapping instances are resolved by **later objects overwriting earlier ones**. Frame order follows sorted frame indices, not necessarily consecutive video frame numbers.

This format is compatible with common downstream tools (ImageJ, TrackMate, custom Python/MATLAB pipelines) for morphology and kinematics analysis.

---

## Project structure

```text
micro-tracker-3/
├── main.py                   # Application entry point
├── requirements.txt
├── VERSION                   # Current release (1.2.0)
├── CHANGELOG.md
├── docs/
│   └── USER_GUIDE.md         # Markdown user guide (same topics as H-key window)
├── LICENSE
├── model_weights/            # Place SAM checkpoints here
└── muggled_sam/
    ├── make_sam.py           # Weight loader & version detection
    ├── v2_sam/               # SAM 2 implementation
    ├── v3_sam/               # SAM 3 implementation
    ├── v3p1_sam/             # SAM 3.1 implementation
    └── demo_helpers/         # UI, I/O, memory banks, saving
        ├── shared_ui_layout.py
        ├── video_data_storage.py
        ├── saving.py
        └── ui/               # OpenCV widget toolkit, F1 shortcuts, H-key user guide (tkinter)
```

---

## Architecture overview

Micro Tracker 3 has three layers:

1. **Application (`main.py`)** — Main loop, state machine (paused / tracking / scrubbing), multi-object scheduling, recording, and resource switching.
2. **Demo helpers (`muggled_sam/demo_helpers/`)** — Reusable OpenCV UI, `SAMVideoMemoryBank`, file dialogs, and TIF export.
3. **Model stack (`muggled_sam/`)** — Pure-PyTorch SAM with two runtime contexts:
   - **Interactive context** — `encode_image`, `encode_prompts`, `generate_masks` (single-frame segmentation)
   - **Tracking context** — `encode_prompt_memory`, `step_video_masking`, `encode_frame_memory` (video propagation)

Each object slot maintains its own prompt memory (up to 32 entries), frame memory deque (default depth 6, configurable via `--max_memories`), and an optional **tracking stop frame index** when a target is lost in default mode.

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

For bugs and feature requests, please open a [GitHub Issue](https://github.com/your-username/micro-tracker-3/issues).
