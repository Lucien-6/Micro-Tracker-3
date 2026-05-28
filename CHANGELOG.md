# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-05-28

### Changed

- **`--keep_bad_objscores`** — Now controls whether video masking continues after a low object score (lost target). Default behavior stops `step_video_masking` for that object from the first loss frame onward (mask still zeroed on low-score frames). With the flag set, inference continues for possible auto-recovery while masks remain zeroed during loss.
- **Per-object loss handling** — Each object slot records a `tracking_stop_frame_idx` when lost (default mode). Inference resumes if the playhead moves to an earlier frame; stopped objects are skipped at or after the loss frame until cleared.
- **Store Prompt** — Requires interactive prompts (foreground/background points or a box; hover-only on already-tracked objects is not accepted). Successful store clears the loss stop marker for that object (revival after loss).

### Added

- `docs/USER_GUIDE.md` — Standalone user guide (workflow, lost-target behavior, CLI reference)
- F1 shortcuts panel section **Tracking & Loss** (in-app help)

## [1.0.0] - 2026-05-28

First stable release of **Micro Tracker 3**.

### Added

- Interactive video segmentation and tracking GUI built on SAM 2 / SAM 3 / SAM 3.1 (`muggled_sam`)
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

[1.1.0]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.1.0
[1.0.0]: https://github.com/your-username/micro-tracker-3/releases/tag/v1.0.0
