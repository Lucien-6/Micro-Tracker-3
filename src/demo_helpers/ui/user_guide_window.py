#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------------------------------------------------
# %% Guide content


@dataclass(frozen=True)
class GuideSection:
    title: str
    blocks: tuple[str | tuple[str, ...], ...]


def _bullets(*items: str) -> tuple[str, ...]:
    return items


GUIDE_CONTENT: dict[str, tuple[GuideSection, ...]] = {
    "en": (
        GuideSection(
            "1. Overview",
            (
                "Micro Tracker 3 is an interactive desktop application for microscopy and general video workflows. "
                "It lets you annotate targets with SAM (Segment Anything Model), propagate instance masks through time, "
                "and export 8-bit label image sequences for analysis in ImageJ, TrackMate, MATLAB, or Python.",
                _bullets(
                    "Up to 255 independent object slots",
                    "SAM 2 / SAM 3 / SAM 3.1 weights via the bundled src/ inference stack",
                    "CUDA, Apple MPS, or CPU inference",
                ),
            ),
        ),
        GuideSection(
            "2. Quick workflow",
            (
                "Step 1 — Load model and video (header bar buttons, or -m / -i on the command line).",
                "Step 2 — Select an object slot (Object 1, Object 2, …; Up/Down or W/S).",
                "Step 3 — Pause playback (Space). Prompt tools are disabled while the video plays.",
                "Step 4 — Choose Hover, Box, FG Point, or BG Point; place prompts on the target.",
                "Step 5 — Store Prompt (Enter or button, while paused). This writes prompts into the tracker memory bank.",
                "Step 6 — Track: press Track or Space to play; masks update on each new frame.",
                "Step 7 — Optional: Enable Recording during tracking, then Save Results to export TIF labels.",
                "Repeat steps 2–5 for additional objects before you start tracking.",
            ),
        ),
        GuideSection(
            "3. Prompt tools",
            (
                "Hover — Live mask preview when the selected slot has no stored prompts yet.",
                "Box — Drag a rectangle around the target.",
                "FG Point — Foreground clicks (include region).",
                "BG Point — Background clicks (exclude region).",
                "Mouse interactions (per tool):",
                _bullets(
                    "Hover — Move: live mask preview. L-click: add an FG point (always appends) and switch to FG. R-click: add a BG point (always appends) and switch to BG. Shift has no effect in Hover.",
                    "FG / BG — L-click: if no points yet, place the first one; otherwise REPLACE the last point. Shift+L-click: APPEND a new point (use this for multi-point prompts). R-click: delete the point nearest to the click (shift has no effect).",
                    "Box — Drag: REPLACE the last box with the new one. Shift+drag: APPEND a new box (use this for multi-box prompts). R-click: delete the box whose corner is nearest to the click. L-click without dragging: discards the last box without adding a new one (avoid).",
                ),
                _bullets(
                    "Ctrl+Z — Undo the most recently added FG/BG point or box (before storing)",
                    "C — Clear on-screen prompts only (does not remove stored tracker memory)",
                    "Enter / Store Prompt — Save current interactive prompts to the selected object (while paused)",
                ),
                "Store Prompt requirements:",
                _bullets(
                    "At least one foreground/background point or a box on the current frame",
                    "Not hover-only prompts on an object that already has stored prompts",
                    "If Store is ignored, switch to FG/BG or Box, add prompts, then Store again",
                ),
            ),
        ),
        GuideSection(
            "4. Tracking memory",
            (
                "Enable History (default on) — Keeps a short deque of per-frame memory. A new entry is stored only for the next adjacent frame in the same direction; a jump or direction change clears that chain and keeps the prompts.",
                "Store Prompt clears the selected object's frame memory unless --keep_history_on_new_prompts is set.",
                "Clear History — Removes frame memory for the selected object; prompt memory stays.",
                "Clear Prompts — Clears stored prompt memory for the selected object.",
                "Tracking runs for every slot with stored prompts on each new frame during play, keyboard step, or timeline scrub release.",
            ),
        ),
        GuideSection(
            "5. Lost targets (out of frame, defocus, occlusion)",
            (
                "SAM reports an object score each frame. Low scores mean the model is unsure the target is visible.",
                "Default (no --keep_bad_objscores):",
                _bullets(
                    "On the first frame below --objscore_threshold (default 0.0), masking runs once, then the mask is zeroed",
                    "A stop frame index is recorded for that object",
                    "On that frame (when revisited) and all later frames, step_video_masking is not called; mask stays zero",
                    "Prompt memory is kept until you recover",
                ),
                "Recovery:",
                _bullets(
                    "Move the playhead to a frame before the stop frame — stop marker clears; tracking can resume",
                    "Pause, add new FG/BG points or a box, then Store Prompt — clears stop and appends prompt memory",
                ),
                "With --keep_bad_objscores — Inference continues every frame (auto-recovery attempt); masks are still zeroed on low-score frames.",
            ),
        ),
        GuideSection(
            "6. Multi-object and export",
            (
                "Loss handling is per object; other slots keep tracking independently.",
                "Combined export uses stable gray levels: 0 = background, and each object keeps the id it was given when created (1-255). Removing a slot erases that id from recorded frames; other ids do not move.",
                "Overlapping pixels go to the higher object score. Equal scores keep the smaller id.",
                "Recording + Save Results writes {video}_MT-Results_{timestamp}/ with TIFF names equal to the source frame index (00240.tif is frame 240), plus frame_index.csv.",
                "If metrics fail after the labels are written, the in-memory frames stay. Retry Metrics writes the CSV files and overlay into that same folder.",
                "Save Session stores the prompts, the model path, the encode side, the square setting, and the video path. Load Session keeps those object ids, applies the encode settings, and re-encodes on that same video. A different video is refused. Unsaved recordings can be saved, discarded, or left in place. Older session files keep the current video and encode settings.",
                "Retry Metrics repeats a failed metrics export with the same frame-gap limit and 16-bit intensity range.",
                "Only recorded frames are saved. Lost objects contribute label 0 where their mask is zero.",
                "On Save, an export dialog asks for frame rate (fps) and pixel size (um/pixel); a progress window shows live status.",
                "Alongside the labels, the app writes analysis files:",
                _bullets(
                    "tracking_metrics.csv — per-frame centroid, area, orientation (fitted-ellipse long axis vs +X), velocity, displacement",
                    "Orientation is clockwise in image coordinates because y increases downward",
                    "tracking_msd.csv — time-averaged Mean Squared Displacement per object",
                    "tracking_overlay.mp4 — colored contour per object + fading ~20-frame trajectory (hidden once an object leaves view)",
                    "Overlay colors are 16 values and then repeat by object id",
                ),
            ),
        ),
        GuideSection(
            "6b. Object sidebar (33+ slots)",
            (
                "The right panel lists objects in two columns (Object 1, Object 2, …).",
                "With 33 or more objects, the list keeps the same button row height as at 32 objects.",
                "Scroll the grid with the mouse wheel. Enable Recording and Add/Remove / Save/Clear stay fixed outside the scroll area.",
                "When you change the active object (sidebar, W/S, arrows, or middle-click on a mask), the list scrolls to keep that slot visible.",
                "Click and hover on Object N buttons match the visible labels (including after scroll).",
            ),
        ),
        GuideSection(
            "7. Playback and timeline",
            (
                _bullets(
                    "Space — Play / pause",
                    "Left / Right or A / D — Step one frame while paused",
                    "R — Reverse playback",
                    "Timeline slider — Scrub; masks clear while dragging; tracking refreshes on release if prompts exist",
                ),
                "While scrubbing, on-screen masks are cleared temporarily. After release, each object respects its stop frame.",
            ),
        ),
        GuideSection(
            "8. Keyboard reference",
            (
                _bullets(
                    "H — Toggle this user guide (English / 中文)",
                    "F1 — Toggle keyboard shortcuts panel (OpenCV window)",
                    "Space — Play / pause",
                    "Enter — Store Prompt (while paused)",
                    "Ctrl+Z — Undo last added prompt (FG/BG point or box)",
                    "C — Clear on-screen prompts",
                    "Tab / Shift+Tab — Switch prompt tool (forward / back)",
                    "↑ / ↓ or W / S — Previous / next object",
                    "+ / Shift+- — Add object / remove object (asks when the slot has data)",
                    "Mouse wheel (over object grid, 33+ objects) — Scroll object list",
                    "[ / ] — Zoom display out / in",
                    "Middle-click — Select object under cursor (tracked masks)",
                    "Q / Esc — Quit",
                ),
            ),
        ),
        GuideSection(
            "9. Command-line options",
            (
                "python main.py [OPTIONS]",
                _bullets(
                    "-i, --video_path — Video, TIFF stack, or image folder",
                    "-m, --model_path — SAM weights (model_weights/)",
                    "-d, --device — cuda | mps | cpu",
                    "-s, --display_size — UI size. Omit to reuse the saved size",
                    "-b, --base_size_px — Encoder longest side (default 1344). SAM 2 native is 1024; SAM 3 / 3.1 native is 1008",
                    "-ar, --use_aspect_ratio — Keep aspect ratio (overrides the saved choice)",
                    "--square — Stretch each frame to a square (overrides the saved choice)",
                    "-f32, --use_float32 — Float32 on GPU (more VRAM). CPU already uses float32",
                    "--max_memories — Frame history depth (default 6)",
                    "--objscore_threshold — Lost threshold. Omit to reuse the saved value",
                    "--keep_bad_objscores — Keep inferencing after loss",
                    "--keep_history_on_new_prompts — Keep frame history on new Store Prompt",
                    "--encode_cache_size — CPU encode cache size (0 disables, default 64)",
                    "--reverse_buffer_size — Reverse-playback frame buffer (0 disables, default 120)",
                    "--mask_select — legacy (default) or official",
                    "--lost_patience — Consecutive low scores before a target is lost (default 1)",
                    "--max_prompt_attn — SAM 3 / 3.1 prompt memories used per frame",
                    "--intensity_range — 16-bit stills: auto, full, or low,high",
                    "--encode_cache_mb — CPU encode-cache megabyte cap (default 2048)",
                ),
            ),
        ),
        GuideSection(
            "10. Microscopy tips",
            (
                _bullets(
                    "Store prompts on a sharp, in-focus frame when possible",
                    "Use FG/BG for irregular cells; boxes for round cells",
                    "If drift builds up, disable History, clear frame memory, re-store on a good frame",
                    "Use a smaller -b if VRAM is limited",
                    "GPU strongly recommended for real-time tracking",
                ),
            ),
        ),
    ),
    "zh": (
        GuideSection(
            "1. 概述",
            (
                "Micro Tracker 3 是一款面向显微镜视频及通用场景的交互式桌面应用。"
                "可使用 SAM（Segment Anything Model）标注目标、在时间上传播实例掩膜，"
                "并导出 8 位灰度标签序列，供 ImageJ、TrackMate、MATLAB 或 Python 后续分析。",
                _bullets(
                    "最多 255 个独立对象槽位",
                    "通过内置 src/ 推理代码支持 SAM 2 / SAM 3 / SAM 3.1 权重",
                    "支持 CUDA、Apple MPS 或 CPU 推理",
                ),
            ),
        ),
        GuideSection(
            "2. 基本流程",
            (
                "步骤 1 — 加载模型与视频（界面顶栏按钮，或命令行 -m / -i）。",
                "步骤 2 — 选择对象槽位（Object 1、Object 2…；↑/↓ 或 W/S）。",
                "步骤 3 — 暂停播放（空格）。播放过程中无法使用提示工具。",
                "步骤 4 — 选择 Hover、Box、前景点或背景点，在目标上标注。",
                "步骤 5 — Store Prompt（Enter 或按钮，暂停时），将提示写入追踪记忆库。",
                "步骤 6 — 追踪：点击 Track 或空格播放；每到新帧自动更新掩膜。",
                "步骤 7 — 可选：开启 Enable Recording 后，用 Save Results 导出 TIF 标签。",
                "开始追踪前，可对多个对象重复步骤 2–5。",
            ),
        ),
        GuideSection(
            "3. 提示工具",
            (
                "Hover — 当前槽位尚无已存提示时，移动鼠标可预览掩膜。",
                "Box — 拖拽矩形框选目标。",
                "FG Point — 前景点（包含区域）。",
                "BG Point — 背景点（排除区域）。",
                "各工具的鼠标交互（详细）：",
                _bullets(
                    "Hover — 移动：实时掩膜预览。左键：追加 1 个前景点并切到 FG 工具（始终追加）。右键：追加 1 个背景点并切到 BG 工具（始终追加）。Hover 工具下 Shift 无效。",
                    "FG / BG — 左键：若无点则放置第一个点；否则替换最后一个点。Shift+左键：追加新点（多点提示请用此方式）。右键：删除离点击位置最近的点（Shift 无效）。",
                    "Box — 拖拽：用新框替换最近一个框。Shift+拖拽：追加新框（多框提示请用此方式）。右键：删除离点击位置最近角点所属的框。仅左键点击不拖动：会丢弃最近框且不新增（请避免）。",
                ),
                _bullets(
                    "Ctrl+Z — 撤销最近添加的一个前/背景点或框（Store 之前）",
                    "C — 仅清除屏幕上的提示（不删除已存入追踪器的记忆）",
                    "Enter / Store Prompt — 将当前交互提示保存到所选对象（暂停时）",
                ),
                "Store Prompt 条件：",
                _bullets(
                    "当前帧至少有一个前/背景点或一个框",
                    "对已 Store 的对象不能仅用 Hover 点作为提示",
                    "若 Store 无效，请切换到 FG/BG 或 Box 添加提示后再 Store",
                ),
            ),
        ),
        GuideSection(
            "4. 追踪记忆",
            (
                "Enable History（默认开启）— 仅在同一方向的相邻帧追加帧记忆；跳帧或换向会清空这段帧记忆，提示保留。",
                "Store Prompt 默认清掉当前对象的帧记忆；加上 --keep_history_on_new_prompts 才保留。",
                "Clear History — 清除当前对象的帧记忆；提示记忆保留。",
                "Clear Prompts — 清除当前对象已存的提示记忆。",
                "对所有已 Store 的对象，在播放、键盘单步或拖动时间轴松手后的新帧上都会运行追踪。",
            ),
        ),
        GuideSection(
            "5. 目标丢失（出画、离焦、遮挡）",
            (
                "SAM 每帧输出 object score；分数过低表示模型认为目标可能不可见。",
                "默认（不加 --keep_bad_objscores）：",
                _bullets(
                    "首次低于 --objscore_threshold（默认 0.0）的帧：仍推理一次，随后掩膜置零",
                    "记录该对象的停止帧索引",
                    "再次处于该帧及之后帧时，不再调用 step_video_masking，掩膜保持全零",
                    "提示记忆保留，直至用户恢复",
                ),
                "恢复方式：",
                _bullets(
                    "将播放头移到停止帧之前 — 清除停止标记，可重新追踪",
                    "暂停后添加新的前/背景点或框，再 Store Prompt — 清除停止并追加提示",
                ),
                "使用 --keep_bad_objscores — 丢失后每帧继续推理（尝试自动找回）；低分帧掩膜仍置零。",
            ),
        ),
        GuideSection(
            "6. 多对象与导出",
            (
                "丢失处理按对象独立；其他槽位不受影响。",
                "合并导出使用创建时分配的固定灰度：0=背景，对象编号不随删除前移。删除某对象会从已录帧里擦掉该灰度。",
                "重叠像素留给得分更高的对象；得分相同则保留较小编号。",
                "录制并 Save Results 生成的 TIF 以视频帧号命名（00240.tif 即第 240 帧），并附 frame_index.csv。",
                "若标签已保存但指标失败，内存中的帧会保留。Retry Metrics 把 CSV 和叠加视频写进同一文件夹。",
                "Save Session 会保存提示、模型路径、编码边长、是否拉伸成正方形，以及视频路径。Load Session 保留原来的对象编号，套用编码设置，并在同一段视频上重新编码。打开的不是该视频时会拒绝加载。未保存的录制可先保存、丢弃或取消。旧会话文件没有这些项时，沿用当前视频和编码设置。",
                "Retry Metrics 用失败那次的帧间隔和 16 位亮度范围，把没写完的指标再算一遍。",
                "仅保存已录制帧；丢失对象在掩膜为零的帧上对应标签 0。",
                "保存时会弹出导出对话框，输入帧率（fps）与像素尺寸（um/pixel）；进度窗口实时显示状态。",
                "除标签序列外，还会写出分析文件：",
                _bullets(
                    "tracking_metrics.csv — 每帧质心、面积、取向角（拟合椭圆长轴与 +X 夹角）、速度、位移",
                    "取向角在图像坐标中为顺时针，因为 y 向下增大",
                    "tracking_msd.csv — 每个对象的时间平均均方位移（MSD）",
                    "tracking_overlay.mp4 — 每个对象不同颜色轮廓 + 近 20 帧渐隐轨迹（对象出画后轨迹消失）",
                    "叠加颜色共 16 种，之后按对象编号循环",
                ),
            ),
        ),
        GuideSection(
            "6b. 对象侧栏（超过 32 个）",
            (
                "右侧面板以两列显示 Object 1、Object 2…",
                "超过 32 个对象时，列表行高与 32 个对象时一致，不再被压扁。",
                "在对象网格上滚动鼠标滚轮浏览；Enable Recording 与 Add/Remove、Save/Clear 固定在列表外。",
                "切换当前对象（侧栏、W/S、方向键或中键点选掩膜）时，列表会自动滚到该槽位。",
                "Object N 按钮的点击与悬停区域与可见标签一致（滚动后亦同）。",
            ),
        ),
        GuideSection(
            "7. 播放与时间轴",
            (
                _bullets(
                    "空格 — 播放 / 暂停",
                    "← / → 或 A / D — 暂停时单帧后退 / 前进",
                    "R — 倒放",
                    "时间轴滑块 — 拖动浏览；拖动时掩膜清空，松手后若有提示则刷新追踪",
                ),
                "拖动时间轴时屏幕掩膜会临时清空；松手后各对象仍遵守其停止帧规则。",
            ),
        ),
        GuideSection(
            "8. 快捷键",
            (
                _bullets(
                    "H — 打开/关闭本使用指南（中 / English）",
                    "F1 — 快捷键参考面板（OpenCV 窗口）",
                    "空格 — 播放 / 暂停",
                    "Enter — Store Prompt（暂停时）",
                    "Ctrl+Z — 撤销最近添加的提示（前/背景点或框）",
                    "C — 清除屏幕提示",
                    "Tab / Shift+Tab — 切换提示工具（向前 / 向后）",
                    "↑ / ↓ 或 W / S — 上 / 下一个对象",
                    "+ / Shift+- — 增加对象 / 删除对象（已有提示或标签时会确认）",
                    "鼠标滚轮（对象网格上，超过 32 个时）— 滚动对象列表",
                    "[ / ] — 缩小 / 放大显示",
                    "中键 — 点选掩膜下的对象",
                    "Q / Esc — 退出",
                ),
            ),
        ),
        GuideSection(
            "9. 命令行参数",
            (
                "python main.py [OPTIONS]",
                _bullets(
                    "-i, --video_path — 输入视频、TIFF 栈或图像文件夹",
                    "-m, --model_path — SAM 权重（model_weights/）",
                    "-d, --device — cuda | mps | cpu",
                    "-s, --display_size — 界面尺寸；省略则沿用已保存的值",
                    "-b, --base_size_px — 编码长边（默认 1344）。SAM 2 原生为 1024，SAM 3 / 3.1 原生为 1008",
                    "-ar, --use_aspect_ratio — 保持宽高比（覆盖已保存的选择）",
                    "--square — 将每帧拉伸为正方形（覆盖已保存的选择）",
                    "-f32, --use_float32 — GPU 上使用 float32（占用更多显存）。CPU 本身即为 float32",
                    "--max_memories — 帧历史深度（默认 6）",
                    "--objscore_threshold — 丢失判定阈值；省略则沿用已保存的值",
                    "--keep_bad_objscores — 丢失后继续推理",
                    "--keep_history_on_new_prompts — 新 Store 时保留帧历史",
                    "--encode_cache_size — CPU 编码缓存大小（0 关闭，默认 64）",
                    "--reverse_buffer_size — 倒放帧缓冲（0 关闭，默认 120）",
                    "--mask_select — legacy（默认）或 official",
                    "--lost_patience — 连续低分帧数达到该值才判为丢失（默认 1）",
                    "--max_prompt_attn — SAM 3 / 3.1 每帧使用的提示记忆条数",
                    "--intensity_range — 16 位静图：auto、full，或 low,high",
                    "--encode_cache_mb — CPU 编码缓存的兆字节上限（默认 2048）",
                ),
            ),
        ),
        GuideSection(
            "10. 显微镜使用建议",
            (
                _bullets(
                    "尽量在清晰、合焦的帧上 Store Prompt",
                    "不规则细胞宜用前/背景点，圆形细胞可用框",
                    "漂移明显时可关闭 History、清空帧记忆后在好帧重新 Store",
                    "显存不足可减小 -b",
                    "实时追踪强烈建议使用 GPU",
                ),
            ),
        ),
    ),
}


UI_STRINGS = {
    "en": {
        "window_title": "Micro Tracker 3 — User Guide",
        "header": "User Guide",
        "subtitle": "Press H to toggle  |  F1 for shortcut keys  |  Non-modal window",
        "lang_en": "English",
        "lang_zh": "中文",
        "footer": "Full markdown guide: docs/USER_GUIDE.md",
    },
    "zh": {
        "window_title": "Micro Tracker 3 — 使用指南",
        "header": "使用指南",
        "subtitle": "按 H 开关本窗口  |  F1 查看快捷键  |  非模态窗口",
        "lang_en": "English",
        "lang_zh": "中文",
        "footer": "完整 Markdown 文档：docs/USER_GUIDE.md",
    },
}


# ---------------------------------------------------------------------------------------------------------------------
# %% Window


class UserGuideWindow:
    """
    Non-modal tkinter user guide with English / Chinese content.
    Toggled with the H key from the main OpenCV window.
    """

    def __init__(self, app_name: str, version: str, author: str, author_email: str, initial_lang: str = "zh"):
        self._app_name = app_name
        self._version = version
        self._author = author
        self._author_email = author_email
        self._lang = initial_lang if initial_lang in GUIDE_CONTENT else "en"
        self._is_visible = False
        self._tk_available = True

        self._root = None
        self._top = None
        self._text = None
        self._header_label = None
        self._subtitle_label = None
        self._footer_label = None
        self._lang_var = None
        self._pending_toggle = False

        try:
            import tkinter as tk  # noqa: F401
        except ImportError:
            self._tk_available = False

    def is_visible(self) -> bool:
        return self._is_visible

    def _build_window(self):
        import tkinter as tk
        from tkinter import scrolledtext, ttk

        from ..tk_host import get_tk_root

        self._root = get_tk_root()
        if self._top is not None:
            return

        ui = UI_STRINGS[self._lang]
        self._top = tk.Toplevel(self._root)
        self._top.title(ui["window_title"])
        self._top.geometry("760x680")
        self._top.minsize(520, 400)
        self._top.protocol("WM_DELETE_WINDOW", self.hide)

        outer = ttk.Frame(self._top, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        header_row = ttk.Frame(outer)
        header_row.pack(fill=tk.X, pady=(0, 6))

        title_block = ttk.Frame(header_row)
        title_block.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._header_label = ttk.Label(title_block, text=ui["header"], font=("Arial", 14, "bold"))
        self._header_label.pack(anchor=tk.W)
        meta = f"{self._app_name} v{self._version}  |  {self._author} <{self._author_email}>"
        ttk.Label(title_block, text=meta, font=("Arial", 9)).pack(anchor=tk.W)
        self._subtitle_label = ttk.Label(title_block, text=ui["subtitle"], font=("Arial", 9))
        self._subtitle_label.pack(anchor=tk.W, pady=(4, 0))

        lang_frame = ttk.LabelFrame(header_row, text="Language / 语言", padding=6)
        lang_frame.pack(side=tk.RIGHT, padx=(12, 0))

        self._lang_var = tk.StringVar(value=self._lang)
        ttk.Radiobutton(
            lang_frame, text=UI_STRINGS["en"]["lang_en"], value="en", variable=self._lang_var, command=self._on_language_changed
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            lang_frame, text=UI_STRINGS["zh"]["lang_zh"], value="zh", variable=self._lang_var, command=self._on_language_changed
        ).pack(anchor=tk.W)

        self._text = scrolledtext.ScrolledText(
            outer,
            wrap=tk.WORD,
            font=("Arial", 10),
            padx=8,
            pady=8,
            state=tk.DISABLED,
            relief=tk.FLAT,
            borderwidth=1,
        )
        self._text.pack(fill=tk.BOTH, expand=True, pady=(4, 6))
        self._text.tag_configure("section", font=("Arial", 11, "bold"), spacing3=4)
        self._text.tag_configure("body", spacing1=2, spacing3=6)

        self._footer_label = ttk.Label(outer, text=ui["footer"], font=("Arial", 9))
        self._footer_label.pack(anchor=tk.W)

        self._fill_text()

    def _fill_text(self):
        if self._text is None:
            return

        import tkinter as tk

        sections = GUIDE_CONTENT[self._lang]
        self._text.configure(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)

        for section in sections:
            self._text.insert(tk.END, section.title + "\n", "section")
            for block in section.blocks:
                if isinstance(block, tuple):
                    for line in block:
                        self._text.insert(tk.END, f"  • {line}\n", "body")
                else:
                    self._text.insert(tk.END, block + "\n", "body")
            self._text.insert(tk.END, "\n", "body")

        self._text.configure(state=tk.DISABLED)
        self._text.yview_moveto(0.0)

    def _on_language_changed(self):
        if self._lang_var is None:
            return
        new_lang = self._lang_var.get()
        if new_lang not in GUIDE_CONTENT:
            return
        self._lang = new_lang
        ui = UI_STRINGS[self._lang]
        if self._top is not None:
            self._top.title(ui["window_title"])
        if self._header_label is not None:
            self._header_label.configure(text=ui["header"])
        if self._subtitle_label is not None:
            self._subtitle_label.configure(text=ui["subtitle"])
        if self._footer_label is not None:
            self._footer_label.configure(text=ui["footer"])
        self._fill_text()

    def show(self):
        if not self._tk_available:
            print("", "Warning: tkinter unavailable, cannot open user guide.", sep="\n", flush=True)
            return self

        if self._is_visible:
            return self

        self._build_window()
        self._top.deiconify()
        self._top.lift()
        self._is_visible = True
        return self

    def hide(self):
        if not self._is_visible or self._top is None:
            self._is_visible = False
            return self
        self._top.withdraw()
        self._is_visible = False
        return self

    def toggle(self):
        self.hide() if self._is_visible else self.show()
        return self

    def request_toggle(self):
        """Queue a show/hide toggle (safe to call from OpenCV waitKey callbacks)."""
        self._pending_toggle = True
        return self

    def process_events(self, display_window=None):
        """
        Run pending UI work and pump tkinter events.

        Must be called from the main loop after DisplayWindow.show() returns, not
        from inside keypress callbacks (avoids GIL errors with waitKeyEx on Windows).
        """
        if not self._tk_available:
            return self

        if self._pending_toggle:
            self._pending_toggle = False
            self.toggle()
            if display_window is not None:
                display_window.refocus()

        if self._is_visible and self._root is not None:
            try:
                self._root.update()
            except Exception:
                self.hide()

        return self

    def close(self):
        """Hide the guide. The shared Tk root stays alive for the dialogs."""

        if self._top is not None:
            try:
                self._top.destroy()
            except Exception:
                pass
            self._top = None
        self._text = None
        self._is_visible = False
        return self

    def attach_h_toggle(self, display_window) -> None:
        """Register H on the main display window to toggle this guide."""

        display_window.attach_keypress_callback("h", self.request_toggle)
