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
                    "Up to 32 independent object slots",
                    "SAM 2 / SAM 3 / SAM 3.1 weights via the bundled muggled_sam stack",
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
                "Step 5 — Store Prompt (Tab or button). This writes prompts into the tracker memory bank.",
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
                _bullets(
                    "C — Clear on-screen prompts only (does not remove stored tracker memory)",
                    "Tab / Store Prompt — Save current interactive prompts to the selected object",
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
                "Enable History (default on) — Keeps a short deque of per-frame memory encodings (--max_memories, default 6).",
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
                "Combined export uses gray levels: 0 = background, 1 = Object 1, 2 = Object 2, … Later objects overwrite overlaps.",
                "Recording + Save Results writes {video}_MT-Results_{timestamp}/00001.tif, …",
                "Only recorded frames are saved. Lost objects contribute label 0 where their mask is zero.",
            ),
        ),
        GuideSection(
            "7. Playback and timeline",
            (
                _bullets(
                    "Space — Play / pause",
                    "A / D — Step one frame while paused",
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
                    "Tab — Store Prompt",
                    "C — Clear on-screen prompts",
                    "← / → — Switch prompt tool",
                    "↑ / ↓ or W / S — Previous / next object",
                    "+ / - — Add / remove object slot",
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
                    "-i, --video_path — Input video",
                    "-m, --model_path — SAM weights (model_weights/)",
                    "-d, --device — cuda | mps | cpu",
                    "-s, --display_size — UI size (default 900)",
                    "-b, --base_size_px — Encoder longest side (default 1344)",
                    "-ar, --use_aspect_ratio — Keep aspect ratio",
                    "-f32, --use_float32 — Float32 (more VRAM)",
                    "--max_memories — Frame history depth (default 6)",
                    "--objscore_threshold — Lost threshold (default 0.0)",
                    "--keep_bad_objscores — Keep inferencing after loss",
                    "--keep_history_on_new_prompts — Keep frame history on new Store Prompt",
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
                    "最多 32 个独立对象槽位",
                    "通过内置 muggled_sam 支持 SAM 2 / SAM 3 / SAM 3.1 权重",
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
                "步骤 5 — Store Prompt（Tab 或按钮），将提示写入追踪记忆库。",
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
                _bullets(
                    "C — 仅清除屏幕上的提示（不删除已存入追踪器的记忆）",
                    "Tab / Store Prompt — 将当前交互提示保存到所选对象",
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
                "Enable History（默认开启）— 保留若干帧的帧记忆编码（--max_memories，默认 6）。",
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
                "合并导出灰度：0=背景，1=对象1，2=对象2… 后绘对象覆盖重叠区域。",
                "录制并 Save Results 生成 {视频名}_MT-Results_{时间戳}/00001.tif 等。",
                "仅保存已录制帧；丢失对象在掩膜为零的帧上对应标签 0。",
            ),
        ),
        GuideSection(
            "7. 播放与时间轴",
            (
                _bullets(
                    "空格 — 播放 / 暂停",
                    "A / D — 暂停时单帧后退 / 前进",
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
                    "Tab — Store Prompt",
                    "C — 清除屏幕提示",
                    "← / → — 切换提示工具",
                    "↑ / ↓ 或 W / S — 上 / 下一个对象",
                    "+ / - — 增加 / 删除对象槽位",
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
                    "-i, --video_path — 输入视频",
                    "-m, --model_path — SAM 权重（model_weights/）",
                    "-d, --device — cuda | mps | cpu",
                    "-s, --display_size — 界面尺寸（默认 900）",
                    "-b, --base_size_px — 编码长边（默认 1344）",
                    "-ar, --use_aspect_ratio — 保持宽高比",
                    "-f32, --use_float32 — 使用 float32（占用更多显存）",
                    "--max_memories — 帧历史深度（默认 6）",
                    "--objscore_threshold — 丢失判定阈值（默认 0.0）",
                    "--keep_bad_objscores — 丢失后继续推理",
                    "--keep_history_on_new_prompts — 新 Store 时保留帧历史",
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

        if self._root is None:
            self._root = tk.Tk()
            self._root.withdraw()

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
        if self._top is not None:
            try:
                self._top.destroy()
            except Exception:
                pass
            self._top = None
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None
        self._is_visible = False
        return self

    def attach_h_toggle(self, display_window) -> None:
        """Register H on the main display window to toggle this guide."""

        display_window.attach_keypress_callback("h", self.request_toggle)
