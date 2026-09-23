#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from collections import OrderedDict

import cv2

from .base import BaseCallback
from .helpers.images import blank_image, get_image_hw_for_max_side_length
from .helpers.text import TextDrawer

# For type hints
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


def create_VideoCapture(video_path):
    """
    Helper used to prevent bug introduced in newer versions of opencv,
    which disabled orientation correction on v4.11 for some reason, see:
        https://github.com/opencv/opencv/issues/26795
    """

    vcap = cv2.VideoCapture(video_path)
    vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    return vcap


class LoopingVideoReader:
    """
    Helper used to provide looping frames from video, along with helpers
    to control playback & frame sizing
    Example usage:

        vreader = LoopingVideoReader("path/to/video.mp4")
        for is_paused, frame_idx, frame in vreader:
            # Do something with frames...
            if i_want_to_stop:
                break

    """

    # .................................................................................................................

    def __init__(self, video_path: str, display_size_px: int | None = None, initial_position_0_to_1: float = 0.0):

        # Store basic video data
        self._video_path = video_path
        self._vcap = create_VideoCapture(self._video_path)
        self.total_frames = int(self._vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._max_frame_idx = self.total_frames - 1
        self._fps = self._vcap.get(cv2.CAP_PROP_FPS)

        # Enable rotated orientation fix (disabled by default on opencv v4.11 for reason)
        # See: https://github.com/opencv/opencv/issues/26795
        self._vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

        # Jump ahead to a different starting position if needed
        if initial_position_0_to_1 > 0.001:
            self._vcap.set(cv2.CAP_PROP_POS_FRAMES, self._max_frame_idx * initial_position_0_to_1)

        # Read sample frame & reset video
        rec_frame, first_frame = self._vcap.read()
        if not rec_frame:
            raise IOError(f"Can't read frames from video! ({video_path})")
        self._vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.sample_frame = first_frame

        # Set up display sizing
        self._need_resize = display_size_px is not None
        self._scale_wh = (first_frame.shape[1], first_frame.shape[0])
        if self._need_resize:
            out_h, out_w = get_image_hw_for_max_side_length(first_frame, display_size_px)
            self._scale_wh = (out_w, out_h)
        self.shape = (self._scale_wh[1], self._scale_wh[0], 3)

        # Allocate storage for 'previous frame', which is re-used when paused &
        self._is_paused = False
        self._frame_idx = 0
        self._pause_frame = self.scale_to_display_wh(first_frame) if self._need_resize else first_frame

        # Decoded-frame buffer (used to avoid expensive re-seeking, e.g. during reverse playback)
        # and a running record of which frame index the next vcap.read() will return.
        self._frame_buffer: "OrderedDict[int, ndarray]" = OrderedDict()
        self._frame_buffer_max = 0
        self._decoder_pos = 0

    # .................................................................................................................

    def scale_to_display_wh(self, image) -> ndarray:
        """Helper used to scale a given image to a target display size (if configured)"""
        return cv2.resize(image, dsize=self._scale_wh)

    # .................................................................................................................

    def set_frame_buffer_size(self, num_frames: int):
        """
        Set the maximum number of decoded frames to keep buffered.
        A larger buffer makes reverse playback / back-and-forth scrubbing smoother
        (avoids re-seeking the video) at the cost of additional system memory.
        Set to 0 to disable buffering.
        """
        self._frame_buffer_max = max(0, int(num_frames))
        while len(self._frame_buffer) > self._frame_buffer_max:
            self._frame_buffer.popitem(last=False)
        return self

    def clear_frame_buffer(self):
        self._frame_buffer.clear()
        return self

    def _buffer_store(self, frame_idx: int, frame: ndarray) -> None:
        if self._frame_buffer_max <= 0:
            return
        self._frame_buffer[frame_idx] = frame
        self._frame_buffer.move_to_end(frame_idx)
        while len(self._frame_buffer) > self._frame_buffer_max:
            self._frame_buffer.popitem(last=False)

    def _buffer_get(self, frame_idx: int) -> ndarray | None:
        if self._frame_buffer_max <= 0 or frame_idx not in self._frame_buffer:
            return None
        self._frame_buffer.move_to_end(frame_idx)
        return self._frame_buffer[frame_idx]

    def _read_frame_at(self, frame_idx: int) -> ndarray | None:
        """
        Return the decoded (display-scaled) frame at the given index, using the
        frame buffer when possible and only seeking the decoder when its position
        does not already match the requested frame. Returns None if the frame
        cannot be decoded.
        """

        # Serve from buffer without touching the decoder (keeps decoder position intact)
        cached_frame = self._buffer_get(frame_idx)
        if cached_frame is not None:
            return cached_frame

        # Only seek when the decoder isn't already positioned at the requested frame
        if self._decoder_pos != frame_idx:
            self._vcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self._decoder_pos = frame_idx

        read_ok, frame = self._vcap.read()
        if not read_ok:
            return None
        self._decoder_pos = frame_idx + 1

        if self._need_resize:
            frame = self.scale_to_display_wh(frame)
        self._buffer_store(frame_idx, frame)

        return frame

    # .................................................................................................................

    def release(self):
        """Close access to video source"""
        self._vcap.release()
        return self

    def pause(self, set_is_paused=True) -> bool:
        """Pause/unpause the video"""
        self._is_paused = set_is_paused
        return self._is_paused

    def toggle_pause(self) -> bool:
        """Helper used to toggle pause state (meant for keypress events)"""
        new_pause_state = not self._is_paused
        self._is_paused = new_pause_state
        return new_pause_state

    # .................................................................................................................

    def get_sample_frame(self) -> ndarray:
        """Helper used to retrieve a sample frame (the first frame), most likely for init use-cases"""
        return self.sample_frame.copy()

    def get_pause_state(self) -> bool:
        """Helper used to figure out if the video is paused (separately from the frame iteration)"""
        return self._is_paused

    def get_fps(self) -> float:
        """Helper used to report the framerate of the source video"""
        return self._fps

    def get_frame_delay_ms(self, max_allowable_ms=1000) -> int:
        """Returns a frame delay (in milliseconds) according to the video's reported framerate"""
        frame_delay_ms = 1000.0 / self._fps
        return int(min(max_allowable_ms, frame_delay_ms))

    # .................................................................................................................

    def get_playback_position(self, normalized=True) -> int | float:
        """Returns playback position either as a frame index or a number between 0 and 1 (if normalized)"""
        frame_idx = self._frame_idx
        if normalized:
            return frame_idx / self._max_frame_idx if self._max_frame_idx > 0 else 0.0
        return int(frame_idx)

    def get_frame_index(self) -> int:
        """Index of the frame currently shown (may differ from vcap read cursor after seek+read)."""
        return self._frame_idx

    def step_frame_by(self, delta: int) -> int:
        """Seek by delta frames while paused. Returns the resulting frame index."""
        if delta == 0:
            return self._frame_idx
        new_idx = max(0, min(self._max_frame_idx, self._frame_idx + delta))
        if new_idx == self._frame_idx:
            return self._frame_idx
        self.pause(True)
        self.set_playback_position(new_idx)
        return new_idx

    # .................................................................................................................

    def replace_source(self, video_path: str, display_size_px: int | None = None):
        """Replace the underlying video source and reset playback to the first frame."""

        self.release()
        self.__init__(video_path, display_size_px, initial_position_0_to_1=0.0)
        return self

    # .................................................................................................................

    def set_playback_position(self, position: int | float, is_normalized=False) -> int:
        """Set position of video playback. Returns frame index"""

        frame_idx = round(position * self._max_frame_idx) if is_normalized else position
        frame_idx = max(min(frame_idx, self._max_frame_idx), 0)

        self._frame_idx = frame_idx

        # If we're paused, but set a new frame, then update the pause frame
        # -> This is important for paused 'timeline scrubbing' to work intuitively
        if self._is_paused:
            frame = self._read_frame_at(frame_idx)
            if frame is not None:
                self._pause_frame = frame
        else:
            # Make sure the decoder will resume from the requested frame
            if self._decoder_pos != frame_idx:
                self._vcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                self._decoder_pos = frame_idx

        return frame_idx

    def get_current_frame(self) -> ndarray:
        """Return a copy of the frame currently held for display."""
        return self._pause_frame.copy()

    def read_frame(self, frame_idx: int) -> ndarray | None:
        """Return one decoded frame by index, or None when it cannot be decoded."""

        frame = self._read_frame_at(int(frame_idx))
        return None if frame is None else frame.copy()

    def _pause_at_frame(self, frame_idx: int) -> tuple[bool, int, ndarray]:
        """
        Seek to the given frame, enter pause state, and return the standard iterator tuple.

        Some containers report more frames than are actually decodable (e.g. variable-frame-rate
        or webm files), so a failed read is recovered by stepping back to the nearest readable
        frame (and clamping the usable range) instead of crashing the application.
        """

        self._is_paused = True
        target_idx = max(0, min(frame_idx, self._max_frame_idx))
        frame = None
        while target_idx >= 0:
            frame = self._read_frame_at(target_idx)
            if frame is not None:
                break
            # Reported frame is not decodable -> shrink the usable range and try the previous frame
            self._max_frame_idx = max(0, target_idx - 1)
            target_idx -= 1

        # If nothing at/below the target could be decoded, keep the last known-good frame
        if frame is None:
            return self._is_paused, self._frame_idx, self._pause_frame.copy()

        self._frame_idx = target_idx
        self._pause_frame = frame
        return self._is_paused, self._frame_idx, self._pause_frame.copy()

    # .................................................................................................................

    def __iter__(self):
        """Called when using this object in an iterator (e.g. for loops)"""
        if not self._vcap.isOpened():
            self._vcap = create_VideoCapture(self._video_path)
            self._decoder_pos = 0
        return self

    # .................................................................................................................

    def __next__(self) -> tuple[bool, int, ndarray]:
        """
        Iterator that provides frame data from a video capture object.
        Returns:
            is_paused, frame_index, frame_bgr
        """

        # Don't read video frames while paused
        if self._is_paused:
            return self._is_paused, self._frame_idx, self._pause_frame.copy()

        # Read next frame, or pause at the end instead of looping back to the beginning
        target_idx = self._frame_idx + 1
        if target_idx > self._max_frame_idx:
            return self._pause_at_frame(self._max_frame_idx)

        frame = self._read_frame_at(target_idx)
        if frame is None:
            return self._pause_at_frame(self._max_frame_idx)

        # Store the displayed frame in case we pause
        self._frame_idx = target_idx
        self._pause_frame = frame

        return self._is_paused, self._frame_idx, self._pause_frame.copy()

    # .................................................................................................................


class ReversibleLoopingVideoReader(LoopingVideoReader):
    """
    Simple variant on the basic looping reader.
    This version supports reading frames in reverse, though
    the implementation is extremely inefficient!

    To use, simply call the '.toggle_reverse_state(True)' function.
    (This can be done inside of a frame reading loop)
    """

    # .................................................................................................................

    def __init__(self, video_path: str, display_size_px: int | None = None, initial_position_0_to_1: float = 0.0):

        # Inherit from parent
        super().__init__(video_path, display_size_px, initial_position_0_to_1)

        # Flag used to keep track of playback direction
        self._is_reversed = False

    # .................................................................................................................

    def toggle_reverse_state(self, set_is_reversed: bool | None = None) -> bool:
        """
        Used to switch from forward-to-reverse frame reading
        If the given target state is None, then the current
        state will be toggled, otherwise it wil be set to
        the given state (True to reverse, False for forward reading).

        Returns: updated_reversal_state
        """

        self._is_reversed = (not self._is_reversed) if set_is_reversed is None else set_is_reversed

        return self._is_reversed

    # .................................................................................................................

    def get_reverse_state(self):
        """Used to check the current forward/reverse state"""
        return self._is_reversed

    # .................................................................................................................

    def __next__(self):
        """
        Iterator that provides frame data from a video capture object.
        Returns:
            is_paused, frame_index, frame_bgr
        """

        # Don't read video frames while paused
        if self._is_paused:
            return self._is_paused, self._frame_idx, self._pause_frame.copy()

        # Forward playback is handled by the (buffer-aware) parent implementation
        if not self._is_reversed:
            return super().__next__()

        # Step backward, or pause at the first frame instead of looping to the end
        if self._frame_idx <= 0:
            return self._pause_at_frame(0)

        target_idx = self._frame_idx - 1
        frame = self._read_frame_at(target_idx)
        if frame is None:
            return self._pause_at_frame(0)

        self._frame_idx = target_idx
        self._pause_frame = frame

        return self._is_paused, self._frame_idx, self._pause_frame.copy()

    # .................................................................................................................


class LoopingVideoPlaybackSlider(BaseCallback):
    """
    Implements a 'playback slider' UI element that is specific to working with videos.
    After initializing with a reference to the video reader whose playback is to be controlled,
    the playback slider only needs a single call (inside of the video loop) to work:
        slider.update(frame_index)

    This will update the slider position indicator (according to the given frame index) as well
    as internally keep track of changes to the slider (i.e. user adjustments).
    To check if the user is actively adjusting playback, use:
        slider.is_adjusting()
    """

    # .................................................................................................................

    def __init__(
        self,
        looping_video_reader: LoopingVideoReader,
        bar_height: int = 56,
        bar_bg_color=(38, 36, 44),
        stay_paused_on_change=False,
    ):
        # Store reference to video capture
        self._vreader = looping_video_reader
        self._total_frames = looping_video_reader.total_frames
        self._reader_pause_state = looping_video_reader.get_pause_state()
        self._stay_paused_on_change = stay_paused_on_change

        # Storage for slider value
        self._max_frame_idx = max(int(self._total_frames) - 1, 0)
        self._slider_idx = looping_video_reader.get_playback_position(normalized=False)

        # Storage for slider state
        self._x_px = 0
        self._is_pressed = False
        self._is_changed = False

        # Display config
        self._bar_bg_color = bar_bg_color
        self._track_margin_x = 12
        self._track_h = 14
        self._track_y_pad = 8
        self._label_y = 18
        self._last_render_w = 128
        self._track_x0 = self._track_margin_x
        self._track_w = 104
        self._txtdraw = TextDrawer(0.4, 1, (188, 186, 198), font=cv2.FONT_HERSHEY_DUPLEX)
        self._txtdraw_dim = TextDrawer(0.36, 1, (120, 118, 132), font=cv2.FONT_HERSHEY_DUPLEX)

        # Inherit from parent & set default helper name for debugging
        super().__init__(bar_height, 128, expand_w=True)

    # .................................................................................................................

    @staticmethod
    def _format_timecode(frame_idx: int, fps: float) -> str:
        if fps <= 0:
            return "--:--.--"
        total_sec = max(frame_idx, 0) / fps
        mins = int(total_sec // 60)
        secs = total_sec - mins * 60
        return f"{mins:02d}:{secs:05.2f}"

    def _slider_norm(self) -> float:
        if self._max_frame_idx <= 0:
            return 0.0
        return self._slider_idx / self._max_frame_idx

    def _track_geometry(self, h: int, w: int) -> tuple[int, int, int, int]:
        track_x0 = self._track_margin_x
        track_w = max(16, w - 2 * self._track_margin_x)
        track_y0 = h - self._track_y_pad - self._track_h
        track_y1 = track_y0 + self._track_h
        self._last_render_w = w
        self._track_x0 = track_x0
        self._track_w = track_w
        return track_x0, track_y0, track_w, track_y1

    # .................................................................................................................

    def is_adjusting(self):
        return self._is_pressed

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        track_x0, track_y0, track_w, track_y1 = self._track_geometry(h, w)
        track_cy = (track_y0 + track_y1) // 2
        img = blank_image(h, w, self._bar_bg_color)

        # Top/bottom chrome
        cv2.line(img, (0, 0), (w, 0), (22, 20, 26), 1, cv2.LINE_AA)
        cv2.line(img, (0, h - 1), (w, h - 1), (16, 14, 20), 1, cv2.LINE_AA)

        # Frame counter & timecode labels
        display_frame = self._slider_idx + 1
        frame_lbl = f"Frame {display_frame} / {self._total_frames}"
        fps = self._vreader.get_fps()
        time_lbl = f"{self._format_timecode(self._slider_idx, fps)} / {self._format_timecode(self._max_frame_idx, fps)}"
        img = self._txtdraw.xy_px(img, frame_lbl, (track_x0, self._label_y))
        time_w, _, _ = self._txtdraw.get_text_size(time_lbl)
        img = self._txtdraw.xy_px(img, time_lbl, (track_x0 + track_w - time_w, self._label_y))

        pct_lbl = f"{100.0 * self._slider_norm():.1f}%"
        pct_w, _, _ = self._txtdraw_dim.get_text_size(pct_lbl)
        pct_x = track_x0 + (track_w - pct_w) // 2
        img = self._txtdraw_dim.xy_px(img, pct_lbl, (pct_x, self._label_y - 1))

        # Track background
        cv2.rectangle(img, (track_x0, track_y0), (track_x0 + track_w, track_y1), (26, 28, 36), -1, cv2.LINE_AA)
        cv2.rectangle(img, (track_x0, track_y0), (track_x0 + track_w, track_y1), (58, 60, 72), 1, cv2.LINE_AA)

        # Played portion fill
        slider_norm = self._slider_norm()
        fill_x = track_x0 + int(round(slider_norm * track_w))
        if fill_x > track_x0:
            cv2.rectangle(img, (track_x0 + 1, track_y0 + 1), (fill_x, track_y1 - 1), (48, 105, 78), -1, cv2.LINE_AA)
            cv2.rectangle(img, (track_x0 + 1, track_y0 + 1), (fill_x, track_y0 + 4), (72, 145, 108), -1, cv2.LINE_AA)

        # Playhead thumb
        thumb_x = track_x0 + int(round(slider_norm * max(track_w - 1, 0)))
        thumb_r = 8 if self._is_pressed else 7
        thumb_fill = (245, 246, 250) if not self._is_pressed else (255, 255, 255)
        cv2.line(img, (thumb_x, track_y0 + 2), (thumb_x, track_y1 - 2), (230, 232, 238), 1, cv2.LINE_AA)
        cv2.circle(img, (thumb_x, track_cy), thumb_r + 1, (18, 20, 26), -1, cv2.LINE_AA)
        cv2.circle(img, (thumb_x, track_cy), thumb_r, thumb_fill, -1, cv2.LINE_AA)
        cv2.circle(img, (thumb_x, track_cy), thumb_r, (40, 44, 54), 1, cv2.LINE_AA)

        return img

    # .................................................................................................................

    def update(self, frame_index):
        """Helper used to update playback position if the slider changes, otherwise updates the indicator line"""

        is_changed, new_frame_idx = self.read()
        if is_changed:
            self._vreader.set_playback_position(new_frame_idx)
        else:
            self._slider_idx = frame_index

        return self

    # .................................................................................................................

    def read(self) -> tuple[bool, float | int]:
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, self._slider_idx

    def set(self, slider_value, use_as_default_value=True):
        new_value = max(0, min(self._max_frame_idx, slider_value))
        if use_as_default_value:
            self._initial_position = new_value
        self._slider_idx = new_value
        return self

    def step_forward(self, num_increments=1):
        return self.set(self._slider_idx + num_increments, use_as_default_value=False)

    def step_backward(self, num_decrements=1):
        return self.set(self._slider_idx - num_decrements, use_as_default_value=False)

    # .................................................................................................................

    def replace_video_reader(self, looping_video_reader: LoopingVideoReader):
        """Attach a new video reader and reset the slider to the first frame."""

        self._vreader = looping_video_reader
        self._total_frames = looping_video_reader.total_frames
        self._max_frame_idx = max(int(self._total_frames) - 1, 0)
        self._reader_pause_state = looping_video_reader.get_pause_state()
        self.set(0)
        return self

    # .................................................................................................................

    def on_left_down(self, cbxy, cbflags) -> None:

        # Ignore clicks outside of the slider
        if not cbxy.is_in_region:
            return

        # Prevent video playback while adjusting slider
        self._is_pressed = True
        self._reader_pause_state = self._vreader.get_pause_state()
        self._vreader.pause()

        # Record changes
        x_px = cbxy.xy_px[0]
        self._is_changed |= x_px != self._x_px
        if self._is_changed:
            self._x_px = x_px
            self._mouse_x_norm_to_slider_idx(cbxy.xy_norm[0])

        return

    def on_left_up(self, cbxy, cbflags) -> None:

        # Don't react to mouse up if we weren't being interacted with
        if not self._is_pressed:
            return
        self._is_pressed = False

        # Adjust pause state after user stops interacting with slider
        if self._stay_paused_on_change:
            self._reader_pause_state = self._vreader.pause()
        else:
            # Restore pause state (prior to modifying slider)
            # -> If the video was paused, this will keep it paused, otherwise unpause it
            self._reader_pause_state = self._vreader.pause(self._reader_pause_state)

        return

    def on_right_click(self, cbxy, cbflags) -> None:

        # Toggle pause when right clicked
        self._vreader.toggle_pause()

        return

    def on_drag(self, cbxy, cbflags) -> None:

        # Update slider value while dragging
        x_px = cbxy.xy_px[0]
        self._is_changed |= x_px != self._x_px
        if self._is_changed:
            self._x_px = x_px
            self._mouse_x_norm_to_slider_idx(cbxy.xy_norm[0])

        return

    # .................................................................................................................

    def _mouse_x_norm_to_slider_idx(self, x_norm: float) -> float | int:
        """Helper used to convert normalized mouse position into slider values"""

        if self._max_frame_idx <= 0:
            self._slider_idx = 0
            return self._slider_idx

        w = max(self._last_render_w, 1)
        track_x0_n = self._track_x0 / max(w - 1, 1)
        track_x1_n = (self._track_x0 + self._track_w - 1) / max(w - 1, 1)
        span = max(track_x1_n - track_x0_n, 1e-6)
        t = max(0.0, min(1.0, (x_norm - track_x0_n) / span))
        self._slider_idx = round(t * self._max_frame_idx)

        return self._slider_idx

    # .................................................................................................................


class ValueChangeTracker:
    """
    Simple helper object that be can be used to keep track of when a specific value of interest changes
    This is meant to be used to keep track of moments where potentially important state changes
    occur (e.g. pausing during video playback) and some important code may need to be run once
    in response to the change (but doesn't need to continue to run afterwards!)

    Main use is to check for changes using:
        value_changed = keeper.is_changed(curr_value, record_value=False)
    And then after responding to the change:
        if value_changed and some_other_condition:
            # do important/heavy work, then record value so we don't react to seeing it again
            keeper.record(curr_value)
    -> If the change will always be reacted to, it can be recorded during the .is_changed (...) check,
       which allows the later .record(...) to be left out!
    """

    # .................................................................................................................

    def __init__(self, initial_value=None):
        self._prev_value = initial_value

    def is_changed(self, value, record_value=False) -> bool:
        """Function used to check if the given value is different from the previously recorded value"""
        value_changed = value != self._prev_value
        if value_changed and record_value:
            self._prev_value = value
        return value_changed

    def record(self, value):
        """Records the given value for use in future 'did it change' checks"""
        self._prev_value = value
        return self

    def clear(self, clear_value=None):
        """Helper that is identical to 'record' but may be nicer to use to indicate intent!"""
        self._prev_value = clear_value

    # .................................................................................................................
