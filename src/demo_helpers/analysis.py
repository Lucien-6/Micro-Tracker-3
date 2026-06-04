#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os.path as osp
import csv
import math
from collections import defaultdict

import cv2
import numpy as np

# Typing
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Constants

# Distinct BGR colors used to draw per-object contours / trajectories in the overlay video
OVERLAY_COLORS = (
    (60, 220, 60),
    (60, 160, 255),
    (235, 90, 60),
    (40, 235, 235),
    (235, 70, 220),
    (235, 215, 70),
    (70, 70, 235),
    (150, 235, 70),
    (235, 150, 70),
    (150, 70, 235),
    (70, 205, 150),
    (205, 150, 235),
    (40, 170, 255),
    (190, 120, 255),
    (170, 170, 60),
    (90, 190, 120),
)

# CSV column headers
METRICS_HEADER = (
    "object_id",
    "frame_index",
    "time_s",
    "centroid_x_px",
    "centroid_y_px",
    "centroid_x_um",
    "centroid_y_um",
    "area_px",
    "area_um2",
    "orientation_deg",
    "velocity_x_um_per_s",
    "velocity_y_um_per_s",
    "speed_um_per_s",
    "net_displacement_um",
    "net_sq_displacement_um2",
)

MSD_HEADER = (
    "object_id",
    "lag_frames",
    "lag_time_s",
    "msd_um2",
    "num_pairs",
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Geometry / statistics helpers


def object_ids_in_label_image(label_img: ndarray) -> list[int]:
    """Return the sorted list of non-zero object ids present in a combined label image."""
    return [int(value) for value in np.unique(label_img) if value != 0]


def compute_object_stats(label_img: ndarray, object_id: int) -> dict | None:
    """
    Compute centroid, area and orientation for a single object in a label image.

    Orientation is the angle (degrees) of the major axis of the equivalent
    (second-moment) ellipse measured from the +X axis, in image pixel
    coordinates (y increases downward), in the range (-90, 90].

    Returns None if the object is not present (zero area).
    """

    mask = (label_img == object_id).astype(np.uint8)
    moments = cv2.moments(mask, binaryImage=True)
    area_px = moments["m00"]
    if area_px <= 0:
        return None

    centroid_x = moments["m10"] / area_px
    centroid_y = moments["m01"] / area_px

    # Orientation from central moments (eigenvector of the covariance matrix, major axis)
    mu20 = moments["mu20"] / area_px
    mu02 = moments["mu02"] / area_px
    mu11 = moments["mu11"] / area_px
    orientation_rad = 0.5 * math.atan2(2.0 * mu11, (mu20 - mu02))
    orientation_deg = math.degrees(orientation_rad)

    return {
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "area_px": float(area_px),
        "orientation_deg": orientation_deg,
    }


def build_object_trajectories(frames_dict: dict[int, ndarray]) -> dict[int, list[tuple[int, dict]]]:
    """
    Build per-object, time-ordered lists of per-frame statistics.

    Returns:
        { object_id: [(frame_index, stats_dict), ...sorted by frame...] }
    """

    per_object: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for frame_idx in sorted(frames_dict.keys()):
        label_img = frames_dict[frame_idx]
        for object_id in object_ids_in_label_image(label_img):
            stats = compute_object_stats(label_img, object_id)
            if stats is not None:
                per_object[object_id].append((int(frame_idx), stats))

    return dict(per_object)


# ---------------------------------------------------------------------------------------------------------------------
# %% Metric building


def build_metric_rows(
    per_object: dict[int, list[tuple[int, dict]]],
    pixel_size_um: float,
    fps: float,
) -> list[dict]:
    """Build per-object, per-frame metric rows (velocity, displacement, orientation, ...)."""

    metric_rows = []
    for object_id in sorted(per_object.keys()):
        sequence = per_object[object_id]
        if len(sequence) == 0:
            continue

        _, first_stats = sequence[0]
        start_x_um = first_stats["centroid_x"] * pixel_size_um
        start_y_um = first_stats["centroid_y"] * pixel_size_um

        prev_entry = None
        for frame_idx, stats in sequence:
            time_s = (frame_idx / fps) if fps > 0 else 0.0
            cx_um = stats["centroid_x"] * pixel_size_um
            cy_um = stats["centroid_y"] * pixel_size_um

            velocity_x = velocity_y = speed = 0.0
            if prev_entry is not None:
                prev_frame_idx, prev_stats = prev_entry
                dt = ((frame_idx - prev_frame_idx) / fps) if fps > 0 else 0.0
                dx_um = (stats["centroid_x"] - prev_stats["centroid_x"]) * pixel_size_um
                dy_um = (stats["centroid_y"] - prev_stats["centroid_y"]) * pixel_size_um
                if dt > 0:
                    velocity_x = dx_um / dt
                    velocity_y = dy_um / dt
                    speed = math.hypot(dx_um, dy_um) / dt

            net_dx = cx_um - start_x_um
            net_dy = cy_um - start_y_um
            net_disp = math.hypot(net_dx, net_dy)

            metric_rows.append(
                {
                    "object_id": object_id,
                    "frame_index": frame_idx,
                    "time_s": time_s,
                    "centroid_x_px": stats["centroid_x"],
                    "centroid_y_px": stats["centroid_y"],
                    "centroid_x_um": cx_um,
                    "centroid_y_um": cy_um,
                    "area_px": stats["area_px"],
                    "area_um2": stats["area_px"] * (pixel_size_um ** 2),
                    "orientation_deg": stats["orientation_deg"],
                    "velocity_x_um_per_s": velocity_x,
                    "velocity_y_um_per_s": velocity_y,
                    "speed_um_per_s": speed,
                    "net_displacement_um": net_disp,
                    "net_sq_displacement_um2": net_disp * net_disp,
                }
            )
            prev_entry = (frame_idx, stats)

    return metric_rows


def build_msd_rows(
    per_object: dict[int, list[tuple[int, dict]]],
    pixel_size_um: float,
    fps: float,
) -> list[dict]:
    """
    Build time-averaged Mean Squared Displacement rows per object.

    MSD(tau) = < |r(t + tau) - r(t)|^2 > averaged over all valid frame pairs
    separated by the given lag (in frames). Lags are capped at half the
    trajectory span so that each estimate retains reasonable statistics.
    """

    msd_rows = []
    for object_id in sorted(per_object.keys()):
        sequence = per_object[object_id]
        if len(sequence) < 2:
            continue

        frame_indices = np.array([frame_idx for frame_idx, _ in sequence], dtype=np.int64)
        xs_um = np.array([stats["centroid_x"] for _, stats in sequence], dtype=np.float64) * pixel_size_um
        ys_um = np.array([stats["centroid_y"] for _, stats in sequence], dtype=np.float64) * pixel_size_um

        num_points = len(frame_indices)
        max_lag = max(1, int((frame_indices[-1] - frame_indices[0]) // 2))

        sq_sum_by_lag: dict[int, float] = defaultdict(float)
        count_by_lag: dict[int, int] = defaultdict(int)
        for i in range(num_points):
            for j in range(i + 1, num_points):
                lag = int(frame_indices[j] - frame_indices[i])
                if lag > max_lag:
                    break
                sq_disp = (xs_um[j] - xs_um[i]) ** 2 + (ys_um[j] - ys_um[i]) ** 2
                sq_sum_by_lag[lag] += float(sq_disp)
                count_by_lag[lag] += 1

        for lag in sorted(sq_sum_by_lag.keys()):
            num_pairs = count_by_lag[lag]
            msd_rows.append(
                {
                    "object_id": object_id,
                    "lag_frames": lag,
                    "lag_time_s": (lag / fps) if fps > 0 else 0.0,
                    "msd_um2": sq_sum_by_lag[lag] / num_pairs,
                    "num_pairs": num_pairs,
                }
            )

    return msd_rows


# ---------------------------------------------------------------------------------------------------------------------
# %% CSV writers


def _write_csv(path: str, header: tuple, rows: list[dict]) -> None:
    with open(path, "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_metrics_csv(path: str, metric_rows: list[dict]) -> None:
    _write_csv(path, METRICS_HEADER, metric_rows)


def write_msd_csv(path: str, msd_rows: list[dict]) -> None:
    _write_csv(path, MSD_HEADER, msd_rows)


# ---------------------------------------------------------------------------------------------------------------------
# %% Overlay video


def _open_video_capture(video_path: str):
    """Open a video capture with orientation auto-correction enabled (see ui/video.py)."""
    vcap = cv2.VideoCapture(video_path)
    vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return vcap


def _create_video_writer(out_path: str, fps: float, frame_wh: tuple[int, int]):
    """
    Create a cv2.VideoWriter, falling back to MJPG/.avi if mp4 encoding is unavailable.
    Returns (writer, actual_out_path).
    """

    safe_fps = fps if fps and fps > 0 else 30.0
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), safe_fps, frame_wh)
    if writer.isOpened():
        return writer, out_path

    # Fall back to a widely-available codec/container
    fallback_path = osp.splitext(out_path)[0] + ".avi"
    writer = cv2.VideoWriter(fallback_path, cv2.VideoWriter_fourcc(*"MJPG"), safe_fps, frame_wh)
    return writer, fallback_path


def render_tracking_overlay_video(
    out_path: str,
    frames_dict: dict[int, ndarray],
    video_path: str,
    fps: float,
    trail_length: int = 20,
    contour_thickness: int = 2,
    trail_thickness: int = 2,
    progress_cb=None,
) -> tuple[str, int] | None:
    """
    Render a single overlay video showing every tracked object with a uniquely
    colored contour plus a fading trajectory (the most recent 'trail_length'
    recorded positions). Original frames are read back from the source video at
    the recorded frame indices.

    A trajectory is only drawn while its object is present in the current frame;
    once the object leaves the field of view, its trail is no longer shown.

    'progress_cb', if given, is called as progress_cb(stage, current, total).

    Returns (output_path, num_frames_written) or None if nothing was written.
    """

    sorted_frames = sorted(frames_dict.keys())
    if len(sorted_frames) == 0:
        return None

    frame_h, frame_w = frames_dict[sorted_frames[0]].shape[0:2]

    # Pre-compute per-object centroid history (ordered by frame)
    per_object = build_object_trajectories(frames_dict)
    centroid_history: dict[int, list[tuple[int, tuple[float, float]]]] = {}
    for object_id, sequence in per_object.items():
        centroid_history[object_id] = [
            (frame_idx, (stats["centroid_x"], stats["centroid_y"])) for frame_idx, stats in sequence
        ]

    vcap = _open_video_capture(video_path)
    writer, actual_out_path = _create_video_writer(out_path, fps, (frame_w, frame_h))
    if not writer.isOpened():
        vcap.release()
        raise IOError(f"Could not open a video writer for: {out_path}")

    num_written = 0
    decoder_pos = -1
    total_frames = len(sorted_frames)
    try:
        for frame_idx in sorted_frames:
            # Read original frame (seek only when not already positioned)
            if frame_idx != decoder_pos:
                vcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                decoder_pos = frame_idx
            read_ok, frame = vcap.read()
            decoder_pos = (decoder_pos + 1) if read_ok else -1

            if not read_ok or frame is None:
                frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            elif frame.shape[0:2] != (frame_h, frame_w):
                frame = cv2.resize(frame, (frame_w, frame_h))

            label_img = frames_dict[frame_idx]
            present_ids = set(object_ids_in_label_image(label_img))

            # Draw per-object contours
            for object_id in present_ids:
                color = OVERLAY_COLORS[(object_id - 1) % len(OVERLAY_COLORS)]
                mask = (label_img == object_id).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, color, contour_thickness, cv2.LINE_AA)

            # Draw fading trajectories (single composite blend per frame for efficiency).
            # Only objects currently in view get a trail, so trajectories disappear
            # once an object leaves the field of view.
            trail_layer = np.zeros_like(frame)
            alpha_canvas = np.zeros((frame_h, frame_w), dtype=np.float32)
            has_trail = False
            for object_id, history in centroid_history.items():
                if object_id not in present_ids:
                    continue
                color = OVERLAY_COLORS[(object_id - 1) % len(OVERLAY_COLORS)]
                points = [xy for hist_idx, xy in history if hist_idx <= frame_idx][-trail_length:]
                num_pts = len(points)
                if num_pts < 2:
                    continue
                has_trail = True
                for k in range(1, num_pts):
                    p0 = (int(round(points[k - 1][0])), int(round(points[k - 1][1])))
                    p1 = (int(round(points[k][0])), int(round(points[k][1])))
                    fade_alpha = k / (num_pts - 1)  # older segments -> lower alpha
                    cv2.line(trail_layer, p0, p1, color, trail_thickness, cv2.LINE_AA)
                    cv2.line(alpha_canvas, p0, p1, float(fade_alpha), trail_thickness, cv2.LINE_8)

            if has_trail:
                alpha3 = alpha_canvas[:, :, None]
                frame = (frame.astype(np.float32) * (1.0 - alpha3) + trail_layer.astype(np.float32) * alpha3)
                frame = frame.astype(np.uint8)

            writer.write(frame)
            num_written += 1
            if progress_cb is not None:
                progress_cb("Rendering overlay video", num_written, total_frames)

    finally:
        writer.release()
        vcap.release()

    return actual_out_path, num_written


# ---------------------------------------------------------------------------------------------------------------------
# %% Orchestration


def export_tracking_analysis(
    save_folder: str,
    frames_dict: dict[int, ndarray],
    video_path: str | None,
    fps: float,
    pixel_size_um: float,
    trail_length: int = 20,
    progress_cb=None,
) -> dict:
    """
    Write the metric CSV, MSD CSV and overlay video into 'save_folder'.

    'progress_cb', if given, is called as progress_cb(stage, current, total).

    Returns a summary dict describing what was written.
    """

    if progress_cb is not None:
        progress_cb("Computing metrics", 0, 1)

    per_object = build_object_trajectories(frames_dict)

    metric_rows = build_metric_rows(per_object, pixel_size_um, fps)
    msd_rows = build_msd_rows(per_object, pixel_size_um, fps)

    metrics_path = osp.join(save_folder, "tracking_metrics.csv")
    msd_path = osp.join(save_folder, "tracking_msd.csv")
    write_metrics_csv(metrics_path, metric_rows)
    write_msd_csv(msd_path, msd_rows)
    if progress_cb is not None:
        progress_cb("Computing metrics", 1, 1)

    overlay_result = None
    if video_path is not None:
        overlay_path = osp.join(save_folder, "tracking_overlay.mp4")
        overlay_result = render_tracking_overlay_video(
            overlay_path, frames_dict, video_path, fps, trail_length=trail_length, progress_cb=progress_cb
        )

    return {
        "num_objects": len(per_object),
        "num_metric_rows": len(metric_rows),
        "metrics_path": metrics_path,
        "msd_path": msd_path,
        "overlay_path": overlay_result[0] if overlay_result is not None else None,
        "overlay_frames": overlay_result[1] if overlay_result is not None else 0,
    }
