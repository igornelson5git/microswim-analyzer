"""
Synthetic validation-video generator for MicroSwim Analyzer.

This script creates a full benchmark suite of microscopy-like synthetic videos with
known ground-truth object trajectories and known camera motion.

Generated trajectory classes:
    1. triangle_closed       - closed triangle path with sharp turns
    2. diagonal_open         - open diagonal trajectory for net-displacement validation
    3. sinusoidal_open       - smooth open sinusoidal trajectory
    4. smooth_loop_closed    - smooth closed loop for curvature/path validation
    5. random_walk_drift     - stochastic-like swimmer with directional drift

Generated camera-motion modes:
    1. static                - no camera motion
    2. perfect_follow        - camera follows object exactly
    3. lag_follow            - delayed camera following
    4. lag_jitter            - delayed following with jitter and turn perturbations

Outputs:
    synthetic_validation/outputs/<trajectory>__<camera>.mp4
    synthetic_validation/outputs/<trajectory>__<camera>_ground_truth.csv
    synthetic_validation/outputs/validation_manifest.csv

Run:
    python validation.py

Optional:
    python validation.py --out-dir synthetic_validation/outputs --fps 30 --duration 20
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass
class ValidationConfig:
    out_dir: str = "synthetic_validation/outputs"
    fps: int = 30
    duration_s: float = 20.0
    width: int = 900
    height: int = 700
    world_width: int = 1600
    world_height: int = 1300
    object_size: int = 22
    microns_per_pixel: float = 1.0
    seed: int = 42
    codec: str = "mp4v"
    add_noise: bool = True
    add_blur: bool = True
    draw_ground_truth_overlay: bool = False

    @property
    def n_frames(self) -> int:
        return int(round(self.fps * self.duration_s))

    @property
    def center(self) -> np.ndarray:
        return np.array([self.width / 2, self.height / 2], dtype=float)


# -----------------------------------------------------------------------------
# Trajectory generation
# -----------------------------------------------------------------------------

def interpolate_polyline(points: np.ndarray, n_frames: int) -> np.ndarray:
    """Interpolate a constant-speed trajectory along a polyline."""
    points = np.asarray(points, dtype=float)
    segments = points[1:] - points[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    total_length = float(lengths.sum())

    if total_length <= 0:
        raise ValueError("Trajectory path length is zero.")

    distances = np.linspace(0, total_length, n_frames)
    out = []

    for d in distances:
        remaining = d
        for i, length in enumerate(lengths):
            if remaining <= length:
                alpha = remaining / length if length > 0 else 0.0
                out.append(points[i] + alpha * segments[i])
                break
            remaining -= length
        else:
            out.append(points[-1])

    return np.asarray(out, dtype=float)


def trajectory_triangle_closed(cfg: ValidationConfig) -> np.ndarray:
    """
    Closed triangular trajectory with sharp turns.

    Best for:
        - path-length validation
        - turn handling
        - closed-loop drift accumulation

    Not ideal for:
        - net displacement percent error
        - meandering index percent error
    """
    points = np.array(
        [
            [220, 180],
            [680, 180],
            [450, 520],
            [220, 180],
        ],
        dtype=float,
    )
    return interpolate_polyline(points, cfg.n_frames)


def trajectory_diagonal_open(cfg: ValidationConfig) -> np.ndarray:
    """
    Open near-linear diagonal trajectory.

    Best for:
        - net displacement validation
        - speed validation
        - field alignment validation
        - directional persistence validation
    """
    points = np.array(
        [
            [260, 260],
            [1120, 860],
        ],
        dtype=float,
    )
    return interpolate_polyline(points, cfg.n_frames)


def trajectory_sinusoidal_open(cfg: ValidationConfig) -> np.ndarray:
    """
    Smooth open sinusoidal trajectory.

    Best for:
        - heading validation
        - smooth curvature validation
        - speed and path-length validation
    """
    t = np.linspace(0, 1, cfg.n_frames)
    x = 160 + 580 * t
    y = 350 + 120 * np.sin(2 * np.pi * 2.0 * t)
    return np.column_stack([x, y]).astype(float)


def trajectory_smooth_loop_closed(cfg: ValidationConfig) -> np.ndarray:
    """
    Smooth closed loop using an ellipse with mild harmonic deformation.

    Best for:
        - closed trajectory validation without sharp heading discontinuities
        - curvature and path-length validation
    """
    t = np.linspace(0, 2 * np.pi, cfg.n_frames)
    x = 450 + 220 * np.cos(t) + 35 * np.cos(3 * t)
    y = 350 + 150 * np.sin(t) + 25 * np.sin(2 * t)
    return np.column_stack([x, y]).astype(float)


def trajectory_random_walk_drift(cfg: ValidationConfig) -> np.ndarray:
    """
    Smooth pseudo-random trajectory with directional drift.

    Best for:
        - realistic swimmer-like movement
        - robustness testing
        - speed and trajectory-error evaluation
    """
    rng = np.random.default_rng(cfg.seed + 101)
    n = cfg.n_frames

    position = np.zeros((n, 2), dtype=float)
    position[0] = np.array([180, 350], dtype=float)

    velocity = np.array([1.3, 0.25], dtype=float)

    for i in range(1, n):
        velocity = (
            0.96 * velocity
            + 0.04 * np.array([1.4, 0.15])
            + rng.normal(0, 0.12, size=2)
        )

        speed = np.linalg.norm(velocity)
        if speed > 2.4:
            velocity *= 2.4 / speed
        if speed < 0.8:
            velocity *= 0.8 / max(speed, 1e-9)

        position[i] = position[i - 1] + velocity

        if position[i, 0] < 120 or position[i, 0] > 780:
            velocity[0] *= -0.8
        if position[i, 1] < 120 or position[i, 1] > 580:
            velocity[1] *= -0.8

        position[i, 0] = np.clip(position[i, 0], 120, 780)
        position[i, 1] = np.clip(position[i, 1], 120, 580)

    return position


TRAJECTORIES: Dict[str, Callable[[ValidationConfig], np.ndarray]] = {
    "triangle_closed": trajectory_triangle_closed,
    "diagonal_open": trajectory_diagonal_open,
    "sinusoidal_open": trajectory_sinusoidal_open,
    "smooth_loop_closed": trajectory_smooth_loop_closed,
    "random_walk_drift": trajectory_random_walk_drift,
}


# -----------------------------------------------------------------------------
# Camera simulation
# -----------------------------------------------------------------------------

def compute_camera_positions(
    world_pos: np.ndarray,
    mode: str,
    cfg: ValidationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return camera top-left world-coordinate position for each frame."""
    camera = np.zeros_like(world_pos, dtype=float)

    if mode == "static":
        return camera

    # Camera position required to keep object at screen center.
    target_camera = world_pos - cfg.center

    if mode == "perfect_follow":
        return target_camera

    if mode in {"lag_follow", "lag_jitter"}:
        alpha = 0.04
        camera[0] = target_camera[0] * 0.2

        for i in range(1, len(world_pos)):
            camera[i] = camera[i - 1] + alpha * (target_camera[i] - camera[i - 1])

            if mode == "lag_jitter":
                # Continuous baseline stage jitter.
                camera[i] += rng.normal(0, 1.6, size=2)

                # Low-frequency drift wobble.
                camera[i, 0] += 2.2 * np.sin(i * 0.035)
                camera[i, 1] += 1.8 * np.cos(i * 0.041)

                # Extra perturbation during directional changes.
                if i >= 2:
                    v1 = world_pos[i - 1] - world_pos[i - 2]
                    v2 = world_pos[i] - world_pos[i - 1]

                    turn_strength = np.linalg.norm(v2 - v1)

                    if turn_strength > 1.2:
                        # Sudden operator correction / stage overshoot.
                        camera[i] += rng.normal(
                            0,
                            min(18.0, 4.0 * turn_strength),
                            size=2,
                        )

                        # One-frame overshoot impulse.
                        camera[i] += 6.0 * (v2 - v1) / (np.linalg.norm(v2 - v1) + 1e-9)

        return camera

    raise ValueError(f"Unknown camera mode: {mode}")


CAMERA_MODES = ["static", "perfect_follow", "lag_follow", "lag_jitter"]


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def make_background(cfg: ValidationConfig, seed: int) -> np.ndarray:
    """Create a microscopy-like textured background with many trackable features."""
    rng = np.random.default_rng(seed)
    bg = np.full((cfg.world_height, cfg.world_width, 3), 212, dtype=np.uint8)

    # Small dust/texture points.
    for _ in range(1600):
        x = int(rng.integers(0, cfg.world_width))
        y = int(rng.integers(0, cfg.world_height))
        r = int(rng.integers(1, 3))
        shade = int(rng.integers(115, 235))
        cv2.circle(bg, (x, y), r, (shade, shade, shade), -1)

    # Faint fibers/debris.
    for _ in range(130):
        x1 = int(rng.integers(0, cfg.world_width))
        y1 = int(rng.integers(0, cfg.world_height))
        x2 = x1 + int(rng.integers(-55, 55))
        y2 = y1 + int(rng.integers(-55, 55))
        shade = int(rng.integers(145, 225))
        cv2.line(bg, (x1, y1), (x2, y2), (shade, shade, shade), 1)

    # Mild illumination gradient.
    xx = np.linspace(-1, 1, cfg.world_width)[None, :]
    yy = np.linspace(-1, 1, cfg.world_height)[:, None]
    gradient = 10 * (xx + 0.5 * yy)
    bg = np.clip(bg.astype(np.float32) + gradient[..., None], 0, 255).astype(np.uint8)

    return bg


def crop_world(world_img: np.ndarray, camera_xy: np.ndarray, cfg: ValidationConfig) -> np.ndarray:
    """Crop viewport from synthetic world according to camera position."""
    cx, cy = camera_xy
    x0 = int(round(cx))
    y0 = int(round(cy))

    output = np.full((cfg.height, cfg.width, 3), 212, dtype=np.uint8)

    src_x1 = max(0, x0)
    src_y1 = max(0, y0)
    src_x2 = min(world_img.shape[1], x0 + cfg.width)
    src_y2 = min(world_img.shape[0], y0 + cfg.height)

    dst_x1 = max(0, -x0)
    dst_y1 = max(0, -y0)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        output[dst_y1:dst_y2, dst_x1:dst_x2] = world_img[src_y1:src_y2, src_x1:src_x2]

    return output


def draw_triangle_object(
    frame: np.ndarray,
    screen_pos: np.ndarray,
    heading_rad: float,
    cfg: ValidationConfig,
) -> None:
    """Draw a dark triangular object with orientation matching movement heading."""
    cx, cy = screen_pos
    size = cfg.object_size

    points = np.array(
        [
            [size, 0],
            [-size * 0.72, -size * 0.58],
            [-size * 0.72, size * 0.58],
        ],
        dtype=float,
    )

    rot = np.array(
        [
            [np.cos(heading_rad), -np.sin(heading_rad)],
            [np.sin(heading_rad), np.cos(heading_rad)],
        ]
    )

    pts = points @ rot.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    pts = pts.astype(np.int32)

    # Slight halo to make it look microscopy-like but still trackable.
    cv2.fillConvexPoly(frame, pts, (35, 35, 35))
    cv2.polylines(frame, [pts], True, (0, 0, 0), 2)
    cv2.circle(frame, (int(round(cx)), int(round(cy))), max(2, size // 7), (15, 15, 15), -1)


def add_video_artifacts(frame: np.ndarray, rng: np.random.Generator, cfg: ValidationConfig) -> np.ndarray:
    """Add mild blur and noise to make tracking less artificially perfect."""
    out = frame
    if cfg.add_blur:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if cfg.add_noise:
        noise = rng.normal(0, 2.0, out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out


# -----------------------------------------------------------------------------
# Ground-truth metrics
# -----------------------------------------------------------------------------

def compute_heading(world_pos: np.ndarray) -> np.ndarray:
    dx = np.gradient(world_pos[:, 0])
    dy = np.gradient(world_pos[:, 1])
    return np.arctan2(dy, dx)


def compute_frame_metrics(world_pos: np.ndarray, camera_pos: np.ndarray, cfg: ValidationConfig) -> pd.DataFrame:
    n = len(world_pos)
    screen_pos = world_pos - camera_pos
    heading = compute_heading(world_pos)

    dx = np.full(n, np.nan)
    dy = np.full(n, np.nan)
    dt = np.full(n, np.nan)
    step = np.full(n, np.nan)
    speed = np.full(n, np.nan)

    dx[1:] = np.diff(world_pos[:, 0]) * cfg.microns_per_pixel
    dy[1:] = np.diff(world_pos[:, 1]) * cfg.microns_per_pixel
    dt[1:] = 1.0 / cfg.fps
    step[1:] = np.sqrt(dx[1:] ** 2 + dy[1:] ** 2)
    speed[1:] = step[1:] / dt[1:]

    heading_unwrapped = np.unwrap(heading)
    turn_deg = np.full(n, np.nan)
    turn_deg[1:] = np.degrees(np.diff(heading_unwrapped))

    total_path = np.nansum(step)
    net_disp = float(np.linalg.norm((world_pos[-1] - world_pos[0]) * cfg.microns_per_pixel))
    meandering = net_disp / total_path if total_path > 0 else np.nan

    return pd.DataFrame(
        {
            "frame": np.arange(n),
            "time_s": np.arange(n) / cfg.fps,
            "true_world_x_px": world_pos[:, 0],
            "true_world_y_px": world_pos[:, 1],
            "true_world_x_um": world_pos[:, 0] * cfg.microns_per_pixel,
            "true_world_y_um": world_pos[:, 1] * cfg.microns_per_pixel,
            "true_camera_x_px": camera_pos[:, 0],
            "true_camera_y_px": camera_pos[:, 1],
            "true_screen_x_px": screen_pos[:, 0],
            "true_screen_y_px": screen_pos[:, 1],
            "true_heading_rad": heading,
            "true_heading_deg": np.degrees(heading),
            "true_dx_um": dx,
            "true_dy_um": dy,
            "true_dt_s": dt,
            "true_step_distance_um": step,
            "true_speed_um_s": speed,
            "true_turn_deg": turn_deg,
            "true_total_path_um_full_video": total_path,
            "true_net_displacement_um_full_video": net_disp,
            "true_meandering_index_full_video": meandering,
            "microns_per_pixel": cfg.microns_per_pixel,
            "fps": cfg.fps,
        }
    )


# -----------------------------------------------------------------------------
# Video generation
# -----------------------------------------------------------------------------

def generate_one_video(
    trajectory_name: str,
    camera_mode: str,
    cfg: ValidationConfig,
) -> Dict[str, object]:
    if trajectory_name not in TRAJECTORIES:
        raise ValueError(f"Unknown trajectory: {trajectory_name}")
    if camera_mode not in CAMERA_MODES:
        raise ValueError(f"Unknown camera mode: {camera_mode}")

    os.makedirs(cfg.out_dir, exist_ok=True)

    combo_seed = abs(hash((trajectory_name, camera_mode, cfg.seed))) % (2**32)
    rng = np.random.default_rng(combo_seed)

    world_pos = TRAJECTORIES[trajectory_name](cfg)
    camera_pos = compute_camera_positions(world_pos, camera_mode, cfg, rng)
    screen_pos = world_pos - camera_pos
    heading = compute_heading(world_pos)

    background = make_background(cfg, seed=cfg.seed)

    base = f"{trajectory_name}__{camera_mode}"
    video_path = os.path.join(cfg.out_dir, f"{base}.mp4")
    csv_path = os.path.join(cfg.out_dir, f"{base}_ground_truth.csv")

    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*cfg.codec),
        cfg.fps,
        (cfg.width, cfg.height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {video_path}")

    for i in range(cfg.n_frames):
        frame = crop_world(background, camera_pos[i], cfg)
        frame = add_video_artifacts(frame, rng, cfg)

        draw_triangle_object(frame, screen_pos[i], heading[i], cfg)

        if cfg.draw_ground_truth_overlay:
            cv2.circle(
                frame,
                (int(round(screen_pos[i, 0])), int(round(screen_pos[i, 1]))),
                3,
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                frame,
                f"{trajectory_name} | {camera_mode}",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)

    writer.release()

    gt = compute_frame_metrics(world_pos, camera_pos, cfg)
    gt["trajectory_name"] = trajectory_name
    gt["camera_mode"] = camera_mode
    gt["video_file"] = os.path.basename(video_path)
    gt.to_csv(csv_path, index=False)

    total_path = float(gt["true_total_path_um_full_video"].iloc[0])
    net_disp = float(gt["true_net_displacement_um_full_video"].iloc[0])
    meandering = float(gt["true_meandering_index_full_video"].iloc[0])
    mean_speed = float(gt["true_speed_um_s"].mean(skipna=True))

    return {
        "trajectory_name": trajectory_name,
        "camera_mode": camera_mode,
        "video_path": video_path,
        "ground_truth_csv": csv_path,
        "frames": cfg.n_frames,
        "fps": cfg.fps,
        "duration_s": cfg.duration_s,
        "microns_per_pixel": cfg.microns_per_pixel,
        "true_total_path_um": total_path,
        "true_net_displacement_um": net_disp,
        "true_meandering_index": meandering,
        "true_mean_speed_um_s": mean_speed,
    }


def generate_suite(cfg: ValidationConfig) -> pd.DataFrame:
    manifest_rows: List[Dict[str, object]] = []

    print("Generating MicroSwim synthetic validation suite...")
    print(f"Output directory: {cfg.out_dir}")
    print(f"Frames per video: {cfg.n_frames}")

    for trajectory_name in TRAJECTORIES:
        for camera_mode in CAMERA_MODES:
            row = generate_one_video(trajectory_name, camera_mode, cfg)
            manifest_rows.append(row)
            print(f"Saved: {os.path.basename(row['video_path'])}")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(cfg.out_dir, "validation_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print("\nValidation suite complete.")
    print(f"Manifest: {manifest_path}")
    print(f"Total videos: {len(manifest)}")

    return manifest


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> ValidationConfig:
    parser = argparse.ArgumentParser(
        description="Generate synthetic ground-truth validation videos for MicroSwim Analyzer."
    )
    parser.add_argument("--out-dir", default="synthetic_validation/outputs")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--world-width", type=int, default=1600)
    parser.add_argument("--world-height", type=int, default=1300)
    parser.add_argument("--object-size", type=int, default=22)
    parser.add_argument("--microns-per-pixel", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--no-blur", action="store_true")
    parser.add_argument("--draw-ground-truth-overlay", action="store_true")

    args = parser.parse_args()

    return ValidationConfig(
        out_dir=args.out_dir,
        fps=args.fps,
        duration_s=args.duration,
        width=args.width,
        height=args.height,
        world_width=args.world_width,
        world_height=args.world_height,
        object_size=args.object_size,
        microns_per_pixel=args.microns_per_pixel,
        seed=args.seed,
        codec=args.codec,
        add_noise=not args.no_noise,
        add_blur=not args.no_blur,
        draw_ground_truth_overlay=args.draw_ground_truth_overlay,
    )


def main() -> None:
    cfg = parse_args()
    generate_suite(cfg)


if __name__ == "__main__":
    main()
