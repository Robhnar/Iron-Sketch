#!/usr/bin/env python3
"""
Canny edge-detection pipeline with saved intermediate images and optional image resizing.
The gradient direction image is visualized in color (HSV hue-based map).

Usage:
  python canny_pipeline.py <image_path> [--low 50] [--high 150] [--ksize 5] [--sigma 1.4] [--resize 512]

If --resize is provided, the image is rescaled so that the smaller dimension equals the given size (e.g., 512 pixels).
Saves all intermediate steps and the final result into a directory named after
input file's stem (e.g., `photo.jpg` -> `photo/`). If that directory exists, a numeric
suffix will be appended (e.g., `photo_2/`).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np

STRONG_VAL = 255
WEAK_VAL = 75

def ensure_output_dir(stem: str, base_dir: Path) -> Path:
    out = base_dir / stem
    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)
        return out
    i = 2
    while True:
        candidate = base_dir / f"{stem}_{i}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        i += 1

def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    m, M = float(np.min(img)), float(np.max(img))
    if M - m < 1e-12:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - m) / (M - m)
    return (norm * 255).astype(np.uint8)

def non_maximum_suppression(mag: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    H, W = mag.shape
    Z = np.zeros((H, W), dtype=np.float32)
    angle = angle_deg.copy()
    angle[angle < 0] += 180
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            q = 0.0
            r = 0.0
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a < 180):
                q = mag[i, j + 1]; r = mag[i, j - 1]
            elif 22.5 <= a < 67.5:
                q = mag[i + 1, j - 1]; r = mag[i - 1, j + 1]
            elif 67.5 <= a < 112.5:
                q = mag[i + 1, j]; r = mag[i - 1, j]
            elif 112.5 <= a < 157.5:
                q = mag[i - 1, j - 1]; r = mag[i + 1, j + 1]
            Z[i, j] = mag[i, j] if mag[i, j] >= q and mag[i, j] >= r else 0.0
    return Z

def double_threshold(img: np.ndarray, low: float, high: float) -> np.ndarray:
    strong = (img >= high)
    weak = (img >= low) & (img < high)
    res = np.zeros_like(img, dtype=np.uint8)
    res[strong] = STRONG_VAL
    res[weak] = WEAK_VAL
    return res

def hysteresis(edge_map: np.ndarray) -> np.ndarray:
    H, W = edge_map.shape
    res = edge_map.copy()
    strong_positions = list(zip(*np.where(res == STRONG_VAL)))
    while strong_positions:
        i, j = strong_positions.pop()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                y, x = i + di, j + dj
                if 0 <= y < H and 0 <= x < W:
                    if res[y, x] == WEAK_VAL:
                        res[y, x] = STRONG_VAL
                        strong_positions.append((y, x))
    res[res != STRONG_VAL] = 0
    return res

def save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)

def resize_to_min_dim(img: np.ndarray, min_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min_size / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def direction_to_color(angle_deg: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    """Convert direction map to HSV color representation (H=angle, V=magnitude)."""
    hue = ((angle_deg % 180) / 180.0 * 179).astype(np.uint8)
    sat = np.full_like(hue, 255, dtype=np.uint8)
    val = to_uint8(magnitude)
    hsv = cv2.merge([hue, sat, val])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def canny_pipeline(image_path: Path, out_root: Path, low_thresh: float = 50.0, high_thresh: float = 150.0, ksize: int = 5, sigma: float = 1.4, resize_dim: int | None = None) -> None:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    stem = image_path.stem
    outdir = ensure_output_dir(stem, out_root)

    if resize_dim:
        bgr = resize_to_min_dim(bgr, resize_dim)
        save_image(outdir / "00_resized.png", bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    save_image(outdir / "01_gray.png", gray)

    if ksize % 2 == 0:
        ksize += 1
    blur = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    save_image(outdir / "02_blur.png", blur)

    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    save_image(outdir / "03_grad_x.png", to_uint8(gx))
    save_image(outdir / "04_grad_y.png", to_uint8(gy))

    mag = np.hypot(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
    save_image(outdir / "05_magnitude.png", to_uint8(mag))

    # Color visualization of direction
    dir_color = direction_to_color(ang, mag)
    save_image(outdir / "06_direction_color.png", dir_color)

    nms = non_maximum_suppression(mag, ang)
    save_image(outdir / "07_nms.png", to_uint8(nms))

    dt = double_threshold(nms, low_thresh, high_thresh)
    save_image(outdir / "08_double_threshold.png", dt)

    edges_custom = hysteresis(dt)
    save_image(outdir / "09_edges_custom.png", edges_custom)

    edges_cv = cv2.Canny(blur, low_thresh, high_thresh, L2gradient=True)
    save_image(outdir / "10_edges_opencv.png", edges_cv)

    params_txt = (f"Input: {image_path}\nOutput dir: {outdir}\n"
                  f"Resized: {resize_dim if resize_dim else 'no'}\n"
                  f"Gaussian ksize: {ksize}\nGaussian sigma: {sigma}\n"
                  f"Low threshold: {low_thresh}\nHigh threshold: {high_thresh}\n")
    (outdir / "_params.txt").write_text(params_txt, encoding="utf-8")
    print(f"Saved intermediates and results to: {outdir}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Canny pipeline with saved intermediates and colored direction visualization")
    p.add_argument("image", type=str, help="Path to input image file")
    p.add_argument("--low", type=float, default=20.0)
    p.add_argument("--high", type=float, default=40.0)
    p.add_argument("--ksize", type=int, default=5)
    p.add_argument("--sigma", type=float, default=1.4)
    p.add_argument("--resize", type=int, default=200, help="Resize so min dimension = given value (e.g., 512)")
    p.add_argument("--out", type=str, default=".")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    canny_pipeline(Path(args.image), Path(args.out), args.low, args.high, args.ksize, args.sigma, args.resize)