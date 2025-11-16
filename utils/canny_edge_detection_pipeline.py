#!/usr/bin/env python3
"""
Canny edge-detection pipeline with saved intermediate images and optional image resizing.
The gradient direction image is visualized in color (HSV hue-based map).

Usage:
  python canny_edge_detection_pipeline.py <image_path>
    [--low 0.1] [--high 0.3]
    [--ksize 5] [--sigma 1.4]
    [--resize 512]
    [--threshold-mode normalized|absolute]

If --resize is provided, the image is rescaled so that the smaller dimension equals the given size (e.g., 512 pixels).
Saves all intermediate steps and the final result into a directory named after
input file's stem (e.g., `photo.jpg` -> `photo/`). If that directory exists, a numeric
suffix will be appended (e.g., `photo_2/`).
"""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import deque
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
    """
    Klasyczny NMS dla Canny'ego.

    Parameters
    ----------
    mag : np.ndarray
        Mapa modułu gradientu (float), zalecane w zakresie [0, 1].
    angle_deg : np.ndarray
        Mapa kierunków w stopniach, zakres [0, 180).

    Returns
    -------
    np.ndarray
        Obraz o tej samej wielkości, wyzerowany poza lokalnymi maksimum.
    """
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
    """
    Zastosuj podwójny próg do obrazu po NMS.

    Parameters
    ----------
    img : np.ndarray
        Obraz (np.float32), w tej samej skali co progi (np. [0,1] lub „absolute”).
    low : float
        Dolny próg.
    high : float
        Górny próg.

    Returns
    -------
    np.ndarray
        Mapa etykiet: 0 = brak krawędzi, WEAK_VAL = słaba, STRONG_VAL = silna.
    """
    strong = (img >= high)
    weak = (img >= low) & (img < high)
    res = np.zeros_like(img, dtype=np.uint8)
    res[strong] = STRONG_VAL
    res[weak] = WEAK_VAL
    return res

def hysteresis(edge_map: np.ndarray) -> np.ndarray:
    H, W = edge_map.shape
    res = edge_map.copy()
    q: deque[tuple[int, int]] = deque(zip(*np.where(res == STRONG_VAL)))

    while q:
        i, j = q.popleft()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                y, x = i + di, j + dj
                if 0 <= y < H and 0 <= x < W and res[y, x] == WEAK_VAL:
                    res[y, x] = STRONG_VAL
                    q.append((y, x))

    res[res != STRONG_VAL] = 0
    return res

def save_image(path: Path, img: np.ndarray, normalize: bool = False) -> None:
    """
    Zapisz obraz do pliku.

    normalize=True – obraz jest najpierw skalowany do uint8 przez to_uint8().
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    out = to_uint8(img) if normalize else img
    cv2.imwrite(str(path), out)

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

def canny_pipeline(
    image_path: Path,
    out_root: Path,
    low_thresh: float = 0.1,
    high_thresh: float = 0.3,
    ksize: int = 5,
    sigma: float = 1.4,
    resize_dim: int | None = None,
    threshold_mode: str = "normalized",  # 'normalized' / 'absolute'
) -> None:
    # Walidacja parametrów
    if low_thresh <= 0 or high_thresh <= 0:
        raise ValueError("Thresholds must be positive.")
    if low_thresh >= high_thresh:
        raise ValueError("low_thresh must be < high_thresh.")
    if ksize <= 0:
        raise ValueError("ksize must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if resize_dim is not None and resize_dim <= 0:
        raise ValueError("resize_dim must be positive.")
    if threshold_mode not in ("normalized", "absolute"):
        raise ValueError("threshold_mode must be 'normalized' or 'absolute'.")

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

    # nie modyfikujemy oryginalnego ksize, używamy ksize_effective
    ksize_effective = ksize if ksize % 2 == 1 else ksize + 1
    blur = cv2.GaussianBlur(gray, (ksize_effective, ksize_effective), sigmaX=sigma, sigmaY=sigma)
    save_image(outdir / "02_blur.png", blur)

    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    save_image(outdir / "03_grad_x.png", gx, normalize=True)
    save_image(outdir / "04_grad_y.png", gy, normalize=True)

    mag = np.hypot(gx, gy)
    mag_norm = mag / (mag.max() + 1e-12)  # zakres ~[0,1]
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
    save_image(outdir / "05_magnitude.png", mag_norm, normalize=True)

    # Color visualization of direction (V = znormalizowana magnituda)
    dir_color = direction_to_color(ang, mag_norm)
    save_image(outdir / "06_direction_color.png", dir_color)

    # --- wybór skali progów dla NMS i double_threshold ---
    if threshold_mode == "normalized":
        # NMS i progi w skali [0,1]
        nms_src = mag_norm
        low_for_dt, high_for_dt = low_thresh, high_thresh
    else:  # 'absolute'
        # NMS i progi w skali mag (Sobel float32)
        nms_src = mag
        low_for_dt, high_for_dt = low_thresh, high_thresh

    nms = non_maximum_suppression(nms_src, ang)
    save_image(outdir / "07_nms.png", nms, normalize=True)

    dt = double_threshold(nms, low_for_dt, high_for_dt)
    save_image(outdir / "08_double_threshold.png", dt)

    edges_custom = hysteresis(dt)
    save_image(outdir / "09_edges_custom.png", edges_custom)

    # OpenCV Canny:
    if threshold_mode == "normalized":
        # progi w [0,1] → skala 0–255
        low_cv = int(round(low_thresh * 255))
        high_cv = int(round(high_thresh * 255))
    else:
        # 'absolute' – progi traktujemy jako wartości w skali obrazu uint8
        low_cv = int(round(low_thresh))
        high_cv = int(round(high_thresh))

    edges_cv = cv2.Canny(blur, low_cv, high_cv, L2gradient=True)
    save_image(outdir / "10_edges_opencv.png", edges_cv)

    params_txt = (
        f"Input: {image_path}\nOutput dir: {outdir}\n"
        f"Resized: {resize_dim if resize_dim else 'no'}\n"
        f"Gaussian ksize (effective): {ksize_effective} (requested: {ksize})\n"
        f"Gaussian sigma: {sigma}\n"
        f"Threshold mode: {threshold_mode}\n"
        f"Low threshold (pipeline): {low_for_dt}\n"
        f"High threshold (pipeline): {high_for_dt}\n"
        f"OpenCV Canny thresholds: low={low_cv}, high={high_cv}\n"
    )
    (outdir / "_params.txt").write_text(params_txt, encoding="utf-8")
    print(f"Saved intermediates and results to: {outdir}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Canny pipeline with saved intermediates and colored direction visualization "
            "(thresholds can be normalized [0,1] or absolute)"
        )
    )
    p.add_argument("image", type=str, help="Path to input image file")
    p.add_argument(
        "--low",
        type=float,
        default=0.1,
        help=(
            "Low threshold. For threshold-mode=normalized: in [0,1]. "
            "For threshold-mode=absolute: w skali wartości obrazu (np. 20)."
        ),
    )
    p.add_argument(
        "--high",
        type=float,
        default=0.3,
        help=(
            "High threshold. For threshold-mode=normalized: in [0,1]. "
            "For threshold-mode=absolute: w skali wartości obrazu (np. 60)."
        ),
    )
    p.add_argument("--ksize", type=int, default=5)
    p.add_argument("--sigma", type=float, default=1.4)
    p.add_argument(
        "--resize",
        type=int,
        default=200,
        help="Resize so min dimension = given value (e.g., 512)",
    )
    p.add_argument(
        "--threshold-mode",
        type=str,
        choices=["normalized", "absolute"],
        default="normalized",
        help=(
            "Sposób interpretacji progów: 'normalized' (domyślnie, progi w [0,1]) "
            "lub 'absolute' (progi w skali obrazu, np. 20/60)."
        ),
    )
    p.add_argument("--out", type=str, default=".")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    canny_pipeline(
        Path(args.image),
        Path(args.out),
        args.low,
        args.high,
        args.ksize,
        args.sigma,
        args.resize,
        args.threshold_mode,
    )
