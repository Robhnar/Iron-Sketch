#!/usr/bin/env python3
"""
run_weld_from_image.py
----------------------

Pipeline:
  1) zdjęcie -> krawędzie (Canny)
  2) krawędzie -> wektorowe ścieżki
  3) zapis skryptu robota (14_vector_edges.script)

Użycie (z katalogu głównego projektu):
  (.venv) PS> python .\run_weld_from_image.py ^
      --image .\Nowy folder\draw2\female.jpg ^
      --resize 512 ^
      --low 0.25 --high 0.45 ^
      --ksize 9 --sigma 2.5

Potem w katalogu obrazu powstanie np.:
  Nowy folder\Nowy folder\draw2\female_5\14_vector_edges.script

Ten plik wgrywasz do robota i uruchamiasz na sterowniku.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys


def run_canny(
    image_path: Path,
    out_root: Path,
    resize: int,
    low: float,
    high: float,
    ksize: int,
    sigma: float,
    threshold_mode: str = "normalized",
) -> Path:
    """
    Uruchamia skrypt Canny i zwraca ścieżkę do katalogu wyjściowego (np. female_5).
    """
    script = out_root / "draw2" / "canny_edge_detection_pipeline.py"
    if not script.is_file():
        raise FileNotFoundError(f"Nie znaleziono {script}")

    cmd = [
        sys.executable,
        str(script),
        str(image_path),
        "--resize",
        str(resize),
        "--low",
        str(low),
        "--high",
        str(high),
        "--ksize",
        str(ksize),
        "--sigma",
        str(sigma),
        "--threshold-mode",
        threshold_mode,
        "--out",
        str(out_root / "draw2"),
    ]
    print("[INFO] Uruchamiam Canny:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    # katalog wyjściowy ma nazwę jak stem pliku, z ewentualnym sufiksem _2, _3...
    # znajdziemy najnowszy katalog zaczynający się od nazwy obrazka
    stem = image_path.stem  # np. "female"
    candidates = sorted((out_root / "draw2").glob(f"{stem}*"))
    if not candidates:
        raise RuntimeError(f"Nie znaleziono katalogu wyjściowego dla {stem}")
    outdir = candidates[-1]
    print(f"[INFO] Wyniki Canny zapisane w: {outdir}")
    return outdir


def run_vectorize(edge_image: Path) -> Path:
    """
    Uruchamia vectorize.py na podanym obrazie krawędzi i zwraca ścieżkę do
    pliku 14_vector_edges.script.
    """
    script = edge_image.parent.parent / "vectorize.py"  # draw2/vectorize.py
    if not script.is_file():
        # alternatywnie: draw2/vectorize.py w tym samym katalogu co edge_image
        script = edge_image.parent / "vectorize.py"
    if not script.is_file():
        raise FileNotFoundError(f"Nie znaleziono vectorize.py dla {edge_image}")

    cmd = [
        sys.executable,
        str(script),
        str(edge_image),
        "--opt-eps",
        "1.0",  # uproszczenie ścieżek (mniej punktów dla robota)
    ]
    print("[INFO] Uruchamiam vectorize:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    out_script = edge_image.parent / "14_vector_edges.script"
    if not out_script.is_file():
        raise RuntimeError(f"Brak pliku {out_script} po vectorize")
    print(f"[INFO] Skrypt robota zapisany jako: {out_script}")
    return out_script


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pipeline: zdjęcie -> krawędzie -> skrypt robota (spawanie po krawędziach)"
    )
    p.add_argument(
        "--image",
        type=str,
        required=True,
        help="Ścieżka do obrazu wejściowego (np. Nowy folder\\draw2\\female.jpg)",
    )
    p.add_argument(
        "--resize",
        type=int,
        default=512,
        help="Rozmiar mniejszego boku obrazu dla Canny (np. 512)",
    )
    p.add_argument(
        "--low",
        type=float,
        default=0.25,
        help="Dolny próg Canny (dla threshold-mode=normalized, w [0,1])",
    )
    p.add_argument(
        "--high",
        type=float,
        default=0.45,
        help="Górny próg Canny (dla threshold-mode=normalized, w [0,1])",
    )
    p.add_argument(
        "--ksize",
        type=int,
        default=9,
        help="Rozmiar kernela Gaussa (musi być nieparzysty, np. 5,7,9)",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=2.5,
        help="Sigma Gaussa (rozmycie; im większe, tym mniej detali tła)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    image_path = Path(args.image).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Obraz wejściowy nie istnieje: {image_path}")

    # 1) Canny: obraz -> krawędzie
    outdir = run_canny(
        image_path=image_path,
        out_root=project_root,
        resize=args.resize,
        low=args.low,
        high=args.high,
        ksize=args.ksize,
        sigma=args.sigma,
        threshold_mode="normalized",
    )

    # Tu wybieramy, którego obrazu krawędzi użyć (09_edges_custom.png lub 10_edges_opencv.png)
    edge_image = outdir / "09_edges_custom.png"
    if not edge_image.is_file():
        edge_image = outdir / "10_edges_opencv.png"
    if not edge_image.is_file():
        raise FileNotFoundError(f"Nie znaleziono obrazu krawędzi w {outdir}")

    # 2) Vectorize: krawędzie -> ścieżki + skrypt robota
    robot_script = run_vectorize(edge_image)

    print("\n[OK] Pipeline zakończony.")
    print("Teraz wgraj do robota plik:")
    print(f"    {robot_script}")
    print("i uruchom go na sterowniku, aby robot naspawał krawędzie z obrazu.")


if __name__ == "__main__":
    main()