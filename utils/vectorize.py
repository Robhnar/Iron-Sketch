#!/usr/bin/env python3
"""
Edge path extraction from a binary edge image with custom adjacency and export to JS, Python, and SVG.

Zmiany zgodnie z prośbą:
- Zawsze zapisujemy trzy pliki **w katalogu wejściowego obrazu** z ustalonymi nazwami:
  * `11_vector_edges.js`
  * `12_vector_edges.py`
  * `13_vector_edges.svg`
- W plikach **JS** i **PY** każda tablica współrzędnych (pojedyncza łamana) jest w **nowej linii**,
  a **przed każdą** umieszczony jest komentarz z numerem ścieżki.

Użycie:
  python edge_paths.py <edge_image>

Wymagania: Python 3.9+, numpy, opencv-python, svgwrite
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import svgwrite

Point = Tuple[int, int]  # (x, y)


# ----------------------------- I/O helpers ---------------------------------

def load_binary_edges(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Nie można wczytać pliku: {path}")
    # białe = krawędzie (true), czarne = tło (false)
    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    return (bin_img > 0).astype(np.uint8)


# ------------------------- Graph construction -------------------------------

def four_neighbors(x: int, y: int, W: int, H: int) -> List[Point]:
    nbs = []
    if x > 0: nbs.append((x-1, y))
    if x < W-1: nbs.append((x+1, y))
    if y > 0: nbs.append((x, y-1))
    if y < H-1: nbs.append((x, y+1))
    return nbs


def diag_neighbors(x: int, y: int, W: int, H: int) -> List[Point]:
    nbs = []
    for dx in (-1, 1):
        for dy in (-1, 1):
            nx, ny = x+dx, y+dy
            if 0 <= nx < W and 0 <= ny < H:
                nbs.append((nx, ny))
    return nbs


def build_graph(mask: np.ndarray) -> Tuple[Dict[Point, List[Point]], List[List[Point]]]:
    """Zwraca (adjacency, components_as_lists_of_nodes).
    Reguła narożników obowiązuje symetrycznie dla obu końców.
    """
    H, W = mask.shape
    ys, xs = np.where(mask == 1)
    white: List[Point] = [(int(x), int(y)) for x, y in zip(xs, ys)]
    white_set = set(white)

    # policz liczbę sąsiadów krawędziowych dla każdego białego piksela
    edge_deg: Dict[Point, int] = {}
    for x, y in white:
        edge_deg[(x, y)] = sum((nx, ny) in white_set for nx, ny in four_neighbors(x, y, W, H))

    # zbuduj sąsiedztwo z regułą narożników
    adj: Dict[Point, List[Point]] = {p: [] for p in white}
    for x, y in white:
        # 4-neighbour: zawsze
        for nx, ny in four_neighbors(x, y, W, H):
            if (nx, ny) in white_set:
                adj[(x, y)].append((nx, ny))
        # diagonale: tylko jeśli oba końce mają <2 edge-neighbors
        if edge_deg[(x, y)] < 2:
            for nx, ny in diag_neighbors(x, y, W, H):
                if (nx, ny) in white_set and edge_deg[(nx, ny)] < 2:
                    adj[(x, y)].append((nx, ny))

    # Upewnij się, że graf jest nieskierowany i zgodny (filtruj jednostronne diagonale)
    for u in list(adj.keys()):
        adj[u] = [v for v in adj[u] if u in adj and v in adj and u != v]
    for u in list(adj.keys()):
        adj[u] = [v for v in adj[u] if u in adj.get(v, [])]

    # znajdź składowe spójne
    visited = set()
    components: List[List[Point]] = []
    for p in white:
        if p in visited:
            continue
        comp = []
        stack = [p]
        visited.add(p)
        while stack:
            s = stack.pop()
            comp.append(s)
            for v in adj[s]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        components.append(comp)

    return adj, components


# ------------------------- Path extraction ----------------------------------

def next_by_smallest_turn(prev: Optional[Point], cur: Point, candidates: List[Point]) -> Point:
    if prev is None or len(candidates) == 1:
        return candidates[0]
    vx, vy = cur[0] - prev[0], cur[1] - prev[1]
    def turn_cost(n: Point) -> float:
        nx, ny = n[0] - cur[0], n[1] - cur[1]
        dot = vx*nx + vy*ny
        nv = (vx*vx + vy*vy) ** 0.5
        nn = (nx*nx + ny*ny) ** 0.5
        if nv == 0 or nn == 0:
            return -1e9
        return dot / (nv*nn)
    candidates.sort(key=turn_cost, reverse=True)
    return candidates[0]


def extract_paths_in_component(adj: Dict[Point, List[Point]], nodes: List[Point]) -> List[List[Point]]:
    deg = {u: len(adj[u]) for u in nodes}
    unvisited = set(nodes)
    paths: List[List[Point]] = []

    def walk(start: Point):
        path: List[Point] = [start]
        unvisited.remove(start)
        prev: Optional[Point] = None
        cur = start
        while True:
            cand = [v for v in adj[cur] if v in unvisited]
            if not cand:
                break
            if prev is not None and len(cand) > 1 and prev in cand:
                cand_ = [v for v in cand if v != prev]
                if cand_:
                    cand = cand_
            nxt = next_by_smallest_turn(prev, cur, cand)
            path.append(nxt)
            unvisited.remove(nxt)
            prev, cur = cur, nxt
        return path

    # 1) Zacznij od końców (deg == 1)
    for u in list(unvisited):
        if u not in unvisited:
            continue
        if deg[u] == 1:
            paths.append(walk(u))

    # 2) Jeśli zostały wierzchołki (np. cykle), zacznij gdziekolwiek
    while unvisited:
        u = next(iter(unvisited))
        paths.append(walk(u))

    return paths


def extract_all_paths(mask: np.ndarray) -> List[List[Point]]:
    adj, comps = build_graph(mask)
    all_paths: List[List[Point]] = []
    for comp in comps:
        paths = extract_paths_in_component(adj, comp)
        all_paths.extend(paths)
    return all_paths


# --------------------------- Path optimization ------------------------------

def _point_line_distance(p: Point, a: Point, b: Point) -> float:
    """Odległość punktu p od prostej wyznaczonej przez odcinek a-b (w pikselach).
    Jeśli a == b, zwróć dystans euklidesowy do a.
    """
    ax, ay = a; bx, by = b; px, py = p
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = (vx*vx + vy*vy)
    if denom == 0:
        # segment zdegenerowany – odległość do punktu a
        dx, dy = px - ax, py - ay
        return (dx*dx + dy*dy) ** 0.5
    # Odległość do prostej (nie przycinamy do odcinka – tak, jak w opisie)
    # area = |v x w| = |vx*wy - vy*wx|
    area = abs(vx*wy - vy*wx)
    return area / (denom ** 0.5)


def _max_deviation(points: List[Point], i0: int, i1: int) -> float:
    """Maksymalna odległość punktów (i0<i<i1) od prostej i0–i1."""
    a = points[i0]; b = points[i1]
    if i1 - i0 <= 1:
        return 0.0
    m = 0.0
    for k in range(i0 + 1, i1):
        d = _point_line_distance(points[k], a, b)
        if d > m:
            m = d
    return m


def simplify_path_greedy(points: List[Point], eps: float) -> List[Point]:
    """Greedy segment-merging zgodnie z opisem użytkownika.
    Zachowuje kolejność, nie łączy ścieżek; minimalnie 2 punkty jeśli wejście ≥2.
    """
    n = len(points)
    if n <= 2 or eps <= 0:
        return points[:]
    out: List[Point] = []
    start = 0
    out.append(points[start])
    i = start + 1
    # zaczynamy od pierwszych dwóch punktów; i1 = aktualny koniec kandydującego segmentu
    i1 = start + 1
    while i1 < n:
        # próbujemy rozszerzać i1, dopóki max odchyłka ≤ eps
        # najpierw zapewnij co najmniej dwa punkty w segmencie
        while i1 < n and _max_deviation(points, start, i1) <= eps:
            i1 += 1
        # wyszliśmy, bo i1==n lub przekroczono eps na punkcie i1
        # akceptujemy segment do i1-1
        accept_to = max(start + 1, i1 - 1)
        out.append(points[accept_to])
        # nowy start
        start = accept_to
        i1 = start + 1
    # upewnij się, że ostatni punkt jest dodany
    if out[-1] != points[-1]:
        out.append(points[-1])
    # deduplikacja potencjalnego duplikatu końców
    dedup = [out[0]]
    for p in out[1:]:
        if p != dedup[-1]:
            dedup.append(p)
    return dedup

# ----------------------------- Exports --------------------------------------

def to_flat_xy(path: List[Point]) -> List[int]:
    out: List[int] = []
    for x, y in path:
        out.extend([int(x), int(y)])
    return out


def save_js(paths: List[List[Point]], out_js: Path) -> None:
    lines: List[str] = []
    lines.append("PATHS = [")
    names: List[str] = []
    # Per-path arrays
    for i, p in enumerate(paths, 1):
        xs = ", ".join(str(int(x)) for x, _ in p)
        ys = ", ".join(str(int(y)) for _, y in p)
        names.extend([f"path_{i}_x", f"path_{i}_y"])
        lines.append(f"\t// path {i}")
        lines.append(f"\t[{xs}],")
        lines.append(f"\t[{ys}],")
    lines.append("];")
    out_js.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_py(paths: List[List[Point]], out_py: Path) -> None:
    lines: List[str] = []
    lines.append("PATHS = [")
    names: List[str] = []
    for i, p in enumerate(paths, 1):
        xs = ", ".join(str(int(x)) for x, _ in p)
        ys = ", ".join(str(int(y)) for _, y in p)
        lines.append(f"\t# path {i}")
        lines.append(f"\t[[{xs}],")
        lines.append(f"\t [{ys}]],")
    lines.append("]")
    out_py.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_svg(paths: List[List[Point]], out_svg: Path, size: Tuple[int,int]) -> None:
    W, H = size
    dwg = svgwrite.Drawing(str(out_svg), size=(W, H), profile='tiny')
    dwg.add(dwg.rect(insert=(0,0), size=(W, H), fill="white"))
    for path in paths:
        if len(path) < 2:
            continue
        dwg.add(
            dwg.polyline(
                points=path,
                stroke="black",
                fill="none",
                stroke_width=0.5,
                stroke_linecap="round",
                stroke_linejoin="round",
            )
        )
    dwg.save()


def save_script(paths: List[List[Point]], out_script: Path) -> None:
    lines: List[str] = []
    lines.append("PATHS = [")
    names: List[str] = []
    # Per-path arrays
    for i, p in enumerate(paths, 1):
        xs = ", ".join(str(int(x)) for x, _ in p)
        ys = ", ".join(str(int(y)) for _, y in p)
        names.extend([f"path_{i}_x", f"path_{i}_y"])
        lines.append(f"\t// path {i}")
        lines.append(f"\t[{xs}],")
        lines.append(f"\t[{ys}],")
    lines.append("];")
    lines.append("var x0 = -825.0;")
    lines.append("var y0 = -115.0;")
    lines.append("var z0 = -363.7;")
    lines.append("var dx = 0;")
    lines.append("var dy = 0;")
    lines.append("var dz = 10;")
    lines.append("var m_v_move = 200;")
    lines.append("var m_v_draw = 50;")
    lines.append("var m_a = 1000;\n")
    lines.append("for (var i = 0; i < PATHS.length; i += 2) {")
    lines.append("\t// Oś Y obrazu rysujemy wzdłuż osi X robota.")
    lines.append("\tconst path_x = PATHS[i+1];")
    lines.append("\t// Oś X obrazu rysujemy wzdłuż osi Y robota.")
    lines.append("\tconst path_y = PATHS[i];")
    lines.append("\tif (path_x.length < 3) {")
    lines.append("\t\tcontinue;")
    lines.append("\t}")
    lines.append("\tmoveLinear('tcp', {x:x0+path_x[0], y:y0+path_y[0], z:z0+dz, rx:180.00, ry:0.00, rz:90.00}, m_v_move, m_a, {'precisely':false});")
    lines.append("\tconsole.log((x0+path_x[0]) + ', ' + (y0+path_y[0]));")
    lines.append("\tfor (var j = 0; j < path_x.length; j++) {")
    lines.append("\t\tmoveLinear('tcp', {x:x0+path_x[j], y:y0+path_y[j], z:z0, rx:180.00, ry:0.00, rz:90.00}, m_v_draw, m_a, {'precisely':false});")
    lines.append("\t}")
    lines.append("\tmoveLinear('tcp', {x:x0+path_x[path_x.length-1], y:y0+path_y[path_y.length-1], z:z0+dz, rx:180.00, ry:0.00, rz:90.00}, m_v_draw, m_a, {'precisely':false});")
    lines.append("}")
    out_script.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------- CLI ------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Wyznaczanie ścieżek z obrazu krawędzi (niestandardowa spójność) + zapis JS/PY/SVG w katalogu wejścia")
    ap.add_argument("edge_image", type=str, help="Ścieżka do binarnego obrazu krawędzi (białe=1, czarne=0)")
    ap.add_argument("--opt-eps", type=float, default=None, help="Opcjonalna tolerancja błędu dla optymalizatora ścieżek (w pikselach)")
    args = ap.parse_args()

    edge_path = Path(args.edge_image)
    out_dir = edge_path.parent
    out_js = out_dir / "11_vector_edges.js"
    out_py = out_dir / "12_vector_edges.py"
    out_svg = out_dir / "13_vector_edges.svg"
    out_script = out_dir / "14_vector_edges.script"

    mask = load_binary_edges(edge_path)
    H, W = mask.shape
    paths = extract_all_paths(mask)

    # Opcjonalna optymalizacja ścieżek (zachowuje liczbę ścieżek)
    if args.opt_eps is not None and args.opt_eps >= 0:
        before_pts = sum(len(p) for p in paths)
        paths = [simplify_path_greedy(p, args.opt_eps) for p in paths]
        after_pts = sum(len(p) for p in paths)
        print(f"Optymalizacja: punkty {before_pts} → {after_pts} (eps={args.opt_eps})")

    save_js(paths, out_js)
    save_py(paths, out_py)
    save_svg(paths, out_svg, size=(W, H))
    save_script(paths, out_script)

    print(f"Zapisano: {out_js}, {out_py}, {out_svg}")
    print(f"Ścieżek: {len(paths)}; pikseli (białych): {int(mask.sum())}")


if __name__ == "__main__":
    main()
