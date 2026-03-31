from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_FILE = PROJECT_ROOT / "data_car_mats" / "car_mats_4.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "generated_images"

COLOR_ALL = "#71c7ec"
COLOR_GREEDY = "#9aa0a6"
COLOR_RANDOM = "#e63946"
EDGE_COLOR = "#274c77"
BG_TOP = "#f9fcff"
BG_BOTTOM = "#d7ecff"


def _slugify_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return cleaned or "dataset"


def build_unique_output_path(output_dir: Path, data_file: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_stem = _slugify_name(Path(data_file).stem)
    base_name = f"greedy_random_selection_{source_stem}"
    candidate = output_dir / f"{base_name}.png"

    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        candidate = output_dir / f"{base_name}_{suffix}.png"
        if not candidate.exists():
            return candidate
        suffix += 1


def parse_items(file_path: Path) -> list[np.ndarray]:
    lines = file_path.read_text(encoding="utf-8").splitlines()
    items: list[np.ndarray] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("PIECE"):
            i += 1
            continue

        i += 1
        quantity_line = lines[i].strip()
        quantity = int(quantity_line.split()[1])

        i += 1
        vertex_line = lines[i].strip()
        vertex_count = int(vertex_line.split()[3])

        i += 1
        if not lines[i].strip().startswith("VERTICES"):
            raise ValueError(f"Expected VERTICES line, got: {lines[i]}")

        i += 1
        vertices: list[list[float]] = []
        for _ in range(vertex_count):
            x_str, y_str = lines[i].strip().split()
            vertices.append([float(x_str), float(y_str)])
            i += 1

        shape = np.asarray(vertices, dtype=float)
        for _ in range(quantity):
            items.append(shape.copy())

    if not items:
        raise ValueError(f"No items parsed from file: {file_path}")

    return items


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def estimate_container_size(items: list[np.ndarray], area_ratio: float) -> tuple[float, float]:
    widths = []
    heights = []
    total_area = 0.0

    for pts in items:
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        w = max(1e-6, float(max_xy[0] - min_xy[0]))
        h = max(1e-6, float(max_xy[1] - min_xy[1]))
        widths.append(w)
        heights.append(h)
        total_area += polygon_area(pts)

    ratio = max(0.15, min(0.95, float(area_ratio)))
    target_container_area = max(1.0, total_area * ratio)

    mean_w = float(np.mean(widths)) if widths else 1.0
    mean_h = float(np.mean(heights)) if heights else 1.0
    aspect = max(0.6, min(2.2, mean_w / max(1e-6, mean_h)))

    width = math.sqrt(target_container_area * aspect)
    height = target_container_area / max(1e-6, width)

    width = max(width, max(widths, default=1.0) * 1.08)
    height = max(height, max(heights, default=1.0) * 1.08)
    return float(width), float(height)


def greedy_select_indices_by_packing(
    items: list[np.ndarray],
    container_width: float,
    container_height: float,
) -> set[int]:
    # First-fit decreasing shelf greedy:
    # items are already sorted by area in `main`.
    # We place each next item into the best fitting existing row (shelf),
    # otherwise open a new shelf if there is vertical space.
    shelves: list[dict[str, float]] = []
    used_height = 0.0
    selected: set[int] = set()

    for idx, pts in enumerate(items):
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        w = max(1e-6, float(max_xy[0] - min_xy[0]))
        h = max(1e-6, float(max_xy[1] - min_xy[1]))

        if w > container_width or h > container_height:
            continue

        best_shelf_idx = None
        best_gap = None
        for shelf_idx, shelf in enumerate(shelves):
            can_fit = (h <= shelf["height"]) and ((shelf["x_cursor"] + w) <= container_width)
            if not can_fit:
                continue

            gap = shelf["height"] - h
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_shelf_idx = shelf_idx

        if best_shelf_idx is not None:
            shelves[best_shelf_idx]["x_cursor"] += w
            selected.add(idx)
            continue

        if used_height + h <= container_height:
            shelves.append({"height": h, "x_cursor": w})
            used_height += h
            selected.add(idx)

    return selected


def choose_random_from_remaining(
    remaining_indices: Iterable[int], count: int, seed: int | None
) -> set[int]:
    remaining = sorted(set(int(i) for i in remaining_indices))
    if not remaining or count <= 0:
        return set()

    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    sample_size = min(int(count), len(remaining))
    return set(rng.sample(remaining, sample_size))


def build_grid_layout(items: list[np.ndarray]):
    normalized_items: list[np.ndarray] = []
    dims: list[tuple[float, float]] = []

    for points in items:
        min_xy = points.min(axis=0)
        shifted = points - min_xy
        normalized_items.append(shifted)
        max_xy = shifted.max(axis=0)
        dims.append((float(max_xy[0]), float(max_xy[1])))

    n = len(items)
    cols = max(1, int(math.ceil(math.sqrt(n * 1.5))))
    rows = int(math.ceil(n / cols))

    max_w = max((w for w, _ in dims), default=1.0)
    max_h = max((h for _, h in dims), default=1.0)

    # Compact spacing: roughly 2x tighter than the previous layout.
    gap_x = max_w * 0.10
    gap_y = max_h * 0.10
    pad = max(max_w, max_h) * 0.12

    col_widths = [0.0 for _ in range(cols)]
    row_heights = [0.0 for _ in range(rows)]
    for idx, (w, h) in enumerate(dims):
        row = idx // cols
        col = idx % cols
        col_widths[col] = max(col_widths[col], w)
        row_heights[row] = max(row_heights[row], h)

    col_starts = [pad]
    for col in range(1, cols):
        prev = col_starts[col - 1] + col_widths[col - 1] + gap_x
        col_starts.append(prev)

    canvas_height = sum(row_heights) + max(0, rows - 1) * gap_y + 2 * pad

    placed_polygons: list[np.ndarray] = []
    for idx, points in enumerate(normalized_items):
        row = idx // cols
        col = idx % cols
        w, h = dims[idx]

        x0 = col_starts[col] + (col_widths[col] - w) * 0.5
        row_before = sum(row_heights[:row]) + row * gap_y
        row_bottom = canvas_height - pad - row_before - row_heights[row]
        y0 = row_bottom + (row_heights[row] - h) * 0.5

        placed_polygons.append(points + np.array([x0, y0]))

    canvas_width = col_starts[-1] + col_widths[-1] + pad
    return placed_polygons, canvas_width, canvas_height


def draw_visualization(
    items: list[np.ndarray],
    greedy_selected: set[int],
    random_selected: set[int],
    output_path: Path,
    show: bool,
):
    polygons, canvas_w, canvas_h = build_grid_layout(items)
    data_ratio = canvas_h / max(canvas_w, 1e-6)
    fig_width = 14.0
    legend_space = 0.55
    fig_height = max(4.5, fig_width * data_ratio + legend_space)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=BG_TOP)

    gradient = np.linspace(0.0, 1.0, 256).reshape(256, 1)
    cmap = LinearSegmentedColormap.from_list("soft_blue_bg", [BG_BOTTOM, BG_TOP])
    ax.imshow(
        gradient,
        extent=(0, canvas_w, 0, canvas_h),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        alpha=0.65,
        zorder=0,
    )

    for idx, poly_points in enumerate(polygons):
        face = COLOR_ALL
        if idx in greedy_selected:
            face = COLOR_GREEDY
        if idx in random_selected:
            face = COLOR_RANDOM

        poly = patches.Polygon(
            poly_points,
            closed=True,
            facecolor=face,
            edgecolor=EDGE_COLOR,
            linewidth=1.05,
            alpha=0.95,
        )
        ax.add_patch(poly)

        center = poly_points.mean(axis=0)
        ax.text(
            center[0],
            center[1],
            str(idx),
            ha="center",
            va="center",
            fontsize=7,
            color="#0b132b",
            alpha=0.9,
        )

    legend_handles = [
        patches.Patch(facecolor=COLOR_ALL, edgecolor=EDGE_COLOR, label="All items"),
        patches.Patch(facecolor=COLOR_GREEDY, edgecolor=EDGE_COLOR, label="Greedy selection"),
        patches.Patch(facecolor=COLOR_RANDOM, edgecolor=EDGE_COLOR, label="Random 10 of remaining"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        framealpha=0.95,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )
    bottom_margin = min(0.16, max(0.07, legend_space / fig_height + 0.02))
    fig.subplots_adjust(bottom=bottom_margin)
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, canvas_h)
    ax.set_aspect("equal")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Build a visualization where all items are blue, shelf-greedy packed are gray, and random remaining are red."
    )
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated PNG images.",
    )
    parser.add_argument(
        "--greedy-area-ratio",
        type=float,
        default=0.6,
        help="Approximate container area ratio for shelf-greedy packing.",
    )
    parser.add_argument("--container-width", type=float, default=None)
    parser.add_argument("--container-height", type=float, default=None)
    parser.add_argument("--random-count", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible random selection. Omit for full random.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    output_path = build_unique_output_path(args.output_dir, args.data_file)

    raw_items = parse_items(args.data_file)
    items = sorted(raw_items, key=polygon_area, reverse=True)
    if args.container_width is not None and args.container_height is not None:
        container_w = float(args.container_width)
        container_h = float(args.container_height)
    else:
        container_w, container_h = estimate_container_size(items, area_ratio=args.greedy_area_ratio)

    greedy_selected = greedy_select_indices_by_packing(
        items,
        container_width=container_w,
        container_height=container_h,
    )
    remaining = set(range(len(items))) - greedy_selected
    random_selected = choose_random_from_remaining(remaining, count=args.random_count, seed=args.seed)

    draw_visualization(
        items=items,
        greedy_selected=greedy_selected,
        random_selected=random_selected,
        output_path=output_path,
        show=args.show,
    )

    print(f"Saved image to: {output_path}")
    print(f"Total items: {len(items)}")
    print("Greedy mode: shelf first-fit decreasing")
    print(f"Greedy container size (W x H): {container_w:.2f} x {container_h:.2f}")
    print(f"Greedy selected: {len(greedy_selected)}")
    print(f"Random highlighted from remaining: {len(random_selected)}")
    print(f"Random indices: {sorted(random_selected)}")


if __name__ == "__main__":
    main()
