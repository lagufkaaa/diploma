from typing import List, Tuple

import numpy as np


def _resolve_solution_y(solution: dict, idx: int, *, height: float, S: int) -> float:
    y_vals = solution.get("y")
    if isinstance(y_vals, list) and idx < len(y_vals):
        return float(y_vals[idx])

    s_vals = solution.get("s")
    if isinstance(s_vals, list) and idx < len(s_vals):
        h = float(height) / float(max(1, int(S)))
        return float(s_vals[idx]) * h

    deltas = solution.get("deltas")
    if isinstance(deltas, list) and idx < len(deltas):
        row = deltas[idx]
        if isinstance(row, list):
            h = float(height) / float(max(1, int(S)))
            strip = next((j for j, val in enumerate(row) if float(val) > 0.5), 0)
            return float(strip) * h

    return 0.0


def unpack_solution_coords(items, solution: dict, *, height: float, S: int) -> List[Tuple[int, float, float]]:
    p_vals = solution.get("p", [])
    x_vals = solution.get("x", [])
    packed = []

    for idx, _it in enumerate(items):
        if idx >= len(p_vals) or float(p_vals[idx]) <= 0.5:
            continue
        x_val = float(x_vals[idx]) if idx < len(x_vals) else 0.0
        y_val = _resolve_solution_y(solution, idx, height=float(height), S=int(S))
        packed.append((idx, x_val, y_val))

    return packed


def _is_solution_valid(solution: dict) -> bool:
    return (
        isinstance(solution, dict)
        and solution.get("objective_value") is not None
        and isinstance(solution.get("x"), list)
    )


def visualize_hybrid_result(
    items,
    hybrid_result: dict,
    *,
    width: float,
    height: float,
    S: int,
    show: bool = True,
    save_path: str | None = None,
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    final_solution = hybrid_result.get("final")
    model_solution = hybrid_result.get("model_result")
    status = str(hybrid_result.get("status", "UNKNOWN"))
    selected_solution = str(hybrid_result.get("selected_solution", ""))
    model_status = str(hybrid_result.get("hybrid_stats", {}).get("model_status", "UNKNOWN"))

    final_valid = _is_solution_valid(final_solution)
    model_valid = _is_solution_valid(model_solution)

    try:
        ncols = 3
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))
    except Exception:
        plt.switch_backend("Agg")
        ncols = 3
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    vis = hybrid_result.get("visualization", {})
    greedy_solution = vis.get("greedy_solution", {})
    if not greedy_solution:
        greedy_solution = hybrid_result.get("final", {}) or {}

    use_top_crop = bool(vis.get("use_top_crop", hybrid_result.get("hybrid_stats", {}).get("use_top_crop", False)))
    crop_h = float(vis.get("used_crop_height", hybrid_result.get("hybrid_stats", {}).get("used_crop_height", 0.0)))
    packing_y_min = float(
        vis.get(
            "packing_y_min",
            max(0.0, float(height) - crop_h) if use_top_crop else 0.0,
        )
    )
    packing_y_max = float(vis.get("packing_y_max", float(height)))
    packing_y_min = max(0.0, min(float(height), packing_y_min))
    packing_y_max = max(0.0, min(float(height), packing_y_max))
    if packing_y_min > packing_y_max:
        packing_y_min = packing_y_max

    candidate_indices = set(int(i) for i in vis.get("candidate_indices", []))
    fixed_indices = set(int(i) for i in vis.get("fixed_indices", []))

    def _setup_axis(ax, title: str):
        ax.axvline(x=0, color="black", linewidth=2)
        ax.axvline(x=width, color="black", linewidth=2)
        ax.axhline(y=0, color="black", linewidth=2)
        ax.axhline(y=height, color="black", linewidth=2)
        ax.set_xlim(-width * 0.05, width * 1.05)
        ax.set_ylim(-height * 0.05, height * 1.05)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_title(title)

    def _draw_solution(ax, solution: dict, *, allowed_indices=None, highlight_indices=None, alpha=0.65):
        packed = unpack_solution_coords(items, solution, height=height, S=S)
        packed = [
            (idx, x_val, y_val)
            for idx, x_val, y_val in packed
            if allowed_indices is None or idx in allowed_indices
        ]
        colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(1, len(packed))))

        for vis_idx, (idx, x_val, y_val) in enumerate(packed):
            arr = np.asarray(items[idx].points, dtype=float)
            coords = [(float(xx) + x_val, float(yy) + y_val) for xx, yy in arr]
            is_highlight = highlight_indices is not None and idx in highlight_indices
            poly = patches.Polygon(
                coords,
                closed=True,
                alpha=0.78 if is_highlight else alpha,
                facecolor="#fb923c" if is_highlight else colors[vis_idx],
                edgecolor="#7c2d12" if is_highlight else "black",
                linewidth=1.8 if is_highlight else 1.0,
                hatch="//" if is_highlight else None,
            )
            ax.add_patch(poly)
            ax.text(coords[0][0], coords[0][1], str(idx), color="white", fontsize=8, weight="bold")

    def _draw_cut_overlay(ax):
        if not use_top_crop:
            return
        y0 = packing_y_min
        y1 = packing_y_max
        if y0 <= 0.0 and y1 >= float(height):
            return

        # forbidden region below the cropped top container
        if y0 > 0.0:
            bottom_forbidden = patches.Rectangle(
                (0.0, 0.0),
                float(width),
                y0,
                facecolor="#f59e0b",
                alpha=0.14,
                edgecolor="#d97706",
                linewidth=1.6,
                linestyle="--",
                hatch="///",
            )
            ax.add_patch(bottom_forbidden)

        # forbidden region above the cropped container (if packing_y_max < height)
        if y1 < float(height):
            top_forbidden = patches.Rectangle(
                (0.0, y1),
                float(width),
                float(height) - y1,
                facecolor="#f59e0b",
                alpha=0.14,
                edgecolor="#d97706",
                linewidth=1.6,
                linestyle="--",
                hatch="///",
            )
            ax.add_patch(top_forbidden)

        # new model container (where repacking is allowed)
        allowed_rect = patches.Rectangle(
            (0.0, y0),
            float(width),
            max(0.0, y1 - y0),
            fill=False,
            edgecolor="#0f766e",
            linewidth=2.0,
        )
        ax.add_patch(allowed_rect)
        ax.axhline(y=y0, color="#0f766e", linewidth=2.0, linestyle="--")
        ax.axhline(y=y1, color="#0f766e", linewidth=2.0, linestyle="--")
        ax.text(
            0.02,
            0.98,
            f"Model window: y in [{y0:.1f}, {y1:.1f}]",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="#0f766e"),
        )

    # panel 1: original greedy backpack
    ax0 = axes[0]
    _draw_solution(ax0, greedy_solution, highlight_indices=candidate_indices)
    _draw_cut_overlay(ax0)
    _setup_axis(ax0, "Original backpack (greedy)")

    # panel 2: backpack after cut/unpack
    ax1 = axes[1]
    if fixed_indices:
        _draw_solution(ax1, greedy_solution, allowed_indices=fixed_indices, alpha=0.60)
    _draw_cut_overlay(ax1)
    _setup_axis(ax1, "Backpack after unpack + cut")

    # panel 3: final solution, model candidate, or explicit infeasible info
    ax2 = axes[2]
    if final_valid:
        _draw_solution(ax2, final_solution)
        _draw_cut_overlay(ax2)
        if selected_solution == "greedy":
            title = "Final solution (greedy fallback)"
        elif status == "NOT_PROVEN":
            title = "Final model solution (NOT_PROVEN)"
        else:
            title = "Final improved solution"
        _setup_axis(ax2, title)
        free_space = hybrid_result.get("hybrid_stats", {}).get("final_free_space_percent")
        if free_space is not None:
            ax2.text(
                0.02,
                0.98,
                f"Free: {float(free_space):.2f}%",
                transform=ax2.transAxes,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
            )
    elif model_valid:
        _draw_solution(ax2, model_solution)
        _draw_cut_overlay(ax2)
        _setup_axis(ax2, f"Model candidate ({status})")
        model_free = hybrid_result.get("hybrid_stats", {}).get("model_free_space_percent")
        if model_free is not None:
            ax2.text(
                0.02,
                0.98,
                f"Free: {float(model_free):.2f}%",
                transform=ax2.transAxes,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
            )
    else:
        _draw_cut_overlay(ax2)
        ax2.text(
            0.5,
            0.55,
            "Model returned no feasible packing",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            bbox=dict(facecolor="#fee2e2", alpha=0.7),
        )
        ax2.text(
            0.5,
            0.45,
            f"status={status}, model_status={model_status}",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            bbox=dict(facecolor="white", alpha=0.7),
        )
        _setup_axis(ax2, "Model stage")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes
