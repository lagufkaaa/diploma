from typing import List, Tuple

import numpy as np


def _resolve_solution_y(solution: dict, idx: int, *, height: float, S: int) -> float:
    """Возвращает y-координату детали из разных форматов решения.

    Приоритет: y -> s (номер полосы) -> deltas (one-hot по полосам).
    """
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
    """Собирает размещенные детали в формате (idx, x, y) для отрисовки."""
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
    """Минимальная проверка, что решение можно безопасно визуализировать."""
    return (
        isinstance(solution, dict)
        and solution.get("objective_value") is not None
        and isinstance(solution.get("x"), list)
    )


def visualize_hybrid_result_width(
    items,
    hybrid_result: dict,
    *,
    width: float,
    height: float,
    S: int,
    show: bool = True,
    save_path: str | None = None,
):
    """Рисует 3 панели гибридной упаковки (crop by width): greedy -> cut/unpack -> final/model.

    Быстрые точки для правки визуализации:
    - геометрия осей/рамки/сетки: `_setup_axis`;
    - цвета, прозрачность, контуры и подписи деталей: `_draw_solution`;
    - стиль подсветки окна repack и запрещенных зон: `_draw_cut_overlay`;
    - тексты заголовков панелей и инфо-бейджей: блоки panel 1/2/3 ниже.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Ключевые решения и статусы из результата гибридного алгоритма.
    final_solution = hybrid_result.get("final")
    model_solution = hybrid_result.get("model_result")
    status = str(hybrid_result.get("status", "UNKNOWN"))
    selected_solution = str(hybrid_result.get("selected_solution", ""))
    model_status = str(hybrid_result.get("hybrid_stats", {}).get("model_status", "UNKNOWN"))

    final_valid = _is_solution_valid(final_solution)
    model_valid = _is_solution_valid(model_solution)

    try:
        # Стандартный режим отрисовки (интерактивный backend).
        ncols = 3
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))
    except Exception:
        # Fallback для headless-сред (например CI без GUI).
        plt.switch_backend("Agg")
        ncols = 3
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Дополнительная мета-информация для визуализации шага repack.
    vis = hybrid_result.get("visualization", {})
    greedy_solution = vis.get("greedy_solution", {})
    if not greedy_solution:
        # Без отдельного greedy-снимка используем final как fallback.
        greedy_solution = hybrid_result.get("final", {}) or {}

    use_right_crop = bool(vis.get("use_right_crop", hybrid_result.get("hybrid_stats", {}).get("use_right_crop", False)))
    crop_w = float(vis.get("used_crop_width", hybrid_result.get("hybrid_stats", {}).get("used_crop_width", 0.0)))
    packing_x_min = float(
        vis.get(
            "packing_x_min",
            max(0.0, float(width) - crop_w) if use_right_crop else 0.0,
        )
    )
    packing_x_max = float(vis.get("packing_x_max", float(width)))
    packing_x_min = max(0.0, min(float(width), packing_x_min))
    packing_x_max = max(0.0, min(float(width), packing_x_max))
    if packing_x_min > packing_x_max:
        packing_x_min = packing_x_max

    greedy_free = hybrid_result.get("hybrid_stats", {}).get("greedy_free_space_percent")
    if greedy_free is None:
        greedy_obj = greedy_solution.get("objective_value")
        container_area = float(width) * float(height)
        if greedy_obj is not None and container_area > 0.0:
            greedy_free = 100.0 * max(0.0, container_area - float(greedy_obj)) / container_area

    candidate_indices = set(int(i) for i in vis.get("candidate_indices", []))
    fixed_indices = set(int(i) for i in vis.get("fixed_indices", []))

    def _setup_axis(ax, title: str):
        # Контур контейнера. Можно сменить цвет/толщину рамки через color/linewidth.
        ax.axvline(x=0, color="black", linewidth=2)
        ax.axvline(x=width, color="black", linewidth=2)
        ax.axhline(y=0, color="black", linewidth=2)
        ax.axhline(y=height, color="black", linewidth=2)

        # Небольшие поля вокруг контейнера (5%), чтобы фигуры не "липли" к краю окна.
        # Если нужно плотнее кадрировать, уменьшай 0.05.
        ax.set_xlim(-width * 0.05, width * 1.05)
        ax.set_ylim(-height * 0.05, height * 1.05)

        # equal: сохраняем реальные пропорции X/Y, чтобы детали не искажались визуально.
        ax.set_aspect("equal")

        # Легкая сетка для ориентира; alpha можно поднять/опустить по вкусу.
        ax.grid(True, alpha=0.2)
        ax.set_title(title)

    def _draw_solution(ax, solution: dict, *, allowed_indices=None, highlight_indices=None, alpha=0.65):
        # allowed_indices: рисуем только выбранные детали (например fixed).
        # highlight_indices: визуально выделяем интересующие детали.
        packed = unpack_solution_coords(items, solution, height=height, S=S)
        packed = [
            (idx, x_val, y_val)
            for idx, x_val, y_val in packed
            if allowed_indices is None or idx in allowed_indices
        ]

        # Базовая палитра фигур. Можно заменить "tab20" на другую cmap matplotlib.
        colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(1, len(packed))))

        for vis_idx, (idx, x_val, y_val) in enumerate(packed):
            arr = np.asarray(items[idx].points, dtype=float)
            coords = [(float(xx) + x_val, float(yy) + y_val) for xx, yy in arr]
            is_highlight = highlight_indices is not None and idx in highlight_indices
            poly = patches.Polygon(
                coords,
                closed=True,
                # alpha для обычных/выделенных деталей настраивается здесь.
                alpha=0.78 if is_highlight else alpha,
                # Цвета выделения кандидатов на repack.
                facecolor="#fb923c" if is_highlight else colors[vis_idx],
                edgecolor="#7c2d12" if is_highlight else "black",
                linewidth=1.8 if is_highlight else 1.0,
                # Штриховка выделенных деталей (можно убрать, поставив None).
                hatch="//" if is_highlight else None,
            )
            ax.add_patch(poly)
            # Подпись индекса детали: цвет/размер/насыщенность меняются тут.
            ax.text(coords[0][0], coords[0][1], str(idx), color="white", fontsize=8, weight="bold")

    def _draw_cut_overlay(ax):
        # Визуализация окна repack:
        # x0..x1 - где модели разрешено переставлять детали после cut.
        if not use_right_crop:
            return
        x0 = packing_x_min
        x1 = packing_x_max
        if x0 <= 0.0 and x1 >= float(width):
            return

        # Запрещенная зона левее окна repack.
        # Если хочешь менее агрессивную подсветку, уменьшай alpha.
        if x0 > 0.0:
            left_forbidden = patches.Rectangle(
                (0.0, 0.0),
                x0,
                float(height),
                facecolor="#f59e0b",
                alpha=0.14,
                edgecolor="#d97706",
                linewidth=1.6,
                linestyle="--",
                hatch="///",
            )
            ax.add_patch(left_forbidden)

        # Запрещенная зона правее окна repack (если x1 < ширины контейнера).
        if x1 < float(width):
            right_forbidden = patches.Rectangle(
                (x1, 0.0),
                float(width) - x1,
                float(height),
                facecolor="#f59e0b",
                alpha=0.14,
                edgecolor="#d97706",
                linewidth=1.6,
                linestyle="--",
                hatch="///",
            )
            ax.add_patch(right_forbidden)

        # Контур разрешенного окна модели + вертикальные разделители.
        allowed_rect = patches.Rectangle(
            (x0, 0.0),
            max(0.0, x1 - x0),
            float(height),
            fill=False,
            edgecolor="#0f766e",
            linewidth=2.0,
        )
        ax.add_patch(allowed_rect)
        ax.axvline(x=x0, color="#0f766e", linewidth=2.0, linestyle="--")
        ax.axvline(x=x1, color="#0f766e", linewidth=2.0, linestyle="--")

        # Лейбл окна repack намеренно убран по требованию пользователя.

    def _draw_strip_rows(ax):
        # Горизонтальные границы strip-ов (S) для финальной панели.
        total_strips = max(1, int(S))
        if total_strips <= 1:
            return
        h_strip = float(height) / float(total_strips)
        for strip_idx in range(1, total_strips):
            y_line = float(strip_idx) * h_strip
            ax.plot(
                [0.0, float(width)],
                [y_line, y_line],
                color="#ef4444",
                linewidth=1.3,
                linestyle=":",
                alpha=0.9,
                zorder=4,
            )

    # Панель 1: исходная greedy-укладка (кандидаты подсвечиваются).
    ax0 = axes[0]
    _draw_solution(ax0, greedy_solution, highlight_indices=candidate_indices)
    # Меняй подпись панели здесь.
    _setup_axis(ax0, "Initial Greedy Packing")
    if greedy_free is not None:
        ax0.text(
            0.02,
            0.98,
            f"Free: {float(greedy_free):.2f}%",
            transform=ax0.transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
        )

    # Панель 2: greedy layout после распаковки последних N (без кандидатов unpack_last_n).
    ax1 = axes[1]
    if fixed_indices:
        _draw_solution(ax1, greedy_solution, allowed_indices=fixed_indices, alpha=0.60)
    _draw_cut_overlay(ax1)
    # Меняй подпись панели здесь.
    _setup_axis(ax1, "Greedy Layout After Unpack Last N")

    # Панель 3: финал, либо кандидат модели, либо сообщение об infeasible.
    ax2 = axes[2]
    if final_valid:
        _draw_solution(ax2, final_solution)
        _draw_cut_overlay(ax2)
        if selected_solution == "greedy":
            title = "Final Packing (Greedy Baseline)"
        elif status == "NOT_PROVEN":
            title = "Final Packing (Hybrid, Not Proven Optimal)"
        else:
            title = "Final Packing (Hybrid Improved)"
        # Меняй текст заголовка финальной панели здесь.
        _setup_axis(ax2, title)
        free_space = hybrid_result.get("hybrid_stats", {}).get("final_free_space_percent")
        if free_space is not None:
            # Инфо-бейдж со свободной площадью (позиция/стиль задаются тут).
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
        _setup_axis(ax2, f"Candidate Packing from Model ({status})")
        model_free = hybrid_result.get("hybrid_stats", {}).get("model_free_space_percent")
        if model_free is not None:
            # Инфо-бейдж для кандидатного решения модели.
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
        # Сценарий, когда модель не вернула допустимую укладку.
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
        _setup_axis(ax2, "Model Search Outcome")

    # По требованию: линии strip-ов только на последней панели.
    _draw_strip_rows(ax2)

    if save_path:
        # При необходимости сохраняем итоговую картинку на диск.
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        # В интерактивном режиме открываем окно matplotlib.
        plt.show()

    return fig, axes
