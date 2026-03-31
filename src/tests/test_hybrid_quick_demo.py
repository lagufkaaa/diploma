import sys
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from core.data import Data
from solvers.hybrid_solver import HybridSolver
from solvers.hybrid_solver_width import HybridSolverWidth
from utils.hybrid_visualization import visualize_hybrid_result
from utils.hybrid_visualization_width import visualize_hybrid_result_width


DEMO_WIDTH = 200.0
DEMO_HEIGHT = 120.0
DEMO_S = 12
DEMO_GREEDY_DELTA_X = 25.0


def _build_demo_rectangles():
    """
    Readable rectangle-only dataset.
    Each piece has height 30, widths are 72/72/56 pattern.
    """
    widths = [72, 72, 56, 72, 72, 56, 72, 72]
    return [
        np.array([[0.0, 0.0], [float(w), 0.0], [float(w), 30.0], [0.0, 30.0]], dtype=float)
        for w in widths
    ]


def _assert_hybrid_improves(result: dict):
    stats = result.get("hybrid_stats", {})
    greedy_obj = float(stats.get("greedy_objective_value") or 0.0)
    final_obj = float(stats.get("final_objective_value") or 0.0)
    model_status = str(stats.get("model_status"))

    print(
        {
            "status": result.get("status"),
            "selected_solution": result.get("selected_solution"),
            "model_status": model_status,
            "greedy_objective_value": greedy_obj,
            "final_objective_value": final_obj,
            "free_space_improvement_percent": stats.get("free_space_improvement_percent"),
            "unpack_last_n": stats.get("unpack_last_n"),
            "random_sample_size": stats.get("random_sample_size"),
        }
    )

    assert model_status in {"OPTIMAL", "FEASIBLE"}, f"Unexpected model status: {model_status}"
    assert result.get("selected_solution") == "model"
    assert final_obj > greedy_obj + 1.0, "Model must improve greedy objective in this demo."


def test_hybrid_height_crop_demo_improves_and_visualizes():
    """
    Height-crop demo:
    - greedy packs part of rectangles with coarse delta_x;
    - hybrid unpacks last items;
    - model re-packs in the TOP cropped window only.
    """
    data = Data(_build_demo_rectangles(), R=1, parallel_nfp=False)
    solver = HybridSolver(
        data,
        height=DEMO_HEIGHT,
        width=DEMO_WIDTH,
        S=DEMO_S,
        greedy_delta_x=DEMO_GREEDY_DELTA_X,
    )

    result = solver.solve(
        unpack_last_n=4,
        crop_height=80.0,  # top 80 of 120 is used by model
        use_top_crop=True,
        free_space_improvement=0.0,
        solver_gap=0.1,
        model_time_limit_sec=6.0,
        stop_after_first_solution=False,
        model_enable_output=False,
        lock_greedy_unpacked=False,
        max_model_unfixed_items=20,
        random_iterations=1,
        random_seed=7,
        random_sample_size=10,
    )

    try:
        visualize_hybrid_result(
            data.items,
            result,
            width=DEMO_WIDTH,
            height=DEMO_HEIGHT,
            S=DEMO_S,
            show=True,
        )
    except Exception as e:
        print("visualization failed:", e)

    _assert_hybrid_improves(result)


def test_hybrid_width_crop_demo_improves_and_visualizes():
    """
    Width-crop demo:
    - greedy packs part of rectangles with coarse delta_x;
    - hybrid unpacks last items;
    - model re-packs in the RIGHT cropped window only.
    """
    data = Data(_build_demo_rectangles(), R=1, parallel_nfp=False)
    solver = HybridSolverWidth(
        data,
        height=DEMO_HEIGHT,
        width=DEMO_WIDTH,
        S=DEMO_S,
        greedy_delta_x=DEMO_GREEDY_DELTA_X,
    )

    result = solver.solve(
        unpack_last_n=3,
        crop_width=120.0,  # right 120 of 200 is used by model
        use_right_crop=True,
        free_space_improvement=0.0,
        solver_gap=0.1,
        model_time_limit_sec=6.0,
        stop_after_first_solution=False,
        model_enable_output=False,
        lock_greedy_unpacked=False,
        max_model_unfixed_items=20,
        random_iterations=1,
        random_seed=7,
        random_sample_size=6,
    )

    try:
        visualize_hybrid_result_width(
            data.items,
            result,
            width=DEMO_WIDTH,
            height=DEMO_HEIGHT,
            S=DEMO_S,
            show=True,
        )
    except Exception as e:
        print("visualization failed:", e)

    _assert_hybrid_improves(result)
