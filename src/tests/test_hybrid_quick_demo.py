import sys
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from core.data import Data
from solvers.hybrid_solver import HybridSolver
from utils.hybrid_visualization import visualize_hybrid_result


DEMO_WIDTH = 220.0
DEMO_HEIGHT = 120.0
DEMO_S = 6
DEMO_GREEDY_DELTA_X = 1.0
DEMO_DIMS = [
    (84, 54),
    (84, 46),
    (92, 30),
    (48, 50),
    (88, 28),
    (40, 46),
    (64, 26),
    (36, 54),
    (32, 42),
]


def _build_demo_rectangles():
    return [
        np.array([[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]], dtype=float)
        for w, h in DEMO_DIMS
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

    assert result.get("status") == "OK"
    assert model_status in {"OPTIMAL", "FEASIBLE"}, f"Unexpected model status: {model_status}"
    assert result.get("selected_solution") == "model"
    assert final_obj > greedy_obj + 1.0, "Model must improve greedy objective in this demo."


def test_hybrid_vertical_crop_demo_improves_and_visualizes():
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
        crop_height=80.0,
        use_top_crop=True,
        free_space_improvement=0.0,
        solver_gap=0.3,
        model_time_limit_sec=8.0,
        stop_after_first_solution=False,
        model_enable_output=False,
        lock_greedy_unpacked=False,
        max_model_unfixed_items=30,
        random_iterations=1,
        random_seed=7,
        random_sample_size=6,
    )

    visualize_hybrid_result(
        data.items,
        result,
        width=DEMO_WIDTH,
        height=DEMO_HEIGHT,
        S=DEMO_S,
        show=True,
    )

    _assert_hybrid_improves(result)
