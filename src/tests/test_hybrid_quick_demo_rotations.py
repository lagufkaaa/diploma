import sys
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from core.data import Data
from solvers.hybrid_solver_width import HybridSolverWidth
from utils.hybrid_visualization_width import visualize_hybrid_result_width


ROT_DEMO_WIDTH = 200.0
ROT_DEMO_HEIGHT = 120.0
ROT_DEMO_S = 12
ROT_DEMO_GREEDY_DELTA_X = 1.0
ROT_DEMO_R = 4  # 0/90/180/270 degrees


def _build_rotation_demo_rectangles():
    """
    Hand-picked rectangle set where model improvement is stable and
    at least one item is packed with a right-angle rotation (90/270).
    """
    dims = [
        (80, 25),
        (72, 30),
        (40, 55),
        (35, 60),
        (60, 35),
        (56, 30),
        (60, 35),
        (40, 55),
    ]
    return [
        np.array([[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]], dtype=float)
        for w, h in dims
    ]


def _build_rotation_demo_rectangles_plus4():
    """
    Extended dataset (+4 rectangles) for 'small delta_x' behavior inspection.
    """
    dims = [
        (80, 25),
        (72, 30),
        (40, 55),
        (35, 60),
        (60, 35),
        (56, 30),
        (60, 35),
        (40, 55),
        (72, 30),
        (56, 30),
        (60, 35),
        (40, 55),
    ]
    return [
        np.array([[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]], dtype=float)
        for w, h in dims
    ]


def _used_rotations_for_packed_items(data: Data, solution: dict):
    p_vals = solution.get("p", [])
    used_rotations = []
    for idx, it in enumerate(data.items):
        if idx >= len(p_vals) or float(p_vals[idx]) <= 0.5:
            continue
        used_rotations.append(float(it.rotation) % 360.0)
    return used_rotations


def test_hybrid_width_crop_with_visible_rotations_visual():
    """
    Rotation demo (width-crop):
    - R=4 enables right-angle rotations;
    - greedy gives a weaker baseline;
    - model improves objective and uses at least one 90/270 rotation.
    """
    data = Data(_build_rotation_demo_rectangles(), R=ROT_DEMO_R, parallel_nfp=False)
    solver = HybridSolverWidth(
        data,
        height=ROT_DEMO_HEIGHT,
        width=ROT_DEMO_WIDTH,
        S=ROT_DEMO_S,
        greedy_delta_x=ROT_DEMO_GREEDY_DELTA_X,
    )

    result = solver.solve(
        unpack_last_n=4,
        crop_width=120.0,
        use_right_crop=True,
        free_space_improvement=0.0,
        solver_gap=0.5,
        model_time_limit_sec=12.0,
        stop_after_first_solution=False,
        model_enable_output=False,
        lock_greedy_unpacked=False,
        max_model_unfixed_items=30,
        random_iterations=1,
        random_seed=7,
        random_sample_size=6,
    )

    try:
        visualize_hybrid_result_width(
            data.items,
            result,
            width=ROT_DEMO_WIDTH,
            height=ROT_DEMO_HEIGHT,
            S=ROT_DEMO_S,
            show=True,
        )
    except Exception as e:
        print("visualization failed:", e)

    stats = result.get("hybrid_stats", {})
    greedy_obj = float(stats.get("greedy_objective_value") or 0.0)
    final_obj = float(stats.get("final_objective_value") or 0.0)
    model_status = str(stats.get("model_status"))
    used_rots = _used_rotations_for_packed_items(data, result.get("final", {}))
    has_right_angle_rotation = any(abs(r - 90.0) < 1e-6 or abs(r - 270.0) < 1e-6 for r in used_rots)

    print(
        {
            "status": result.get("status"),
            "selected_solution": result.get("selected_solution"),
            "model_status": model_status,
            "greedy_objective_value": greedy_obj,
            "final_objective_value": final_obj,
            "free_space_improvement_percent": stats.get("free_space_improvement_percent"),
            "used_rotations": used_rots,
        }
    )

    assert model_status in {"OPTIMAL", "FEASIBLE"}, f"Unexpected model status: {model_status}"
    assert result.get("selected_solution") == "model"
    assert final_obj > greedy_obj + 1.0, "Model must improve greedy objective in this rotation demo."
    assert has_right_angle_rotation, "Expected at least one packed item with 90/270 degree rotation."


def test_hybrid_width_crop_with_visible_rotations_plus4_small_delta_visual():
    """
    Small-delta behavior demo with +4 rectangles:
    this test is exploratory (not strictly 'must improve').
    """
    data = Data(_build_rotation_demo_rectangles_plus4(), R=ROT_DEMO_R, parallel_nfp=False)
    solver = HybridSolverWidth(
        data,
        height=ROT_DEMO_HEIGHT,
        width=ROT_DEMO_WIDTH,
        S=6,  # faster than S=12 for the larger (+4) scenario
        greedy_delta_x=5.0,
    )

    result = solver.solve(
        unpack_last_n=4,
        crop_width=120.0,
        use_right_crop=True,
        free_space_improvement=0.0,
        solver_gap=0.5,
        model_time_limit_sec=None,
        stop_after_first_solution=False,
        model_enable_output=True,
        lock_greedy_unpacked=False,
        max_model_unfixed_items=30,
        random_iterations=5,
        random_seed=51,
        random_sample_size=6,
    )

    try:
        visualize_hybrid_result_width(
            data.items,
            result,
            width=ROT_DEMO_WIDTH,
            height=ROT_DEMO_HEIGHT,
            S=6,
            show=True,
        )
    except Exception as e:
        print("visualization failed:", e)

    stats = result.get("hybrid_stats", {})
    greedy_obj = float(stats.get("greedy_objective_value") or 0.0)
    final_obj = float(stats.get("final_objective_value") or 0.0)
    model_status = str(stats.get("model_status"))
    used_rots = _used_rotations_for_packed_items(data, result.get("final", {}))
    has_right_angle_rotation = any(abs(r - 90.0) < 1e-6 or abs(r - 270.0) < 1e-6 for r in used_rots)

    print(
        {
            "status": result.get("status"),
            "selected_solution": result.get("selected_solution"),
            "model_status": model_status,
            "greedy_objective_value": greedy_obj,
            "final_objective_value": final_obj,
            "free_space_improvement_percent": stats.get("free_space_improvement_percent"),
            "used_rotations": used_rots,
        }
    )

    assert result.get("status") in {"OK", "NOT_PROVEN", "NOT_IMPROVED"}
    assert greedy_obj >= 0.0 and final_obj >= 0.0
    assert has_right_angle_rotation, "Expected packed solution to include at least one 90/270 rotation."
