import sys
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from core.data import Data
from solvers.hybrid_solver import HybridSolver
from utils.hybrid_visualization import visualize_hybrid_result


ROT_DEMO_WIDTH = 200.0
ROT_DEMO_HEIGHT = 120.0
ROT_DEMO_S = 12
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
        (72, 30),
        (56, 30),
        (60, 35),
        (40, 55),
        (15, 15),
        (15, 15),
        (15, 15),
        (15, 15),
    ]
    return [
        np.array([[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]], dtype=float)
        for w, h in dims
    ]


def _build_rotation_demo_rectangles_plus4():
    """
    Extended dataset (+4 rectangles) for behavior inspection.
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
        (15, 15),
        (15, 15),
        (15, 15),
        (15, 15),
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


def test_hybrid_top_crop_with_visible_rotations_visual():
    """
    Rotation demo (top-crop):
    - R=4 enables right-angle rotations;
    - greedy and model are both visualized;
    - we validate that top-crop flow runs and returns a valid solution.
    """
    data = Data(_build_rotation_demo_rectangles_plus4(), R=ROT_DEMO_R, parallel_nfp=False)
    solver = HybridSolver(
        data,
        height=ROT_DEMO_HEIGHT,
        width=ROT_DEMO_WIDTH,
        S=ROT_DEMO_S,
    )

    result = solver.solve(
        unpack_last_n=4,
        crop_height=80.0,
        use_top_crop=True,
        free_space_improvement=False,
        solver_gap=0.5,
        model_time_limit_sec=10.0,
        stop_after_first_solution=False,
        model_enable_output=False,
        min_unpacked_in_sample=2,
        random_iterations=1,
        random_seed=41,
        random_sample_size=6,
    )

    try:
        visualize_hybrid_result(
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

    assert result.get("status") in {"OK", "NOT_PROVEN", "NOT_IMPROVED"}
    assert model_status in {"OPTIMAL", "FEASIBLE", "NOT_SOLVED", "INFEASIBLE"}
    assert result.get("selected_solution") in {"model", "greedy"}
    assert np.isclose(float(stats.get("used_crop_height", 0.0)), 80.0)
    assert bool(stats.get("use_top_crop")) is True
    assert int(stats.get("min_unpacked_in_sample", -1)) == 2
    assert greedy_obj >= 0.0 and final_obj >= 0.0
    assert len(used_rots) > 0
    assert all(abs((r % 90.0)) < 1e-6 for r in used_rots)
    if has_right_angle_rotation:
        print("right-angle rotation is present in packed solution")


def test_hybrid_top_crop_with_visible_rotations_plus4_small_delta_visual():
    """
    Behavior demo with +4 rectangles:
    this test is exploratory (not strictly 'must improve').
    """
    data = Data(_build_rotation_demo_rectangles_plus4(), R=ROT_DEMO_R, parallel_nfp=False)
    solver = HybridSolver(
        data,
        height=ROT_DEMO_HEIGHT,
        width=ROT_DEMO_WIDTH,
        S=6,  # faster than S=12 for the larger (+4) scenario
    )

    result = solver.solve(
        unpack_last_n=4,
        crop_height=80.0,
        use_top_crop=True,
        free_space_improvement=True,
        solver_gap=0.0,
        model_time_limit_sec=15.0,
        stop_after_first_solution=False,
        model_enable_output=False,
        min_unpacked_in_sample=2,
        random_iterations=2,
        random_seed=231,
        random_sample_size=12,
    )

    try:
        visualize_hybrid_result(
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
    assert model_status in {"OPTIMAL", "FEASIBLE", "NOT_SOLVED", "INFEASIBLE"}
    assert result.get("selected_solution") in {"model", "greedy"}
    assert np.isclose(float(stats.get("used_crop_height", 0.0)), 80.0)
    assert bool(stats.get("use_top_crop")) is True
    assert int(stats.get("min_unpacked_in_sample", -1)) == 2
    assert greedy_obj >= 0.0 and final_obj >= 0.0
    assert len(used_rots) > 0
    assert all(abs((r % 90.0)) < 1e-6 for r in used_rots)
    if has_right_angle_rotation:
        print("right-angle rotation is present in packed solution")
