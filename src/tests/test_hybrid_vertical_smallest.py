import sys
import time
from pathlib import Path

import pytest


SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from core.data import Data
from solvers.hybrid_solver_vertical_smallest import HybridSolver
from utils.helpers import util_model
from utils.hybrid_visualization import visualize_hybrid_result


DATA_DIR = Path(__file__).resolve().parents[2] / "data_car_mats"
SHARED_NFP_CACHE = {}


def test_hybrid_vertical_smallest_basic():
    file_path = DATA_DIR / "car_mats_2.txt"
    items = util_model.parse_items(str(file_path))
    assert len(items) > 0, "no items parsed from test file"

    R = 4
    S = 15
    height = 10000.0
    width = 10000.0

    total_start = time.time()

    print(f"hybrid-vertical-smallest input file: {file_path}")
    print("starting data")

    data_start = time.time()
    data = Data(items, R, parallel_nfp=False, shared_memory_cache=SHARED_NFP_CACHE)
    data_time = time.time() - data_start

    print("starting hybrid-vertical-smallest")

    solver_start = time.time()
    solver = HybridSolver(data, height=height, width=width, S=S)
    solver_time = time.time() - solver_start

    random_iterations = 1
    random_seed = None
    random_sample_size = 9
    unpack_last_n = 5
    crop_height = height * 1.0 / 3.0
    
    solve_start = time.time()
    result = solver.solve(
        unpack_last_n=unpack_last_n,
        crop_height=crop_height,
        use_top_crop=True,
        free_space_improvement=True,
        solver_gap=0.3,
        model_time_limit_sec=None,
        stop_after_first_solution=False,
        model_enable_output=True,
        lock_greedy_unpacked=False,
        max_model_unfixed_items=None,
        random_iterations=random_iterations,
        random_seed=random_seed,
        random_sample_size=random_sample_size,
    )
    solve_time = time.time() - solve_start

    total_time = time.time() - total_start

    print("--- hybrid-vertical-smallest solve results ---")
    print(
        {
            "status": result.get("status"),
            "selected_solution": result.get("selected_solution"),
            "hybrid_stats": result.get("hybrid_stats"),
        }
    )
    print(f"Data creation time: {data_time:.4f} seconds")
    print(f"Solver creation time: {solver_time:.4f} seconds")
    print(f"Solve() call time: {solve_time:.4f} seconds")
    print(f"Total (Data + Solver + Solve): {total_time:.4f} seconds")

    status = result.get("status")
    assert status in {"OK", "NOT_IMPROVED", "NOT_PROVEN"}, (
        f"Unexpected hybrid-vertical-smallest status: {status}"
    )

    stats = result.get("hybrid_stats", {})
    print(f"greedy objective: {stats.get('greedy_objective_value')}")
    print(f"model objective: {stats.get('model_objective_value')}")
    print(f"model status: {stats.get('model_status')}")
    print(f"random_iterations_requested: {stats.get('random_iterations_requested')}")
    print(f"random_iterations_executed: {stats.get('random_iterations_executed')}")
    print(f"random_sample_size_requested: {stats.get('random_sample_size_requested')}")
    print(f"random_sample_size: {stats.get('random_sample_size')}")
    print(f"sampling_strategy: {stats.get('sampling_strategy')}")
    print(f"best_model_iteration: {stats.get('best_model_iteration')}")
    print(f"use_top_crop: {stats.get('use_top_crop')}")
    print(f"used_crop_height: {stats.get('used_crop_height')}")
    print(f"random_seed_requested: {stats.get('random_seed_requested')}")
    print(f"random_seed_used: {stats.get('random_seed_used')}")

    assert stats.get("random_iterations_requested") == random_iterations
    assert stats.get("random_iterations_executed") == random_iterations
    assert int(stats.get("random_sample_size_requested", -1)) == max(0, int(random_sample_size))
    model_pool_count = int(stats.get("model_pool_ids_count", 0))
    expected_sample_size = min(model_pool_count, max(0, int(random_sample_size)))
    assert int(stats.get("random_sample_size", -1)) == expected_sample_size
    assert int(stats.get("sampled_model_item_ids_count", -1)) == expected_sample_size
    assert stats.get("sampling_strategy") == "smallest_area"
    assert stats.get("random_seed_requested") == random_seed
    assert stats.get("random_seed_used") is None
    assert stats.get("use_top_crop") is True
    assert float(stats.get("used_crop_height", 0.0)) == pytest.approx(crop_height)

    try:
        visualize_hybrid_result(
            data.items,
            result,
            width=width,
            height=height,
            S=S,
            show=True,
        )
    except Exception as exc:
        print("visualization failed:", exc)
