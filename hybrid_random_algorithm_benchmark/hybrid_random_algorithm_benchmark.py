import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from core.data import Data
from solvers.hybrid_solver import HybridSolver
from utils.helpers import util_model
from utils.hybrid_visualization import visualize_hybrid_result


DATA_DIR = ROOT_DIR / "data_car_mats"
OUTPUT_DIR = ROOT_DIR / "hybrid_random_algorithm_benchmark"
RESULTS_DIR = OUTPUT_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
IMAGES_DIR = RESULTS_DIR / "images"


# ============================================================
# Benchmark configuration (random hybrid solver)
# ============================================================
DATA_FILE = DATA_DIR / "car_mats_2.txt"
R = 4
S = 5
HEIGHT = 10000.0
WIDTH = 10000.0
SOLVER_NAME = "SCIP"

NUM_RUNS = 20
BASE_RANDOM_SEED: Optional[int] = 1000

UNPACK_LAST_N = 6
CROP_HEIGHT_RATIO = 1.0 / 3.0
FREE_SPACE_IMPROVEMENT = True
SOLVER_GAP = 1.0
MODEL_TIME_LIMIT_SEC: Optional[float] = 120.0
MODEL_NUM_THREADS: Optional[int] = None
STOP_AFTER_FIRST_SOLUTION = False
MODEL_ENABLE_OUTPUT = False

RANDOM_ITERATIONS = 10
RANDOM_SAMPLE_SIZE = 10
MIN_UNPACKED_IN_SAMPLE = 2

GREEDY_ENABLE_OUTPUT = False
HYBRID_ENABLE_OUTPUT = False
SAVE_IMAGES = True

GREEDY_USE_RESULT_CACHE = True
GREEDY_RESULT_CACHE_PATH = None
GREEDY_RESULT_CACHE_TTL_DAYS = None
GREEDY_SHARED_RESULT_CACHE = {}

IMPROVEMENT_EPS = 1e-9


SHARED_NFP_CACHE = {}


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(mean(values))


def safe_pstdev(values: List[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(pstdev(values))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary_text(run_rows: List[Dict[str, Any]], iteration_rows: List[Dict[str, Any]]) -> str:
    greedy_values = [float(r["greedy_objective"]) for r in run_rows if r.get("greedy_objective") is not None]
    final_values = [float(r["final_objective"]) for r in run_rows if r.get("final_objective") is not None]
    deltas = [float(r["final_minus_greedy"]) for r in run_rows if r.get("final_minus_greedy") is not None]
    improved_runs = [r for r in run_rows if bool(r.get("improved_over_greedy"))]

    iter_deltas = [float(r["iteration_minus_greedy"]) for r in iteration_rows if r.get("iteration_minus_greedy") is not None]
    iter_improved = [r for r in iteration_rows if bool(r.get("iteration_improved_vs_greedy"))]
    iter_feasible = [
        r
        for r in iteration_rows
        if r.get("iteration_status") in {"OPTIMAL", "FEASIBLE"} and r.get("iteration_objective") is not None
    ]

    final_std = safe_pstdev(final_values)
    final_mean = safe_mean(final_values)
    cv_percent = None
    if final_std is not None and final_mean not in (None, 0.0):
        cv_percent = 100.0 * float(final_std) / float(final_mean)

    lines: List[str] = []
    lines.append("random hybrid benchmark summary")
    lines.append(f"created_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("quality_vs_greedy")
    lines.append(f"runs_total: {len(run_rows)}")
    lines.append(f"runs_with_final_objective: {len(final_values)}")
    lines.append(f"avg_greedy_objective: {safe_mean(greedy_values)}")
    lines.append(f"avg_final_objective: {safe_mean(final_values)}")
    lines.append(f"avg_final_minus_greedy: {safe_mean(deltas)}")
    lines.append(f"min_final_minus_greedy: {min(deltas) if deltas else None}")
    lines.append(f"max_final_minus_greedy: {max(deltas) if deltas else None}")
    lines.append("")
    lines.append("improvement_frequency")
    lines.append(f"improved_runs_count: {len(improved_runs)}")
    lines.append(
        f"improved_runs_rate_percent: {100.0 * len(improved_runs) / len(run_rows) if run_rows else 0.0:.2f}"
    )
    lines.append("")
    lines.append("robustness_to_randomness")
    lines.append(f"final_objective_stddev: {final_std}")
    lines.append(f"final_objective_cv_percent: {cv_percent}")
    lines.append(f"final_objective_min: {min(final_values) if final_values else None}")
    lines.append(f"final_objective_max: {max(final_values) if final_values else None}")
    lines.append(
        "final_objective_range: "
        f"{(max(final_values) - min(final_values)) if len(final_values) >= 2 else 0.0 if final_values else None}"
    )
    lines.append("")
    lines.append("iteration_level")
    lines.append(f"iterations_total_logged: {len(iteration_rows)}")
    lines.append(f"iterations_feasible_count: {len(iter_feasible)}")
    lines.append(
        f"iterations_feasible_rate_percent: {100.0 * len(iter_feasible) / len(iteration_rows) if iteration_rows else 0.0:.2f}"
    )
    lines.append(f"iterations_improved_vs_greedy_count: {len(iter_improved)}")
    lines.append(
        "iterations_improved_vs_greedy_rate_percent: "
        f"{100.0 * len(iter_improved) / len(iteration_rows) if iteration_rows else 0.0:.2f}"
    )
    lines.append(f"avg_iteration_minus_greedy: {safe_mean(iter_deltas)}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_IMAGES:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_file": str(DATA_FILE.resolve()),
        "R": int(R),
        "S": int(S),
        "height": float(HEIGHT),
        "width": float(WIDTH),
        "solver_name": SOLVER_NAME,
        "num_runs": int(NUM_RUNS),
        "base_random_seed": BASE_RANDOM_SEED,
        "unpack_last_n": int(UNPACK_LAST_N),
        "crop_height_ratio": float(CROP_HEIGHT_RATIO),
        "free_space_improvement": bool(FREE_SPACE_IMPROVEMENT),
        "solver_gap": float(SOLVER_GAP),
        "model_time_limit_sec": MODEL_TIME_LIMIT_SEC,
        "model_num_threads": MODEL_NUM_THREADS,
        "stop_after_first_solution": bool(STOP_AFTER_FIRST_SOLUTION),
        "model_enable_output": bool(MODEL_ENABLE_OUTPUT),
        "random_iterations": int(RANDOM_ITERATIONS),
        "random_sample_size": int(RANDOM_SAMPLE_SIZE),
        "min_unpacked_in_sample": int(MIN_UNPACKED_IN_SAMPLE),
        "greedy_enable_output": bool(GREEDY_ENABLE_OUTPUT),
        "hybrid_enable_output": bool(HYBRID_ENABLE_OUTPUT),
        "save_images": bool(SAVE_IMAGES),
        "greedy_use_result_cache": bool(GREEDY_USE_RESULT_CACHE),
        "greedy_result_cache_path": GREEDY_RESULT_CACHE_PATH,
        "greedy_result_cache_ttl_days": GREEDY_RESULT_CACHE_TTL_DAYS,
    }
    (RESULTS_DIR / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    items = util_model.parse_items(str(DATA_FILE))
    if not items:
        raise RuntimeError(f"No items parsed from file: {DATA_FILE}")

    print(f"Data file: {DATA_FILE.resolve()}")
    print(f"Runs: {NUM_RUNS}, random_iterations per run: {RANDOM_ITERATIONS}")

    data = Data(
        items,
        R,
        parallel_nfp=False,
        shared_memory_cache=SHARED_NFP_CACHE,
    )

    run_rows: List[Dict[str, Any]] = []
    iteration_rows: List[Dict[str, Any]] = []

    for run_idx in range(1, int(NUM_RUNS) + 1):
        run_seed = None if BASE_RANDOM_SEED is None else int(BASE_RANDOM_SEED) + (run_idx - 1)
        run_started = datetime.now().isoformat(timespec="seconds")

        solver = HybridSolver(
            data=data,
            height=HEIGHT,
            width=WIDTH,
            S=S,
            solver_name=SOLVER_NAME,
            greedy_enable_output=GREEDY_ENABLE_OUTPUT,
            greedy_use_result_cache=GREEDY_USE_RESULT_CACHE,
            greedy_result_cache_path=GREEDY_RESULT_CACHE_PATH,
            greedy_result_cache_ttl_days=GREEDY_RESULT_CACHE_TTL_DAYS,
            greedy_shared_result_cache=GREEDY_SHARED_RESULT_CACHE,
            hybrid_enable_output=HYBRID_ENABLE_OUTPUT,
        )

        t0 = time.perf_counter()
        result = solver.solve(
            unpack_last_n=UNPACK_LAST_N,
            crop_height=CROP_HEIGHT_RATIO * HEIGHT,
            use_top_crop=True,
            free_space_improvement=FREE_SPACE_IMPROVEMENT,
            solver_gap=SOLVER_GAP,
            model_time_limit_sec=MODEL_TIME_LIMIT_SEC,
            model_num_threads=MODEL_NUM_THREADS,
            stop_after_first_solution=STOP_AFTER_FIRST_SOLUTION,
            model_enable_output=MODEL_ENABLE_OUTPUT,
            min_unpacked_in_sample=MIN_UNPACKED_IN_SAMPLE,
            random_iterations=RANDOM_ITERATIONS,
            random_seed=run_seed,
            random_sample_size=RANDOM_SAMPLE_SIZE,
        )
        elapsed = time.perf_counter() - t0
        run_finished = datetime.now().isoformat(timespec="seconds")

        image_file: Optional[Path] = None
        if SAVE_IMAGES:
            image_file = IMAGES_DIR / f"run_{run_idx:04d}.png"
            try:
                visualize_hybrid_result(
                    data.items,
                    result,
                    width=WIDTH,
                    height=HEIGHT,
                    S=S,
                    show=False,
                    save_path=str(image_file),
                )
            except Exception:
                image_file = None

        stats = result.get("hybrid_stats", {}) if isinstance(result, dict) else {}
        greedy_obj = to_float(stats.get("greedy_objective_value"))
        model_obj = to_float(stats.get("model_objective_value"))
        final_obj = to_float(stats.get("final_objective_value"))
        delta = (final_obj - greedy_obj) if final_obj is not None and greedy_obj is not None else None
        improved = bool(delta is not None and delta > IMPROVEMENT_EPS)

        run_row = {
            "run_idx": int(run_idx),
            "run_started": run_started,
            "run_finished": run_finished,
            "elapsed_wall_sec": float(elapsed),
            "run_seed_requested": run_seed,
            "run_seed_used": stats.get("random_seed_used"),
            "status": result.get("status") if isinstance(result, dict) else "UNKNOWN",
            "selected_solution": result.get("selected_solution") if isinstance(result, dict) else None,
            "model_status": stats.get("model_status"),
            "greedy_objective": greedy_obj,
            "model_objective": model_obj,
            "final_objective": final_obj,
            "final_minus_greedy": delta,
            "improved_over_greedy": improved,
            "greedy_time_sec": to_float(stats.get("greedy_time_sec")),
            "model_time_sec": to_float(stats.get("model_time_sec")),
            "total_time_sec": to_float(stats.get("total_time_sec")),
            "random_iterations_requested": stats.get("random_iterations_requested"),
            "random_iterations_executed": stats.get("random_iterations_executed"),
            "best_model_iteration": stats.get("best_model_iteration"),
            "random_sample_size_requested": stats.get("random_sample_size_requested"),
            "random_sample_size_effective": stats.get("random_sample_size"),
            "min_unpacked_in_sample": stats.get("min_unpacked_in_sample"),
            "unpack_last_n": stats.get("unpack_last_n"),
            "unpack_ids_count": stats.get("unpack_ids_count"),
            "sampled_model_item_ids_count": stats.get("sampled_model_item_ids_count"),
            "image_file": str(image_file.resolve()) if image_file is not None else None,
        }
        run_rows.append(run_row)

        iter_stats = stats.get("iteration_stats")
        if not isinstance(iter_stats, list):
            iter_stats = []
        for rec in iter_stats:
            iter_obj = to_float(rec.get("objective_value"))
            iter_delta = (iter_obj - greedy_obj) if iter_obj is not None and greedy_obj is not None else None
            iter_improved = bool(iter_delta is not None and iter_delta > IMPROVEMENT_EPS)
            iteration_rows.append(
                {
                    "run_idx": int(run_idx),
                    "run_seed_requested": run_seed,
                    "run_seed_used": stats.get("random_seed_used"),
                    "iteration": rec.get("iteration"),
                    "selected_as_best": rec.get("selected_as_best"),
                    "iteration_status": rec.get("status"),
                    "iteration_objective": iter_obj,
                    "iteration_minus_greedy": iter_delta,
                    "iteration_improved_vs_greedy": iter_improved,
                    "iter_time_sec": to_float(rec.get("iter_time_sec")),
                    "model_item_ids_count": rec.get("model_item_ids_count"),
                    "sampled_model_item_ids_count": rec.get("sampled_model_item_ids_count"),
                    "fixed_records_count": rec.get("fixed_records_count"),
                    "fixed_blockers_count": rec.get("fixed_blockers_count"),
                    "local_min_objective": to_float(rec.get("local_min_objective")),
                    "local_max_objective": to_float(rec.get("local_max_objective")),
                    "model_data_empty": rec.get("model_data_empty"),
                }
            )

        run_json = {
            "run_idx": int(run_idx),
            "run_seed_requested": run_seed,
            "run_seed_used": stats.get("random_seed_used"),
            "run_started": run_started,
            "run_finished": run_finished,
            "elapsed_wall_sec": float(elapsed),
            "image_file": str(image_file.resolve()) if image_file is not None else None,
            "result": result,
        }
        (RUNS_DIR / f"run_{run_idx:04d}.json").write_text(
            json.dumps(run_json, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        print(
            f"run {run_idx}/{NUM_RUNS}: status={run_row['status']}, "
            f"model_status={run_row['model_status']}, improved={run_row['improved_over_greedy']}, "
            f"final_minus_greedy={run_row['final_minus_greedy']}"
        )

        # Save intermediate artifacts after each run, so long executions are resumable/observable.
        run_fieldnames = [
            "run_idx",
            "run_started",
            "run_finished",
            "elapsed_wall_sec",
            "run_seed_requested",
            "run_seed_used",
            "status",
            "selected_solution",
            "model_status",
            "greedy_objective",
            "model_objective",
            "final_objective",
            "final_minus_greedy",
            "improved_over_greedy",
            "greedy_time_sec",
            "model_time_sec",
            "total_time_sec",
            "random_iterations_requested",
            "random_iterations_executed",
            "best_model_iteration",
            "random_sample_size_requested",
            "random_sample_size_effective",
            "min_unpacked_in_sample",
            "unpack_last_n",
            "unpack_ids_count",
            "sampled_model_item_ids_count",
            "image_file",
        ]
        iteration_fieldnames = [
            "run_idx",
            "run_seed_requested",
            "run_seed_used",
            "iteration",
            "selected_as_best",
            "iteration_status",
            "iteration_objective",
            "iteration_minus_greedy",
            "iteration_improved_vs_greedy",
            "iter_time_sec",
            "model_item_ids_count",
            "sampled_model_item_ids_count",
            "fixed_records_count",
            "fixed_blockers_count",
            "local_min_objective",
            "local_max_objective",
            "model_data_empty",
        ]
        write_csv(RESULTS_DIR / "run_metrics.csv", run_rows, run_fieldnames)
        write_csv(RESULTS_DIR / "iteration_metrics.csv", iteration_rows, iteration_fieldnames)
        (RESULTS_DIR / "summary.txt").write_text(build_summary_text(run_rows, iteration_rows), encoding="utf-8")
        print(f"Saved summary (intermediate): {RESULTS_DIR / 'summary.txt'}")
    run_fieldnames = [
        "run_idx",
        "run_started",
        "run_finished",
        "elapsed_wall_sec",
        "run_seed_requested",
        "run_seed_used",
        "status",
        "selected_solution",
        "model_status",
        "greedy_objective",
        "model_objective",
        "final_objective",
        "final_minus_greedy",
        "improved_over_greedy",
        "greedy_time_sec",
        "model_time_sec",
        "total_time_sec",
        "random_iterations_requested",
        "random_iterations_executed",
        "best_model_iteration",
        "random_sample_size_requested",
        "random_sample_size_effective",
        "min_unpacked_in_sample",
        "unpack_last_n",
        "unpack_ids_count",
        "sampled_model_item_ids_count",
        "image_file",
    ]
    iteration_fieldnames = [
        "run_idx",
        "run_seed_requested",
        "run_seed_used",
        "iteration",
        "selected_as_best",
        "iteration_status",
        "iteration_objective",
        "iteration_minus_greedy",
        "iteration_improved_vs_greedy",
        "iter_time_sec",
        "model_item_ids_count",
        "sampled_model_item_ids_count",
        "fixed_records_count",
        "fixed_blockers_count",
        "local_min_objective",
        "local_max_objective",
        "model_data_empty",
    ]

    write_csv(RESULTS_DIR / "run_metrics.csv", run_rows, run_fieldnames)
    write_csv(RESULTS_DIR / "iteration_metrics.csv", iteration_rows, iteration_fieldnames)
    (RESULTS_DIR / "summary.txt").write_text(build_summary_text(run_rows, iteration_rows), encoding="utf-8")

    print(f"\nSaved: {RESULTS_DIR / 'run_metrics.csv'}")
    print(f"Saved: {RESULTS_DIR / 'iteration_metrics.csv'}")
    print(f"Saved: {RESULTS_DIR / 'summary.txt'}")
    print(f"Saved run JSON files: {RUNS_DIR}")


if __name__ == "__main__":
    main()
