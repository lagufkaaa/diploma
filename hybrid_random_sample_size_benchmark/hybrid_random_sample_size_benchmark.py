import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from core.data import Data
from solvers.hybrid_solver import HybridSolver as HybridSolverRandomSampling
from solvers.hybrid_solver_vertical_largest import HybridSolver as HybridSolverLargestSampling
from solvers.hybrid_solver_vertical_smallest import HybridSolver as HybridSolverSmallestSampling
from utils.helpers import util_model
from utils.hybrid_visualization import visualize_hybrid_result


DATA_DIR = ROOT_DIR / "data_car_mats"
OUTPUT_DIR = ROOT_DIR / "hybrid_random_sample_size_benchmark"
IMAGES_DIR = OUTPUT_DIR / "images"
TIMINGS_DIR = OUTPUT_DIR / "timings"


# ============================================================
# Benchmark configuration (style is close to test_hybrid_basic)
# Edit only this block for your experiments.
# ============================================================
DATA_FILE = DATA_DIR / "car_mats_2.txt"
R = 4
HEIGHT = 10000.0
WIDTH = 10000.0
S_VALUE = 5

# Loop by RANDOM_SAMPLE_SIZE: start, step, stop.
RANDOM_SAMPLE_SIZE_START = 5
RANDOM_SAMPLE_SIZE_STEP = 1
RANDOM_SAMPLE_SIZE_END = 15

# Hybrid solver params (same idea as in test_hybrid_basic).
UNPACK_LAST_N = 6
CROP_HEIGHT_RATIO = 1.0 / 3.0
FREE_SPACE_IMPROVEMENT = True
SOLVER_GAP = 1.0
MODEL_TIME_LIMIT_SEC = 7200
MODEL_NUM_THREADS = None
STOP_AFTER_FIRST_SOLUTION = False
MODEL_ENABLE_OUTPUT = True
RANDOM_ITERATIONS = 1
RANDOM_SEED = None
MIN_UNPACKED_IN_SAMPLE = UNPACK_LAST_N // 2
GREEDY_ENABLE_OUTPUT = True
HYBRID_ENABLE_OUTPUT = True

# Shared greedy-result cache options.
GREEDY_USE_RESULT_CACHE = True
GREEDY_RESULT_CACHE_PATH = None
GREEDY_RESULT_CACHE_TTL_DAYS = None
GREEDY_SHARED_RESULT_CACHE = {}

# Algorithm parameters (same as in algorithm benchmark).
# sampling strategy: random | smallest | largest
# greedy order strategy: deterministic | random
SAMPLING_STRATEGY = "random"
GREEDY_ORDER_STRATEGY = "deterministic"
GREEDY_RANDOM_SEED = None


SHARED_NFP_CACHE = {}


def normalize_sampling_strategy(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"random", "default"}:
        return "random"
    if v in {"smallest", "smallest_area", "small"}:
        return "smallest_area"
    if v in {"largest", "largest_area", "biggest"}:
        return "largest_area"
    raise ValueError(f"Unsupported sampling strategy: {value}")


def normalize_greedy_order_strategy(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"deterministic", "default", "area_desc", "sorted"}:
        return "deterministic"
    if v in {"random", "shuffle", "random_order"}:
        return "random"
    raise ValueError(f"Unsupported greedy order strategy: {value}")


def resolve_solver_class(sampling_strategy: str):
    if sampling_strategy == "smallest_area":
        return HybridSolverSmallestSampling
    if sampling_strategy == "largest_area":
        return HybridSolverLargestSampling
    return HybridSolverRandomSampling


def iter_random_sample_sizes(start: int, step: int, end: int):
    if step == 0:
        raise ValueError("RANDOM_SAMPLE_SIZE_STEP must not be 0")

    current = int(start)
    if step > 0:
        while current <= end:
            yield max(0, int(current))
            current += step
    else:
        while current >= end:
            yield max(0, int(current))
            current += step


def build_summary_text(results: list[dict]) -> str:
    lines = []
    lines.append("summary (model only, without greedy/data build)")

    for row in results:
        model_time_val = row.get("model_time_sec")
        model_time_text = (
            f"{float(model_time_val):.6f}" if model_time_val is not None else "None"
        )
        lines.append(
            f"RANDOM_SAMPLE_SIZE={row['random_sample_size']}"
            f" | model_time_sec={model_time_text}"
            f" | status={row['status']}"
            f" | model_status={row['model_status']}"
            f" | final_objective={row['final_objective']}"
            f" | random_seed_used={row['random_seed_used']}"
            f" | sampling_strategy={row.get('sampling_strategy')}"
            f" | greedy_order_strategy={row.get('greedy_order_strategy')}"
        )

    valid_rows = [r for r in results if r.get("model_time_sec") is not None]
    if valid_rows:
        times = [float(r["model_time_sec"]) for r in valid_rows]
        best_time_row = min(valid_rows, key=lambda r: float(r["model_time_sec"]))
        worst_time_row = max(valid_rows, key=lambda r: float(r["model_time_sec"]))
        lines.append("")
        lines.append("time_stats_model_only")
        lines.append(f"count: {len(times)}")
        lines.append(
            "min_model_time_sec: "
            f"{min(times):.6f} (RANDOM_SAMPLE_SIZE={best_time_row['random_sample_size']})"
        )
        lines.append(
            "max_model_time_sec: "
            f"{max(times):.6f} (RANDOM_SAMPLE_SIZE={worst_time_row['random_sample_size']})"
        )
        lines.append(f"avg_model_time_sec: {sum(times) / len(times):.6f}")

    lines.append("")
    return "\n".join(lines)


def build_detailed_sample_text(
    *,
    sample_size_value: int,
    sampling_strategy: str,
    greedy_order_strategy: str,
    greedy_random_seed_requested: Optional[int],
    elapsed: float,
    result: dict,
    image_file: Path,
    run_started: str,
    run_finished: str,
) -> str:
    hybrid_stats = result.get("hybrid_stats", {}) if isinstance(result, dict) else {}
    random_seed_requested = hybrid_stats.get("random_seed_requested", RANDOM_SEED)
    random_seed_used = hybrid_stats.get("random_seed_used", RANDOM_SEED)

    lines = []
    lines.append(f"detailed report for RANDOM_SAMPLE_SIZE={sample_size_value}")
    lines.append(f"run_started: {run_started}")
    lines.append(f"run_finished: {run_finished}")
    lines.append("")

    lines.append("input_data")
    lines.append(f"data_file: {DATA_FILE.resolve()}")
    lines.append(f"R: {R}")
    lines.append(f"height: {HEIGHT}")
    lines.append(f"width: {WIDTH}")
    lines.append(f"S: {S_VALUE}")
    lines.append("")

    lines.append("random_sample_size_value")
    lines.append(f"RANDOM_SAMPLE_SIZE: {sample_size_value}")
    lines.append("")

    lines.append("model_params")
    lines.append(f"unpack_last_n: {UNPACK_LAST_N}")
    lines.append(f"crop_height_ratio: {CROP_HEIGHT_RATIO}")
    lines.append(f"free_space_improvement: {FREE_SPACE_IMPROVEMENT}")
    lines.append(f"solver_gap: {SOLVER_GAP}")
    lines.append(f"model_time_limit_sec: {MODEL_TIME_LIMIT_SEC}")
    lines.append(f"model_num_threads: {MODEL_NUM_THREADS}")
    lines.append(f"stop_after_first_solution: {STOP_AFTER_FIRST_SOLUTION}")
    lines.append(f"model_enable_output: {MODEL_ENABLE_OUTPUT}")
    lines.append(f"random_iterations: {RANDOM_ITERATIONS}")
    lines.append(f"random_seed_requested: {random_seed_requested}")
    lines.append(f"random_seed_used: {random_seed_used}")
    lines.append(f"random_sample_size: {sample_size_value}")
    lines.append(f"sampling_strategy: {sampling_strategy}")
    lines.append(f"greedy_order_strategy: {greedy_order_strategy}")
    lines.append(f"greedy_random_seed_requested: {greedy_random_seed_requested}")
    lines.append(f"min_unpacked_in_sample: {MIN_UNPACKED_IN_SAMPLE}")
    lines.append(f"greedy_enable_output: {GREEDY_ENABLE_OUTPUT}")
    lines.append(f"hybrid_enable_output: {HYBRID_ENABLE_OUTPUT}")
    lines.append(f"greedy_use_result_cache: {GREEDY_USE_RESULT_CACHE}")
    lines.append(f"greedy_result_cache_path: {GREEDY_RESULT_CACHE_PATH}")
    lines.append(f"greedy_result_cache_ttl_days: {GREEDY_RESULT_CACHE_TTL_DAYS}")
    lines.append("")

    lines.append("run_result")
    lines.append(f"time_sec: {elapsed:.6f}")
    lines.append(f"status: {result.get('status')}")
    lines.append(f"selected_solution: {result.get('selected_solution')}")
    lines.append(f"final_objective: {hybrid_stats.get('final_objective_value')}")
    lines.append(f"model_status: {hybrid_stats.get('model_status')}")
    lines.append(f"greedy_cache_hit: {hybrid_stats.get('greedy_cache_hit')}")
    lines.append(f"image_file: {image_file.resolve()}")
    lines.append("")

    lines.append("hybrid_stats_json")
    lines.append(json.dumps(hybrid_stats, ensure_ascii=False, indent=2, default=str))
    lines.append("")

    lines.append("full_result_json")
    lines.append(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
    sampling_strategy = normalize_sampling_strategy(SAMPLING_STRATEGY)
    greedy_order_strategy = normalize_greedy_order_strategy(GREEDY_ORDER_STRATEGY)
    solver_cls = resolve_solver_class(sampling_strategy)

    items = util_model.parse_items(str(DATA_FILE))
    if not items:
        raise RuntimeError(f"No items parsed from file: {DATA_FILE}")

    print(f"Data file: {DATA_FILE.resolve()}")
    print(
        "RANDOM_SAMPLE_SIZE loop: "
        f"start={RANDOM_SAMPLE_SIZE_START}, step={RANDOM_SAMPLE_SIZE_STEP}, end={RANDOM_SAMPLE_SIZE_END}"
    )
    print(
        "Algorithm params: "
        f"sampling_strategy={sampling_strategy}, greedy_order_strategy={greedy_order_strategy}"
    )
    print("Building Data once (as in test_hybrid_basic)...")

    data = Data(
        items,
        R,
        parallel_nfp=False,
        shared_memory_cache=SHARED_NFP_CACHE,
    )

    all_results = []

    for sample_size_value in iter_random_sample_sizes(
        RANDOM_SAMPLE_SIZE_START,
        RANDOM_SAMPLE_SIZE_STEP,
        RANDOM_SAMPLE_SIZE_END,
    ):
        print(f"\n=== Running hybrid for RANDOM_SAMPLE_SIZE={sample_size_value} ===")
        run_started = datetime.now().isoformat(timespec="seconds")

        greedy_seed_requested = (
            None
            if greedy_order_strategy != "random"
            else (int(RANDOM_SEED) if RANDOM_SEED is not None else GREEDY_RANDOM_SEED)
        )

        solver = solver_cls(
            data,
            height=HEIGHT,
            width=WIDTH,
            S=S_VALUE,
            greedy_enable_output=GREEDY_ENABLE_OUTPUT,
            hybrid_enable_output=HYBRID_ENABLE_OUTPUT,
            greedy_use_result_cache=GREEDY_USE_RESULT_CACHE,
            greedy_result_cache_path=GREEDY_RESULT_CACHE_PATH,
            greedy_result_cache_ttl_days=GREEDY_RESULT_CACHE_TTL_DAYS,
            greedy_shared_result_cache=GREEDY_SHARED_RESULT_CACHE,
            greedy_order_strategy=greedy_order_strategy,
            greedy_random_seed=greedy_seed_requested,
        )

        solve_start = time.perf_counter()
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
            random_seed=RANDOM_SEED,
            random_sample_size=sample_size_value,
            greedy_order_strategy=greedy_order_strategy,
            greedy_random_seed=greedy_seed_requested,
        )
        elapsed = time.perf_counter() - solve_start
        run_finished = datetime.now().isoformat(timespec="seconds")

        image_file = IMAGES_DIR / f"hybrid_result_sample_{sample_size_value}.png"
        try:
            visualize_hybrid_result(
                data.items,
                result,
                width=WIDTH,
                height=HEIGHT,
                S=S_VALUE,
                show=False,
                save_path=str(image_file),
            )
            print(f"Saved image: {image_file}")
        except Exception as exc:
            print(
                "Visualization failed for "
                f"RANDOM_SAMPLE_SIZE={sample_size_value}: {exc}"
            )

        result_safe = result if isinstance(result, dict) else {"status": "UNKNOWN"}
        detailed_text = build_detailed_sample_text(
            sample_size_value=int(sample_size_value),
            sampling_strategy=sampling_strategy,
            greedy_order_strategy=greedy_order_strategy,
            greedy_random_seed_requested=greedy_seed_requested,
            elapsed=float(elapsed),
            result=result_safe,
            image_file=image_file,
            run_started=run_started,
            run_finished=run_finished,
        )
        detailed_path = TIMINGS_DIR / f"sample_{int(sample_size_value)}.txt"
        detailed_path.write_text(detailed_text, encoding="utf-8")
        print(f"Saved details: {detailed_path}")

        hybrid_stats = result_safe.get("hybrid_stats", {})
        row = {
            "random_sample_size": int(sample_size_value),
            "full_time_sec": float(elapsed),
            "model_time_sec": hybrid_stats.get("model_time_sec"),
            "status": str(result_safe.get("status", "UNKNOWN")),
            "final_objective": hybrid_stats.get("final_objective_value"),
            "model_status": hybrid_stats.get("model_status"),
            "greedy_cache_hit": hybrid_stats.get("greedy_cache_hit"),
            "random_seed_used": hybrid_stats.get("random_seed_used"),
            "min_unpacked_in_sample": hybrid_stats.get("min_unpacked_in_sample"),
            "sampling_strategy": hybrid_stats.get("sampling_strategy"),
            "greedy_order_strategy": hybrid_stats.get("greedy_order_strategy"),
            "greedy_random_seed_used": hybrid_stats.get("greedy_random_seed_used"),
        }
        all_results.append(row)

        model_time_val = row.get("model_time_sec")
        model_time_text = (
            f"{float(model_time_val):.6f}" if model_time_val is not None else "None"
        )
        print(
            "random_sample_size - время модели (без greedy): "
            f"{row['random_sample_size']} - {model_time_text} sec"
            f" | status={row['status']}"
        )

        summary_text = build_summary_text(all_results)
        summary_path = TIMINGS_DIR / "summary.txt"
        summary_path.write_text(summary_text, encoding="utf-8")
        print(f"Saved summary (intermediate): {summary_path}")

    summary_text = build_summary_text(all_results)

    summary_path = TIMINGS_DIR / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
