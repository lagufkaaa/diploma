import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


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
OUTPUT_DIR = ROOT_DIR / "hybrid_algorithms_benchmark"


# ============================================================
# Benchmark configuration (algorithm combinations)
# ============================================================
DATA_FILE = DATA_DIR / "car_mats_2.txt"
R = 4
S = 10
HEIGHT = 6000.0
WIDTH = 6000.0
SOLVER_NAME = "SCIP"

NUM_RUNS = 20
BASE_RANDOM_SEED: Optional[int] = None

UNPACK_LAST_N = 3
CROP_HEIGHT_RATIO = 1.0 / 3.0
# Crop selection for top-crop:
# - fixed_height: always use crop_height
# - unpacked_lowest_y: use adaptive logic based on lowest unpacked point
CROP_SELECTION_MODE = "unpacked_lowest_y"
CROP_ZERO_TOLERANCE = 1e-6
CROP_LOWEST_MULTIPLIER = 1.5
# True requires any positive improvement; a positive number means percentage points
# of required free-space improvement; anything else disables the constraint.
FREE_SPACE_IMPROVEMENT = False
EARLY_STOP_FREE_SPACE_IMPROVEMENT = 3
SOLVER_GAP = 0.5
MODEL_TIME_LIMIT_SEC: Optional[float] = 3600.0
MODEL_NUM_THREADS: Optional[int] = None
STOP_AFTER_FIRST_SOLUTION = False
MODEL_ENABLE_OUTPUT = True

RANDOM_ITERATIONS = 5
RANDOM_SAMPLE_SIZE = 7
MIN_UNPACKED_IN_SAMPLE = 0

GREEDY_ENABLE_OUTPUT = True
HYBRID_ENABLE_OUTPUT = True
SAVE_IMAGES = True

GREEDY_USE_RESULT_CACHE = True
GREEDY_RESULT_CACHE_PATH = None
GREEDY_RESULT_CACHE_TTL_DAYS = None
GREEDY_SHARED_RESULT_CACHE = {}

# Selected algorithm parameters:
# - greedy order strategy: deterministic | random
# - model sampling strategy: random | smallest | largest
GREEDY_ORDER_STRATEGY = "random"
SAMPLING_STRATEGY = "random"

# Run-output settings.
# For parallel runs set different RUN_TAG values manually OR keep AUTO_RUN_TAG=True.
RUN_TAG: Optional[str] = None
AUTO_RUN_TAG = True

# Seed behavior:
# If True and BASE_RANDOM_SEED is None, each run gets a unique random seed.
FORCE_UNIQUE_RANDOM_SEEDS_PER_RUN = True

IMPROVEMENT_EPS = 1e-9


SHARED_NFP_CACHE = {}


def draw_unique_seed(used: set[int], rng: random.SystemRandom) -> int:
    while True:
        candidate = int(rng.randrange(0, 2**63))
        if candidate in used:
            continue
        used.add(candidate)
        return candidate


def sanitize_run_tag(raw_tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw_tag).strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        return "run"
    return cleaned


def resolve_effective_run_tag() -> str:
    if RUN_TAG is not None:
        return sanitize_run_tag(RUN_TAG)
    if AUTO_RUN_TAG:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return "default"


EFFECTIVE_RUN_TAG = resolve_effective_run_tag()
RESULTS_DIR = OUTPUT_DIR / "results" / EFFECTIVE_RUN_TAG
RUNS_DIR = RESULTS_DIR / "runs"
IMAGES_DIR = RESULTS_DIR / "images"


RUN_FIELDNAMES = [
    "algorithm_id",
    "run_idx_within_algorithm",
    "global_run_idx",
    "run_started",
    "run_finished",
    "elapsed_wall_sec",
    "run_seed_requested",
    "run_seed_used",
    "status",
    "selected_solution",
    "model_status",
    "greedy_order_strategy_requested",
    "greedy_order_strategy_effective",
    "greedy_random_seed_requested",
    "greedy_random_seed_used",
    "sampling_strategy_requested",
    "sampling_strategy_effective",
    "crop_selection_mode_requested",
    "crop_selection_mode_effective",
    "crop_decision",
    "crop_zero_tolerance",
    "crop_lowest_multiplier",
    "lowest_unpacked_y_for_crop",
    "scaled_lowest_distance_for_crop",
    "used_crop_height",
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

ITERATION_FIELDNAMES = [
    "algorithm_id",
    "run_idx_within_algorithm",
    "global_run_idx",
    "run_seed_requested",
    "run_seed_used",
    "greedy_order_strategy_requested",
    "greedy_order_strategy_effective",
    "sampling_strategy_requested",
    "sampling_strategy_effective",
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


def normalize_greedy_order_strategy(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"deterministic", "default", "area_desc", "sorted"}:
        return "deterministic"
    if v in {"random", "shuffle", "random_order"}:
        return "random"
    raise ValueError(f"Unsupported greedy order strategy: {value}")


def normalize_sampling_strategy(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"random", "random_sample", "default"}:
        return "random"
    if v in {"smallest", "smallest_area", "small"}:
        return "smallest_area"
    if v in {"largest", "largest_area", "biggest"}:
        return "largest_area"
    raise ValueError(f"Unsupported sampling strategy: {value}")


def strategy_slug(sampling_strategy: str) -> str:
    if sampling_strategy == "smallest_area":
        return "smallest"
    if sampling_strategy == "largest_area":
        return "largest"
    return "random"


def normalize_crop_selection_mode(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"fixed_height", "fixed", "default", "crop_height"}:
        return "fixed_height"
    if v in {
        "unpacked_lowest_y",
        "unpacked_lowest",
        "lowest_unpacked_y",
        "adaptive_lowest",
    }:
        return "unpacked_lowest_y"
    raise ValueError(f"Unsupported crop selection mode: {value}")


def resolve_solver_class(sampling_strategy: str):
    if sampling_strategy == "smallest_area":
        return HybridSolverSmallestSampling
    if sampling_strategy == "largest_area":
        return HybridSolverLargestSampling
    return HybridSolverRandomSampling


def build_algorithm_variants() -> List[Dict[str, str]]:
    greedy_strategy = normalize_greedy_order_strategy(GREEDY_ORDER_STRATEGY)
    sampling_strategy = normalize_sampling_strategy(SAMPLING_STRATEGY)
    algorithm_id = (
        f"greedy_{greedy_strategy}__sampling_{strategy_slug(sampling_strategy)}"
    )
    return [
        {
            "algorithm_id": algorithm_id,
            "greedy_order_strategy": greedy_strategy,
            "sampling_strategy": sampling_strategy,
        }
    ]


def build_summary_text(run_rows: List[Dict[str, Any]], iteration_rows: List[Dict[str, Any]]) -> str:
    by_algorithm_runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_algorithm_iterations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in run_rows:
        by_algorithm_runs[str(row.get("algorithm_id"))].append(row)
    for row in iteration_rows:
        by_algorithm_iterations[str(row.get("algorithm_id"))].append(row)

    lines: List[str] = []
    lines.append("hybrid algorithms benchmark summary")
    lines.append(f"created_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"algorithms_total: {len(by_algorithm_runs)}")
    lines.append("")

    for algorithm_id in sorted(by_algorithm_runs.keys()):
        algo_runs = by_algorithm_runs.get(algorithm_id, [])
        algo_iters = by_algorithm_iterations.get(algorithm_id, [])
        if not algo_runs:
            continue

        first = algo_runs[0]
        greedy_values = [float(r["greedy_objective"]) for r in algo_runs if r.get("greedy_objective") is not None]
        final_values = [float(r["final_objective"]) for r in algo_runs if r.get("final_objective") is not None]
        deltas = [float(r["final_minus_greedy"]) for r in algo_runs if r.get("final_minus_greedy") is not None]
        improved_runs = [r for r in algo_runs if bool(r.get("improved_over_greedy"))]

        iter_deltas = [float(r["iteration_minus_greedy"]) for r in algo_iters if r.get("iteration_minus_greedy") is not None]
        iter_improved = [r for r in algo_iters if bool(r.get("iteration_improved_vs_greedy"))]
        iter_feasible = [
            r
            for r in algo_iters
            if r.get("iteration_status") in {"OPTIMAL", "FEASIBLE"} and r.get("iteration_objective") is not None
        ]

        final_std = safe_pstdev(final_values)
        final_mean = safe_mean(final_values)
        cv_percent = None
        if final_std is not None and final_mean not in (None, 0.0):
            cv_percent = 100.0 * float(final_std) / float(final_mean)

        lines.append(f"algorithm: {algorithm_id}")
        lines.append(f"greedy_order_strategy: {first.get('greedy_order_strategy_effective')}")
        lines.append(f"sampling_strategy: {first.get('sampling_strategy_effective')}")
        lines.append("quality_vs_greedy")
        lines.append(f"runs_total: {len(algo_runs)}")
        lines.append(f"runs_with_final_objective: {len(final_values)}")
        lines.append(f"avg_greedy_objective: {safe_mean(greedy_values)}")
        lines.append(f"avg_final_objective: {safe_mean(final_values)}")
        lines.append(f"avg_final_minus_greedy: {safe_mean(deltas)}")
        lines.append(f"min_final_minus_greedy: {min(deltas) if deltas else None}")
        lines.append(f"max_final_minus_greedy: {max(deltas) if deltas else None}")
        lines.append("improvement_frequency")
        lines.append(f"improved_runs_count: {len(improved_runs)}")
        lines.append(
            f"improved_runs_rate_percent: {100.0 * len(improved_runs) / len(algo_runs) if algo_runs else 0.0:.2f}"
        )
        lines.append("robustness_to_randomness")
        lines.append(f"final_objective_stddev: {final_std}")
        lines.append(f"final_objective_cv_percent: {cv_percent}")
        lines.append(f"final_objective_min: {min(final_values) if final_values else None}")
        lines.append(f"final_objective_max: {max(final_values) if final_values else None}")
        lines.append(
            "final_objective_range: "
            f"{(max(final_values) - min(final_values)) if len(final_values) >= 2 else 0.0 if final_values else None}"
        )
        lines.append("iteration_level")
        lines.append(f"iterations_total_logged: {len(algo_iters)}")
        lines.append(f"iterations_feasible_count: {len(iter_feasible)}")
        lines.append(
            f"iterations_feasible_rate_percent: {100.0 * len(iter_feasible) / len(algo_iters) if algo_iters else 0.0:.2f}"
        )
        lines.append(f"iterations_improved_vs_greedy_count: {len(iter_improved)}")
        lines.append(
            "iterations_improved_vs_greedy_rate_percent: "
            f"{100.0 * len(iter_improved) / len(algo_iters) if algo_iters else 0.0:.2f}"
        )
        lines.append(f"avg_iteration_minus_greedy: {safe_mean(iter_deltas)}")
        lines.append("")

    return "\n".join(lines)


def persist_intermediate(
    run_rows: List[Dict[str, Any]],
    iteration_rows: List[Dict[str, Any]],
) -> None:
    write_csv(RESULTS_DIR / "run_metrics.csv", run_rows, RUN_FIELDNAMES)
    write_csv(RESULTS_DIR / "iteration_metrics.csv", iteration_rows, ITERATION_FIELDNAMES)
    (RESULTS_DIR / "summary.txt").write_text(build_summary_text(run_rows, iteration_rows), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_IMAGES:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    algorithm_variants = build_algorithm_variants()

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_file": str(DATA_FILE.resolve()),
        "R": int(R),
        "S": int(S),
        "height": float(HEIGHT),
        "width": float(WIDTH),
        "solver_name": SOLVER_NAME,
        "num_runs_per_algorithm": int(NUM_RUNS),
        "base_random_seed": BASE_RANDOM_SEED,
        "unpack_last_n": int(UNPACK_LAST_N),
        "crop_height_ratio": float(CROP_HEIGHT_RATIO),
        "crop_selection_mode": str(CROP_SELECTION_MODE),
        "crop_zero_tolerance": float(CROP_ZERO_TOLERANCE),
        "crop_lowest_multiplier": float(CROP_LOWEST_MULTIPLIER),
        "free_space_improvement": FREE_SPACE_IMPROVEMENT,
        "early_stop_free_space_improvement": EARLY_STOP_FREE_SPACE_IMPROVEMENT,
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
        "greedy_order_strategy": GREEDY_ORDER_STRATEGY,
        "sampling_strategy": SAMPLING_STRATEGY,
        "force_unique_random_seeds_per_run": bool(FORCE_UNIQUE_RANDOM_SEEDS_PER_RUN),
        "run_tag_requested": RUN_TAG,
        "auto_run_tag": bool(AUTO_RUN_TAG),
        "run_tag_effective": EFFECTIVE_RUN_TAG,
        "results_dir": str(RESULTS_DIR.resolve()),
        "algorithm_variants": algorithm_variants,
    }
    (RESULTS_DIR / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    items = util_model.parse_items(str(DATA_FILE))
    if not items:
        raise RuntimeError(f"No items parsed from file: {DATA_FILE}")

    print(f"Data file: {DATA_FILE.resolve()}")
    print(
        "Algorithm selection: "
        f"greedy_order_strategy={GREEDY_ORDER_STRATEGY}, sampling_strategy={SAMPLING_STRATEGY}"
    )
    print(
        "Crop selection: "
        f"mode={CROP_SELECTION_MODE}, zero_tolerance={CROP_ZERO_TOLERANCE}, "
        f"lowest_multiplier={CROP_LOWEST_MULTIPLIER}"
    )
    print(f"Run tag: requested={RUN_TAG}, effective={EFFECTIVE_RUN_TAG}")
    print(f"Results dir: {RESULTS_DIR.resolve()}")
    print(f"Algorithm variants: {len(algorithm_variants)}")
    print(f"Runs per algorithm: {NUM_RUNS}, random_iterations per run: {RANDOM_ITERATIONS}")
    print(f"Force unique random seeds per run: {FORCE_UNIQUE_RANDOM_SEEDS_PER_RUN}")

    data = Data(
        items,
        R,
        parallel_nfp=False,
        shared_memory_cache=SHARED_NFP_CACHE,
    )

    run_rows: List[Dict[str, Any]] = []
    iteration_rows: List[Dict[str, Any]] = []
    global_run_idx = 0
    system_rng = random.SystemRandom()
    used_model_seeds: set[int] = set()
    used_greedy_seeds: set[int] = set()

    for variant in algorithm_variants:
        algorithm_id = str(variant["algorithm_id"])
        greedy_strategy = str(variant["greedy_order_strategy"])
        sampling_strategy = str(variant["sampling_strategy"])
        solver_cls = resolve_solver_class(sampling_strategy)

        print(
            f"\n=== Algorithm: {algorithm_id} "
            f"(greedy={greedy_strategy}, sampling={sampling_strategy}) ==="
        )
        if greedy_strategy == "random" and GREEDY_USE_RESULT_CACHE:
            print(
                "Random greedy selected: greedy result cache is force-disabled per run "
                "to guarantee fresh packing."
            )

        algo_image_dir: Optional[Path] = None
        if SAVE_IMAGES:
            algo_image_dir = IMAGES_DIR / algorithm_id
            algo_image_dir.mkdir(parents=True, exist_ok=True)

        for run_idx in range(1, int(NUM_RUNS) + 1):
            global_run_idx += 1
            if BASE_RANDOM_SEED is None:
                if FORCE_UNIQUE_RANDOM_SEEDS_PER_RUN:
                    run_seed = draw_unique_seed(used_model_seeds, system_rng)
                else:
                    run_seed = None
            else:
                run_seed = int(BASE_RANDOM_SEED) + (run_idx - 1)

            if greedy_strategy == "random":
                if BASE_RANDOM_SEED is None:
                    if FORCE_UNIQUE_RANDOM_SEEDS_PER_RUN:
                        greedy_seed_requested = draw_unique_seed(used_greedy_seeds, system_rng)
                    else:
                        greedy_seed_requested = None
                else:
                    greedy_seed_requested = int(BASE_RANDOM_SEED) + 10_000_000 + (run_idx - 1)
            else:
                greedy_seed_requested = None

            effective_greedy_use_result_cache = bool(
                GREEDY_USE_RESULT_CACHE and greedy_strategy != "random"
            )
            run_started = datetime.now().isoformat(timespec="seconds")

            solver = solver_cls(
                data=data,
                height=HEIGHT,
                width=WIDTH,
                S=S,
                solver_name=SOLVER_NAME,
                greedy_enable_output=GREEDY_ENABLE_OUTPUT,
                greedy_use_result_cache=effective_greedy_use_result_cache,
                greedy_result_cache_path=GREEDY_RESULT_CACHE_PATH,
                greedy_result_cache_ttl_days=GREEDY_RESULT_CACHE_TTL_DAYS,
                greedy_shared_result_cache=GREEDY_SHARED_RESULT_CACHE,
                greedy_order_strategy=greedy_strategy,
                greedy_random_seed=greedy_seed_requested,
                hybrid_enable_output=HYBRID_ENABLE_OUTPUT,
            )

            t0 = time.perf_counter()
            result = solver.solve(
                unpack_last_n=UNPACK_LAST_N,
                crop_height=CROP_HEIGHT_RATIO * HEIGHT,
                crop_selection_mode=CROP_SELECTION_MODE,
                crop_zero_tolerance=CROP_ZERO_TOLERANCE,
                crop_lowest_multiplier=CROP_LOWEST_MULTIPLIER,
                use_top_crop=True,
                free_space_improvement=FREE_SPACE_IMPROVEMENT,
                early_stop_free_space_improvement=EARLY_STOP_FREE_SPACE_IMPROVEMENT,
                solver_gap=SOLVER_GAP,
                model_time_limit_sec=MODEL_TIME_LIMIT_SEC,
                model_num_threads=MODEL_NUM_THREADS,
                stop_after_first_solution=STOP_AFTER_FIRST_SOLUTION,
                model_enable_output=MODEL_ENABLE_OUTPUT,
                min_unpacked_in_sample=MIN_UNPACKED_IN_SAMPLE,
                random_iterations=RANDOM_ITERATIONS,
                random_seed=run_seed,
                random_sample_size=RANDOM_SAMPLE_SIZE,
                greedy_order_strategy=greedy_strategy,
                greedy_random_seed=greedy_seed_requested,
            )
            elapsed = time.perf_counter() - t0
            run_finished = datetime.now().isoformat(timespec="seconds")

            image_file: Optional[Path] = None
            if SAVE_IMAGES and algo_image_dir is not None:
                image_file = algo_image_dir / f"run_{run_idx:04d}.png"
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

            sampling_effective = normalize_sampling_strategy(
                str(stats.get("sampling_strategy") or sampling_strategy)
            )
            greedy_effective = normalize_greedy_order_strategy(
                str(stats.get("greedy_order_strategy") or greedy_strategy)
            )
            crop_mode_effective = normalize_crop_selection_mode(
                str(stats.get("crop_selection_mode_effective") or CROP_SELECTION_MODE)
            )

            run_row = {
                "algorithm_id": algorithm_id,
                "run_idx_within_algorithm": int(run_idx),
                "global_run_idx": int(global_run_idx),
                "run_started": run_started,
                "run_finished": run_finished,
                "elapsed_wall_sec": float(elapsed),
                "run_seed_requested": run_seed,
                "run_seed_used": stats.get("random_seed_used"),
                "status": result.get("status") if isinstance(result, dict) else "UNKNOWN",
                "selected_solution": result.get("selected_solution") if isinstance(result, dict) else None,
                "model_status": stats.get("model_status"),
                "greedy_order_strategy_requested": greedy_strategy,
                "greedy_order_strategy_effective": greedy_effective,
                "greedy_random_seed_requested": greedy_seed_requested,
                "greedy_random_seed_used": stats.get("greedy_random_seed_used"),
                "sampling_strategy_requested": sampling_strategy,
                "sampling_strategy_effective": sampling_effective,
                "crop_selection_mode_requested": str(CROP_SELECTION_MODE),
                "crop_selection_mode_effective": crop_mode_effective,
                "crop_decision": stats.get("crop_decision"),
                "crop_zero_tolerance": to_float(stats.get("crop_zero_tolerance")),
                "crop_lowest_multiplier": to_float(stats.get("crop_lowest_multiplier")),
                "lowest_unpacked_y_for_crop": to_float(stats.get("lowest_unpacked_y_for_crop")),
                "scaled_lowest_distance_for_crop": to_float(
                    stats.get("scaled_lowest_distance_for_crop")
                ),
                "used_crop_height": to_float(stats.get("used_crop_height")),
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
                        "algorithm_id": algorithm_id,
                        "run_idx_within_algorithm": int(run_idx),
                        "global_run_idx": int(global_run_idx),
                        "run_seed_requested": run_seed,
                        "run_seed_used": stats.get("random_seed_used"),
                        "greedy_order_strategy_requested": greedy_strategy,
                        "greedy_order_strategy_effective": greedy_effective,
                        "sampling_strategy_requested": sampling_strategy,
                        "sampling_strategy_effective": sampling_effective,
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
                "algorithm_id": algorithm_id,
                "run_idx_within_algorithm": int(run_idx),
                "global_run_idx": int(global_run_idx),
                "sampling_strategy_requested": sampling_strategy,
                "greedy_order_strategy_requested": greedy_strategy,
                "greedy_random_seed_requested": greedy_seed_requested,
                "run_seed_requested": run_seed,
                "run_seed_used": stats.get("random_seed_used"),
                "run_started": run_started,
                "run_finished": run_finished,
                "elapsed_wall_sec": float(elapsed),
                "image_file": str(image_file.resolve()) if image_file is not None else None,
                "result": result,
            }
            (RUNS_DIR / f"{algorithm_id}__run_{run_idx:04d}.json").write_text(
                json.dumps(run_json, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            print(
                f"{algorithm_id} run {run_idx}/{NUM_RUNS}: status={run_row['status']}, "
                f"model_status={run_row['model_status']}, improved={run_row['improved_over_greedy']}, "
                f"final_minus_greedy={run_row['final_minus_greedy']}"
            )

            persist_intermediate(run_rows, iteration_rows)
            print(f"Saved summary (intermediate): {RESULTS_DIR / 'summary.txt'}")

    persist_intermediate(run_rows, iteration_rows)

    print(f"\nSaved: {RESULTS_DIR / 'run_metrics.csv'}")
    print(f"Saved: {RESULTS_DIR / 'iteration_metrics.csv'}")
    print(f"Saved: {RESULTS_DIR / 'summary.txt'}")
    print(f"Saved run JSON files: {RUNS_DIR}")


if __name__ == "__main__":
    main()
