import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from core.data import Data, resolve_nfp_cache_path
from utils.helpers import util_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Warm the persistent NFP cache for a car_mats file without running "
            "greedy/hybrid benchmarks."
        )
    )
    parser.add_argument(
        "file_path",
        help="Path to the input file, for example data_car_mats/car_mats_10.txt",
    )
    parser.add_argument(
        "--rotations",
        "-R",
        type=int,
        default=4,
        help="Number of allowed rotations (default: 4).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of NFP worker processes. Defaults to Data() auto-detection.",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Disable parallel NFP computation and run in one process.",
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Optional custom SQLite cache path or cache directory.",
    )
    parser.add_argument(
        "--cache-ttl-days",
        type=float,
        default=None,
        help="Optional TTL for reading cached NFP entries. Writes are still persisted.",
    )
    parser.add_argument(
        "--cache-flush-interval-sec",
        type=float,
        default=30.0,
        help="Flush pending NFP rows to disk periodically during compute phase (default: 30s).",
    )
    parser.add_argument(
        "--log-interval-sec",
        type=float,
        default=10.0,
        help="Progress log interval while warming cache (default: 10s).",
    )
    parser.add_argument(
        "--nfp-jobs-per-payload",
        type=int,
        default=32,
        help="How many NFP jobs a worker handles in one batch before reporting back (default: 32).",
    )
    parser.add_argument(
        "--no-memory-cache",
        action="store_true",
        help="Disable in-process memory cache while warming disk cache.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs and print only the final JSON summary.",
    )
    return parser


def _resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve(strict=False)


def _build_summary(
    *,
    file_path: Path,
    items_count: int,
    data: Data,
    elapsed_sec: float,
) -> dict[str, Any]:
    stats = dict(getattr(data, "nfp_stats", {}) or {})
    stats.setdefault("cache_path", str(getattr(data, "cache_path", "")))
    return {
        "file_path": str(file_path),
        "items_count": int(items_count),
        "expanded_items_count": int(len(data.items)),
        "rotations": int(getattr(data, "R", 0)),
        "elapsed_sec_total": float(elapsed_sec),
        "nfp_stats": stats,
    }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    file_path = _resolve_input_path(args.file_path)
    if not file_path.exists():
        parser.error(f"Input file does not exist: {file_path}")

    started = time.perf_counter()
    items = util_model.parse_items(str(file_path))
    if not items:
        parser.error(f"No items parsed from file: {file_path}")

    resolved_cache_path = resolve_nfp_cache_path(args.cache_path, file_path.stem)
    progress_enabled = not bool(args.quiet)
    workers_label = 1 if args.serial else (args.workers if args.workers is not None else "auto")

    if progress_enabled:
        print(
            (
                f"[warm_nfp_cache] file={file_path} items={len(items)} rotations={max(1, int(args.rotations))} "
                f"parallel={not bool(args.serial)} workers={workers_label}"
            ),
            flush=True,
        )
        print(
            (
                f"[warm_nfp_cache] cache_path={resolved_cache_path} "
                f"flush_interval_sec={args.cache_flush_interval_sec} log_interval_sec={args.log_interval_sec} "
                f"jobs_per_payload={args.nfp_jobs_per_payload}"
            ),
            flush=True,
        )

    data = Data(
        items,
        R=max(1, int(args.rotations)),
        parallel_nfp=not bool(args.serial),
        nfp_workers=args.workers,
        use_cache=True,
        cache_path=str(resolved_cache_path),
        cache_ttl_days=args.cache_ttl_days,
        cache_flush_interval_sec=args.cache_flush_interval_sec,
        nfp_jobs_per_payload=args.nfp_jobs_per_payload,
        use_memory_cache=not bool(args.no_memory_cache),
        enable_progress_log=progress_enabled,
        log_interval_sec=args.log_interval_sec,
    )

    summary = _build_summary(
        file_path=file_path,
        items_count=len(items),
        data=data,
        elapsed_sec=time.perf_counter() - started,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
