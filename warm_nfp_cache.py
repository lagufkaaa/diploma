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


from core.data import Data
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
        "--no-memory-cache",
        action="store_true",
        help="Disable in-process memory cache while warming disk cache.",
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

    data = Data(
        items,
        R=max(1, int(args.rotations)),
        parallel_nfp=not bool(args.serial),
        nfp_workers=args.workers,
        use_cache=True,
        cache_path=args.cache_path,
        cache_identifier=file_path.stem,
        cache_ttl_days=args.cache_ttl_days,
        use_memory_cache=not bool(args.no_memory_cache),
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
