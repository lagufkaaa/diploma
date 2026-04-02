import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import MutableMapping, Optional

import numpy as np

GREEDY_RESULT_CACHE_VERSION = "greedy_result_v1"
DEFAULT_GREEDY_CACHE_DIR = Path("cache") / "greedy"
DEFAULT_GREEDY_CACHE_FILE = "greedy_result_cache.sqlite3"
ENV_GREEDY_CACHE_PATH = "DIPLOMA_GREEDY_CACHE_PATH"
ENV_GREEDY_CACHE_DIR = "DIPLOMA_GREEDY_CACHE_DIR"

GLOBAL_GREEDY_RESULT_MEMORY_CACHE: dict[str, bytes] = {}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_cache_path(path: Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = _project_root() / p

    file_suffixes = {".sqlite", ".sqlite3", ".db"}
    if p.suffix.lower() not in file_suffixes:
        p = p / DEFAULT_GREEDY_CACHE_FILE
    return p.resolve(strict=False)


def default_greedy_result_cache_path() -> Path:
    env_cache_path = os.getenv(ENV_GREEDY_CACHE_PATH)
    if env_cache_path:
        return _normalize_cache_path(Path(env_cache_path))

    env_cache_dir = os.getenv(ENV_GREEDY_CACHE_DIR)
    if env_cache_dir:
        return _normalize_cache_path(Path(env_cache_dir))

    return _normalize_cache_path(_project_root() / DEFAULT_GREEDY_CACHE_DIR / DEFAULT_GREEDY_CACHE_FILE)


def resolve_greedy_result_cache_path(path: Optional[str]) -> Path:
    if path:
        return _normalize_cache_path(Path(path))
    return default_greedy_result_cache_path()


def _float_token(value: float) -> str:
    return f"{float(value):.10f}"


def build_greedy_result_cache_key(
    data,
    *,
    height: float,
    width: float,
    S: int,
    eps_area: float,
) -> str:
    id_to_group_idx = {}
    items_payload = []

    for it in data.items:
        if it.id not in id_to_group_idx:
            id_to_group_idx[it.id] = len(id_to_group_idx)

        points_arr = np.asarray(it.points, dtype=np.float64)
        points_hash = hashlib.sha256(points_arr.tobytes()).hexdigest()
        items_payload.append(
            {
                "group_idx": int(id_to_group_idx[it.id]),
                "rotation": _float_token(getattr(it, "rotation", 0.0)),
                "points_hash": points_hash,
            }
        )

    payload = {
        "version": GREEDY_RESULT_CACHE_VERSION,
        "R": int(getattr(data, "R", 1)),
        "height": _float_token(height),
        "width": _float_token(width),
        "S": int(S),
        "eps_area": _float_token(eps_area),
        "items": items_payload,
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    return f"{GREEDY_RESULT_CACHE_VERSION}:{digest}"


class GreedyResultDiskCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=60.0)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS greedy_result_cache (
                cache_key TEXT PRIMARY KEY,
                payload BLOB NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, cache_key: str, ttl_seconds: Optional[float] = None) -> Optional[bytes]:
        row = self.conn.execute(
            "SELECT payload, created_at FROM greedy_result_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None

        if ttl_seconds is not None:
            created_at = float(row[1])
            if (time.time() - created_at) > float(ttl_seconds):
                return None

        return bytes(row[0])

    def put(self, cache_key: str, payload: bytes) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO greedy_result_cache(cache_key, payload, created_at)
            VALUES (?, ?, ?)
            """,
            (cache_key, sqlite3.Binary(payload), time.time()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
