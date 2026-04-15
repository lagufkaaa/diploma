import math
import os
import sqlite3
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, MutableMapping, Optional

import numpy as np
from numpy import array
from shapely import wkb as shapely_wkb
from shapely.geometry import Polygon
from shapely.ops import unary_union

from utils.helpers import util_NFP, util_model

NFP_CACHE_VERSION = "nfp_v1"
DEFAULT_CACHE_DIR = Path("cache") / "nfp"
DEFAULT_CACHE_FILE = "nfp_cache.sqlite3"
ENV_CACHE_PATH = "DIPLOMA_NFP_CACHE_PATH"
ENV_CACHE_DIR = "DIPLOMA_CACHE_DIR"
_GLOBAL_NFP_MEMORY_CACHE: Dict[str, bytes] = {}

import re


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_cache_path(path: Path) -> Path:
    """
    Normalize cache path to an absolute file path.
    - Relative paths are resolved from project root.
    - Directory-like paths are converted to <dir>/nfp_cache.sqlite3.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = _project_root() / p

    file_suffixes = {".sqlite", ".sqlite3", ".db"}
    if p.suffix.lower() not in file_suffixes:
        p = p / DEFAULT_CACHE_FILE

    return p.resolve(strict=False)


def _default_cache_path() -> Path:
    env_cache_path = os.getenv(ENV_CACHE_PATH)
    if env_cache_path:
        return _normalize_cache_path(Path(env_cache_path))

    env_cache_dir = os.getenv(ENV_CACHE_DIR)
    if env_cache_dir:
        return _normalize_cache_path(Path(env_cache_dir))

    return _normalize_cache_path(_project_root() / DEFAULT_CACHE_DIR / DEFAULT_CACHE_FILE)


def resolve_nfp_cache_path(path: Optional[str], identifier: Optional[str] = None) -> Path:
    """
    Resolve an NFP cache path.
    - If `path` is provided, normalize and return it.
    - If `identifier` is provided and `path` is None, create a per-identifier
      cache file under the default cache dir: `nfp_cache_{identifier}.sqlite3`.
    - Otherwise return the default cache path.
    """
    if path:
        return _normalize_cache_path(Path(path))

    if identifier:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(identifier))
        p = _project_root() / DEFAULT_CACHE_DIR / f"nfp_cache_{safe}.sqlite3"
        return _normalize_cache_path(p)

    return _default_cache_path()


def _points_signature(points: np.ndarray) -> str:
    arr = np.asarray(points, dtype=np.float64)
    return arr.tobytes().hex()


def _pair_cache_key(signature_i: str, signature_j: str) -> str:
    return f"{NFP_CACHE_VERSION}:{signature_i}:{signature_j}"


def _compute_nfp_batch(payload):
    """
    Worker payload:
      (i, points_i, jobs)
      jobs: list[(j, points_j, cache_key)]
    Returns:
      list[(i, j, cache_key, geom_wkb, compute_ms)]
    """
    i, points_i, jobs = payload
    points_i_arr = np.asarray(points_i, dtype=float)
    poly_i = Polygon(points_i_arr)
    anchor_point = (float(points_i_arr[0][0]), float(points_i_arr[0][1]))

    results = []
    for j, points_j, cache_key in jobs:
        t0 = time.perf_counter()
        points_j_arr = np.asarray(points_j, dtype=float)
        poly_j = Polygon(points_j_arr)

        minkowski = util_NFP.minkowski_difference(poly_i, poly_j, anchor_point)
        base_polygon = poly_j.buffer(0)
        geom = unary_union([base_polygon, minkowski])

        results.append((i, j, cache_key, geom.wkb, (time.perf_counter() - t0) * 1000.0))
    return results


class _NFPDiskCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=60.0)
        self._ensure_schema()

    def _ensure_schema(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nfp_cache (
                cache_key TEXT PRIMARY KEY,
                geom_wkb BLOB NOT NULL,
                created_at REAL NOT NULL,
                compute_ms REAL
            )
            """
        )
        self.conn.commit()

    def get(self, cache_key: str, ttl_seconds: Optional[float] = None) -> Optional[bytes]:
        row = self.conn.execute(
            "SELECT geom_wkb, created_at FROM nfp_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None

        if ttl_seconds is not None:
            created_at = float(row[1])
            if time.time() - created_at > ttl_seconds:
                return None

        return bytes(row[0])

    def put_many(self, rows):
        if not rows:
            return
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO nfp_cache(cache_key, geom_wkb, created_at, compute_ms)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


class Data:
    def __init__(
        self,
        items: list,
        R: int,
        *,
        parallel_nfp: bool = True,
        nfp_workers: Optional[int] = None,
        use_cache: bool = True,
        cache_path: Optional[str] = None,
        cache_identifier: Optional[str] = None,
        cache_ttl_days: Optional[float] = None,
        use_memory_cache: bool = True,
        shared_memory_cache: Optional[MutableMapping[str, bytes]] = None,
        enable_progress_log: bool = False,
        log_interval_sec: float = 2.0,
    ):
        self.R = R
        self.angle = 360 / R
        self.N = len(items)
        self.parallel_nfp = bool(parallel_nfp)
        self.nfp_workers = self._resolve_workers(nfp_workers)
        self.use_cache = bool(use_cache)
        self.cache_path = resolve_nfp_cache_path(cache_path, cache_identifier)
        self.cache_ttl_seconds = (
            None if cache_ttl_days is None else max(0.0, float(cache_ttl_days) * 24.0 * 3600.0)
        )
        self.use_memory_cache = bool(use_memory_cache)
        self.memory_cache: Optional[MutableMapping[str, bytes]] = (
            shared_memory_cache
            if shared_memory_cache is not None
            else (_GLOBAL_NFP_MEMORY_CACHE if self.use_memory_cache else None)
        )
        if not self.use_memory_cache:
            self.memory_cache = None

        # Progress logging for long-running NFP build
        self.enable_progress_log = bool(enable_progress_log)
        self.log_interval_sec = max(0.1, float(log_interval_sec))

        if items and isinstance(items[0], Item):
            items_with_rotation, dict_rot = self._get_items_with_rotation(items)
        else:
            items_with_rotation, dict_rot = self._get_items_with_rotation([Item(points) for points in items])

        self.items = items_with_rotation
        self.dict_rot = dict_rot

        for it in self.items:
            it.data = self
        self._build_nfp()

    def _resolve_workers(self, nfp_workers: Optional[int]) -> int:
        if nfp_workers is not None:
            return max(1, int(nfp_workers))
        cpu = os.cpu_count() or 1
        return max(1, min(8, cpu))

    def _build_nfp(self):
        started = time.perf_counter()
        cache = _NFPDiskCache(self.cache_path) if self.use_cache else None

        pair_geoms: Dict[tuple[int, int], object] = {}
        pending_by_i: Dict[int, list] = {}
        cache_rows_to_write = []

        total_pairs = 0
        memory_cache_hits = 0
        disk_cache_hits = 0
        cache_hits = 0
        cache_misses = 0

        points_lists = [it.points.tolist() for it in self.items]
        signatures = [_points_signature(it.points) for it in self.items]

        try:
            last_log_ts = started - self.log_interval_sec
            estimated_total_pairs = max(0, self.N * (self.N - 1))

            for i, it_i in enumerate(self.items):
                for j, it_j in enumerate(self.items):
                    if it_j.id == it_i.id:
                        continue

                    total_pairs += 1
                    cache_key = _pair_cache_key(signatures[i], signatures[j])

                    if self.memory_cache is not None:
                        cached_wkb = self.memory_cache.get(cache_key)
                        if cached_wkb is not None:
                            pair_geoms[(i, j)] = shapely_wkb.loads(cached_wkb)
                            memory_cache_hits += 1
                            cache_hits += 1
                            continue

                    if cache is not None:
                        cached_wkb = cache.get(cache_key, self.cache_ttl_seconds)
                        if cached_wkb is not None:
                            pair_geoms[(i, j)] = shapely_wkb.loads(cached_wkb)
                            if self.memory_cache is not None:
                                self.memory_cache[cache_key] = cached_wkb
                            disk_cache_hits += 1
                            cache_hits += 1
                            continue

                    cache_misses += 1
                    pending_by_i.setdefault(i, []).append((j, points_lists[j], cache_key))

                # Periodic progress log while scanning cache/miss status
                if self.enable_progress_log:
                    now = time.perf_counter()
                    if (now - last_log_ts) >= self.log_interval_sec:
                        print(
                            f"[nfp] scan progress: i={i+1}/{self.N}, total_pairs={total_pairs}/{estimated_total_pairs}, "
                            f"mem_hits={memory_cache_hits}, disk_hits={disk_cache_hits}, misses={cache_misses}",
                            flush=True,
                        )
                        last_log_ts = now

            payloads = []
            for i, jobs in pending_by_i.items():
                payloads.append((i, points_lists[i], jobs))

            computed_pairs = 0
            total_to_compute = cache_misses
            if self.enable_progress_log:
                print(
                    f"[nfp] compute phase: payloads={len(payloads)}, to_compute={total_to_compute}, "
                    f"parallel={self.parallel_nfp}, workers={self.nfp_workers}",
                    flush=True,
                )

            if payloads:
                if self.parallel_nfp and self.nfp_workers > 1 and len(payloads) > 1:
                    with ProcessPoolExecutor(max_workers=self.nfp_workers) as executor:
                        futures = [executor.submit(_compute_nfp_batch, payload) for payload in payloads]
                        for future in as_completed(futures):
                            rows = future.result()
                            now_ts = time.time()
                            for i, j, cache_key, geom_wkb, compute_ms in rows:
                                pair_geoms[(i, j)] = shapely_wkb.loads(geom_wkb)
                                computed_pairs += 1
                                if self.memory_cache is not None:
                                    self.memory_cache[cache_key] = bytes(geom_wkb)
                                if cache is not None:
                                    cache_rows_to_write.append(
                                        (cache_key, sqlite3.Binary(geom_wkb), now_ts, compute_ms)
                                    )
                            if self.enable_progress_log:
                                now = time.perf_counter()
                                if (now - last_log_ts) >= self.log_interval_sec:
                                    print(
                                        f"[nfp] compute progress: computed_pairs={computed_pairs}/{total_to_compute}",
                                        flush=True,
                                    )
                                    last_log_ts = now
                else:
                    for payload in payloads:
                        rows = _compute_nfp_batch(payload)
                        now_ts = time.time()
                        for i, j, cache_key, geom_wkb, compute_ms in rows:
                            pair_geoms[(i, j)] = shapely_wkb.loads(geom_wkb)
                            computed_pairs += 1
                            if self.memory_cache is not None:
                                self.memory_cache[cache_key] = bytes(geom_wkb)
                            if cache is not None:
                                cache_rows_to_write.append(
                                    (cache_key, sqlite3.Binary(geom_wkb), now_ts, compute_ms)
                                )
                        if self.enable_progress_log:
                            now = time.perf_counter()
                            if (now - last_log_ts) >= self.log_interval_sec:
                                print(
                                    f"[nfp] compute progress: computed_pairs={computed_pairs}/{total_to_compute}",
                                    flush=True,
                                )
                                last_log_ts = now

            if cache is not None and cache_rows_to_write:
                cache.put_many(cache_rows_to_write)
                if self.enable_progress_log:
                    print(
                        f"[nfp] wrote {len(cache_rows_to_write)} rows to disk cache: {self.cache_path}",
                        flush=True,
                    )

            for i, it_i in enumerate(self.items):
                nfp_dict = {}
                for j, it_j in enumerate(self.items):
                    if it_j.id == it_i.id:
                        continue
                    nfp_dict[it_j] = pair_geoms[(i, j)]
                it_i.nfp = nfp_dict

            self.nfp_stats = {
                "pairs_total": total_pairs,
                "cache_enabled": self.use_cache,
                "cache_path": str(self.cache_path) if self.use_cache else None,
                "memory_cache_enabled": self.memory_cache is not None,
                "memory_cache_hits": memory_cache_hits,
                "disk_cache_hits": disk_cache_hits,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "computed_pairs": computed_pairs,
                "cache_rows_written": len(cache_rows_to_write),
                "parallel_enabled": self.parallel_nfp,
                "workers_used": self.nfp_workers if self.parallel_nfp else 1,
                "elapsed_sec": time.perf_counter() - started,
            }
            if self.enable_progress_log:
                print(
                    (
                        f"[nfp] build finished: pairs_total={total_pairs}, computed={computed_pairs}, "
                        f"mem_hits={memory_cache_hits}, disk_hits={disk_cache_hits}, misses={cache_misses}, "
                        f"elapsed_sec={self.nfp_stats['elapsed_sec']:.2f}"
                    ),
                    flush=True,
                )
        finally:
            if cache is not None:
                cache.close()

    def _get_items_with_rotation(self, items):
        temp_dict = {}
        temp_all_items = []

        for it in items:
            temp_dict[it] = []
            # keep the original orientation
            temp_all_items.append(it)
            temp_dict[it].append(it)

            for r in range(1, self.R):
                new_it = Item(it.points.copy())
                # Same physical piece in another rotation -> keep group id.
                new_it.id = it.id
                new_it.rotation = it.rotation
                new_it.change_rotation(r * self.angle)
                temp_all_items.append(new_it)
                temp_dict[it].append(new_it)
        return temp_all_items, temp_dict
        
class Item:
    def __init__(self, points: array, data: 'Data' = None):
        self.id = uuid.uuid4() # одинаковый у разных поворотов одного предмета!!!!
        self.nfp = None
        self.rotation = 0

        pts = np.asarray(points, dtype=float)
        anchor = pts[0].copy()
        pts = pts - anchor
        self.points = pts
        
        self.data = data
        self.polygon = Polygon(self.points)
        
        self.area = self.polygon.area

        bbox = util_model.find_bounding_box_numpy(self.points)
        
        self.xmin = bbox['min_x']
        self.xmax = bbox['max_x'] 
        self.ymin = bbox['min_y']
        self.ymax = bbox['max_y']


    def area(self):
        return self.polygon.area
    
    def change_rotation(self, angle: int):
        rotated_points = []
        angle_rad = math.radians(angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        for x, y in self.points:
            x_rotated = x * cos_angle - y * sin_angle
            y_rotated = x * sin_angle + y * cos_angle
            rotated_points.append((x_rotated, y_rotated))

        self.points = array(rotated_points)
        self.polygon = Polygon(self.points)
        self.area = self.polygon.area
        bbox = util_model.find_bounding_box_numpy(self.points)
        self.xmin = bbox['min_x']
        self.xmax = bbox['max_x']
        self.ymin = bbox['min_y']
        self.ymax = bbox['max_y']
        self.rotation = (self.rotation + angle) % 360

    def compute_nfp(self):
        """Вычислить и сохранить NFP — вызывается после присоединения `data` к Item."""
        if self.data is None:
            return
        self.nfp = self.NFP(self.data)

    def outer_NFP(self, other_item, anchor_point=(0, 0)): # TODO Почему anchor_point=(0, 0) а не первая точка в points?
        if not isinstance(self.polygon, Polygon):
            raise TypeError(f"[outer_no_fit_polygon] polygon1 должен быть Polygon, получен {type(self.polygon).__name__}")
        if not isinstance(other_item.polygon, Polygon):
            raise TypeError(f"[outer_no_fit_polygon] polygon2 должен быть Polygon, получен {type(other_item.polygon).__name__}")

        minkowski = util_NFP.minkowski_difference(self.polygon, other_item.polygon, anchor_point)
        base_polygon = other_item.polygon.buffer(0)

        return unary_union([base_polygon, minkowski])
    
    def NFP(self, data):
        anchor_point = self.points[0]
        NFP_dict = {}
        for item in data.items:
            if item.id == self.id:
                continue

            NFP_dict[item] = self.outer_NFP(item, anchor_point)

        return NFP_dict
