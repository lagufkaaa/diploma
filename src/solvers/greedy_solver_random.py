import random
from typing import Dict, List, MutableMapping, Optional

from core.data import Data, Item
from solvers.greedy_solver import GreedySolver


class GreedySolverRandom(GreedySolver):
    """
    Random-order variant of GreedySolver.

    The packing procedure is the same BLF, but the processing order of item-id
    groups is randomized instead of sorted by descending area.
    """

    def __init__(
        self,
        data: Data,
        height: float,
        width: float,
        S: int = 1,
        eps_area: float = 1e-6,
        enable_progress_log: bool = False,
        log_interval_sec: float = 2.0,
        log_prefix: str = "[greedy-random]",
        use_result_cache: bool = False,
        result_cache_path: Optional[str] = None,
        result_cache_ttl_days: Optional[float] = None,
        shared_result_cache: Optional[MutableMapping[str, bytes]] = None,
        random_seed: Optional[int] = None,
    ):
        # For randomized strategy the result cache is disabled by default.
        # If a caller still enables it, use a separate cache file to avoid
        # collisions with deterministic greedy results.
        if use_result_cache and result_cache_path is None:
            result_cache_path = "cache/greedy_random/greedy_result_cache.sqlite3"

        super().__init__(
            data=data,
            height=height,
            width=width,
            S=S,
            eps_area=eps_area,
            enable_progress_log=enable_progress_log,
            log_interval_sec=log_interval_sec,
            log_prefix=log_prefix,
            use_result_cache=use_result_cache,
            result_cache_path=result_cache_path,
            result_cache_ttl_days=result_cache_ttl_days,
            shared_result_cache=shared_result_cache,
        )

        self.random_seed = random_seed
        self.last_random_seed_used: Optional[int] = None
        self.last_group_order_ids: List[object] = []
        self._group_order_rng: Optional[random.Random] = None

    def solve(self):
        if self.random_seed is None:
            self.last_random_seed_used = random.SystemRandom().randrange(0, 2**63)
        else:
            self.last_random_seed_used = int(self.random_seed)

        self._group_order_rng = random.Random(self.last_random_seed_used)
        self.last_group_order_ids = []
        try:
            result = super().solve()
        finally:
            self._group_order_rng = None

        if isinstance(result, dict):
            result["group_order_strategy"] = "random"
            result["random_seed_used"] = self.last_random_seed_used
            result["group_order_ids"] = list(self.last_group_order_ids)
        return result

    def _build_item_groups(self) -> List[List[Item]]:
        grouped: Dict[object, List[Item]] = {}
        for it in self.data.items:
            grouped.setdefault(it.id, []).append(it)

        groups = list(grouped.values())
        if self._group_order_rng is None:
            groups.sort(
                key=lambda group: max(float(it.area) for it in group),
                reverse=True,
            )
        else:
            self._group_order_rng.shuffle(groups)

        self.last_group_order_ids = [group[0].id for group in groups if group]
        return groups
