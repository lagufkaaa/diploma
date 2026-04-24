from typing import Optional


def should_restart_greedy_after_empty_unpack(
    *,
    greedy_order_strategy: str,
    unpack_ids_count: int,
    restarts_used: int,
    max_restarts: int,
) -> bool:
    return (
        str(greedy_order_strategy) == "random"
        and int(unpack_ids_count) <= 0
        and int(restarts_used) < int(max_restarts)
    )


def next_greedy_random_seed_requested(
    *,
    initial_seed_requested: Optional[int],
    restart_index: int,
) -> Optional[int]:
    if initial_seed_requested is None:
        return None
    return int(initial_seed_requested) + int(restart_index)
