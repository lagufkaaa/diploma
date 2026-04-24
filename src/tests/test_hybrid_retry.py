import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from solvers.hybrid_retry import (
    next_greedy_random_seed_requested,
    should_restart_greedy_after_empty_unpack,
)


def test_should_restart_greedy_after_empty_unpack_only_for_random_and_remaining_retries():
    assert should_restart_greedy_after_empty_unpack(
        greedy_order_strategy="random",
        unpack_ids_count=0,
        restarts_used=0,
        max_restarts=1,
    )
    assert not should_restart_greedy_after_empty_unpack(
        greedy_order_strategy="deterministic",
        unpack_ids_count=0,
        restarts_used=0,
        max_restarts=1,
    )
    assert not should_restart_greedy_after_empty_unpack(
        greedy_order_strategy="random",
        unpack_ids_count=2,
        restarts_used=0,
        max_restarts=1,
    )
    assert not should_restart_greedy_after_empty_unpack(
        greedy_order_strategy="random",
        unpack_ids_count=0,
        restarts_used=1,
        max_restarts=1,
    )


def test_next_greedy_random_seed_requested_keeps_none_or_increments_from_initial_seed():
    assert next_greedy_random_seed_requested(
        initial_seed_requested=None,
        restart_index=1,
    ) is None
    assert next_greedy_random_seed_requested(
        initial_seed_requested=12345,
        restart_index=1,
    ) == 12346
    assert next_greedy_random_seed_requested(
        initial_seed_requested=12345,
        restart_index=3,
    ) == 12348
