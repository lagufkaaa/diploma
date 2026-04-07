import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.data import Data
from solvers.greedy_solver_random import GreedySolverRandom
from utils.helpers import util_model


DATA_DIR = Path(__file__).resolve().parents[2] / "data_car_mats"
SHARED_NFP_CACHE = {}


def test_greedy_random_order_seed_controls_group_order():
    file_path = DATA_DIR / "test.txt"
    items = util_model.parse_items(str(file_path))
    assert len(items) > 0, "no items parsed from test file"

    data = Data(
        items,
        R=1,
        parallel_nfp=False,
        shared_memory_cache=SHARED_NFP_CACHE,
    )

    solver_a = GreedySolverRandom(
        data,
        height=500.0,
        width=500.0,
        S=6,
        random_seed=123,
    )
    solver_b = GreedySolverRandom(
        data,
        height=500.0,
        width=500.0,
        S=6,
        random_seed=123,
    )
    solver_c = GreedySolverRandom(
        data,
        height=500.0,
        width=500.0,
        S=6,
        random_seed=124,
    )

    result_a = solver_a.solve()
    result_b = solver_b.solve()
    result_c = solver_c.solve()

    assert result_a.get("status") == "OPTIMAL"
    assert result_b.get("status") == "OPTIMAL"
    assert result_c.get("status") == "OPTIMAL"

    order_a = result_a.get("group_order_ids")
    order_b = result_b.get("group_order_ids")
    order_c = result_c.get("group_order_ids")

    assert isinstance(order_a, list) and len(order_a) > 0
    assert order_a == order_b
    assert order_a != order_c
    assert result_a.get("group_order_strategy") == "random"
    assert int(result_a.get("random_seed_used")) == 123
