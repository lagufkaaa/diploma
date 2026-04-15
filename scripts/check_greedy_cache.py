import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.data import Data
from solvers.greedy_solver import GreedySolver
from solvers.greedy_result_cache import resolve_greedy_result_cache_path

# Two simple rectangles
items = [
    [[0,0],[100,0],[100,50],[0,50]],
    [[0,0],[60,0],[60,40],[0,40]],
]

data = Data(items, R=1, parallel_nfp=False, use_cache=True, cache_identifier='test_greedy')

# compute per-data greedy cache path
greedy_cache_path = str(resolve_greedy_result_cache_path(None, identifier='test_greedy'))
print('greedy_cache_path ->', greedy_cache_path)

solver = GreedySolver(
    data=data,
    height=1000,
    width=1000,
    S=1,
    use_result_cache=True,
    result_cache_path=greedy_cache_path,
)

res = solver.solve()
print('result objective', res.get('objective_value') if isinstance(res, dict) else res)

# run a second time to ensure cache hit
solver2 = GreedySolver(
    data=data,
    height=1000,
    width=1000,
    S=1,
    use_result_cache=True,
    result_cache_path=greedy_cache_path,
)
res2 = solver2.solve()
print('second run cache hit:', solver2.last_result_cache_hit)
