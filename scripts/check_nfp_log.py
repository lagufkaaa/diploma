import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.data import Data

items = [
    [[0,0],[100,0],[100,50],[0,50]],
    [[0,0],[60,0],[60,40],[0,40]],
]

print("Creating Data with enable_progress_log=True")
data = Data(items, R=1, parallel_nfp=False, use_cache=True, cache_identifier='test_nfp_log', enable_progress_log=True, log_interval_sec=0.5)
print("Data.nfp_stats:", data.nfp_stats)
