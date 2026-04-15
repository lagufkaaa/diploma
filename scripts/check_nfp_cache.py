import sqlite3
import os
p = os.path.join('cache','nfp','nfp_cache_car_mats_2.sqlite3')
print('checking:', os.path.abspath(p))
conn = sqlite3.connect(p)
cur = conn.execute('SELECT COUNT(*) FROM nfp_cache')
print('rows:', cur.fetchone()[0])
cur = conn.execute('SELECT MIN(created_at), MAX(created_at) FROM nfp_cache')
print('minmax created_at:', cur.fetchone())
conn.close()
