#!/usr/bin/env python3
"""Test Widesurf API concurrency limits: 100 and 1000 concurrent requests."""
import requests, time, concurrent.futures

API_URL = 'http://10.0.0.94:8020'
API_KEY = '69xn13ehEccqzDJxw29KH0mzzbIIgI2NRttD7m6p9gA'

SYMBOLS = [
    'TSLA','AAPL','MSFT','NVDA','AMZN','GOOG','META','AMD','NFLX','INTC',
    'JPM','V','MA','BAC','WFC','GS','MS','C','AXP','BLK',
    'UNH','JNJ','PFE','MRK','ABBV','LLY','TMO','ABT','BMY','AMGN',
    'HD','MCD','NKE','SBUX','TGT','WMT','COST','LOW','TJX','DG',
    'DIS','CMCSA','T','VZ','TMUS','CHTR','CRM','PARA','WBD','FOX',
    'XOM','CVX','COP','EOG','SLB','MPC','PSX','VLO','OXY','HAL',
    'CAT','DE','BA','RTX','LMT','GE','HON','MMM','UPS','FDX',
    'ORCL','SAP','ADBE','NOW','SNOW','DDOG','PANW','ZS','CRWD','SHOP',
    'SQ','PYPL','COIN','MELI','SE','GRAB','NU','SOFI','HOOD','ROKU',
    'SPOT','UBER','LYFT','DASH','ABNB','RBLX','U','PLTR','SNAP','PINS',
]

def fetch_one(sym):
    start = time.time()
    try:
        resp = requests.get(
            f'{API_URL}/v1/aggs/ticker/{sym}/range/1/minute/2024-01-01/2025-01-01/1000',
            params={'apiKey': API_KEY, 'adjusted': 'true', 'sort': 'asc', 'limit': 50000},
            timeout=30
        )
        elapsed = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            return (sym, len(data.get('results', [])), elapsed, 'OK')
        return (sym, 0, elapsed, f'HTTP {resp.status_code}')
    except Exception as e:
        return (sym, 0, time.time() - start, str(e)[:60])

def run_test(n, label):
    # Build request list by repeating symbols
    reqs = []
    for i in range(n):
        reqs.append(SYMBOLS[i % len(SYMBOLS)])

    print(f'\n{"="*60}')
    print(f'  {label}: {n} CONCURRENT REQUESTS')
    print(f'{"="*60}')

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(fetch_one, reqs))
    total = time.time() - t0

    ok = [r for r in results if r[3] == 'OK']
    fail = [r for r in results if r[3] != 'OK']
    times = [r[2] for r in ok]
    bars = [r[1] for r in ok]

    print(f'  Total wall time: {total:.2f}s')
    print(f'  Success: {len(ok)}/{n} ({len(ok)/n*100:.1f}%)')
    print(f'  Failed:  {len(fail)}/{n}')
    if times:
        print(f'  Latency: avg={sum(times)/len(times):.2f}s  min={min(times):.2f}s  max={max(times):.2f}s')
        print(f'  Throughput: {len(ok)/total:.1f} req/s')
    if bars:
        print(f'  Avg bars/response: {sum(bars)/len(bars):.0f}')
    if fail:
        # Group failures by reason
        reasons = {}
        for r in fail:
            reasons[r[3]] = reasons.get(r[3], 0) + 1
        print(f'  Failure reasons:')
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f'    {reason}: {count}x')

    return len(ok), len(fail), total

# Warm up
print("Warming up with 1 request...")
fetch_one('TSLA')

# Run tests
run_test(100, "TEST 1")
time.sleep(2)
run_test(1000, "TEST 2")
