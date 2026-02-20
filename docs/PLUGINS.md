# Extensibility Guide

## Adding a New Trading Strategy

### Step 1: GPU Backtester (`optimizer/gpu_backtest.py`)

Create a new class following the pattern of existing backtesters:

```python
class GPUMyStrategyBacktester:
    def __init__(self, bars: list, capital: float = 100000):
        # Load price arrays to GPU/CPU
        # Pre-compute entry signals
        pass

    def grid_search(self, buy_range: tuple, sell_range: tuple, metric: str) -> dict:
        # Return: {"best_params": {...}, "metrics": {...}, "combinations_tested": N}
        pass
```

Create a module-level entry point:

```python
def run_gpu_mystrategy_optimization(bars, capital, buy_range, sell_range, metric):
    bt = GPUMyStrategyBacktester(bars, capital)
    return bt.grid_search(buy_range, sell_range, metric)
```

### Step 2: Optimizer Server (`optimizer/server.py`)

Add a dispatch branch in the optimization handler:

```python
elif algo_type == "my_strategy":
    from gpu_backtest import run_gpu_mystrategy_optimization
    gpu_result = await loop.run_in_executor(
        _GPU_EXECUTOR,
        run_gpu_mystrategy_optimization,
        bars, config.capital, buy_range, sell_range, config.optimization_metric
    )
```

Add a trade log branch in `generate_trade_log()`:

```python
elif algo == "my_strategy":
    # Day-by-day trade simulation matching the GPU backtester
    ...
```

### Step 3: Frontend (`frontend/index.html`)

Add algorithm to the dropdown:

```html
<option value="my_strategy">My Strategy</option>
```

### Step 4: Documentation

Update `MODULES.md` with the new backtester class and its grid dimensions.
Log the design decision in `DECISIONS.md`.

## Adding a New Data Source

### Step 1: Backend API (`backend/server.py`)

Add a fetcher function and route:

```python
async def fetch_from_my_source(symbol, start, end, timeframe):
    # Return list of {"t": timestamp, "o": open, "h": high, "l": low, "c": close, "v": volume}
    pass
```

### Step 2: Optimizer enrichment (`optimizer/server.py`)

Add VWAP calculation logic if the source doesn't provide native VWAP.

## Adding a New Optimization Metric

### Step 1: Backtester

Add metric computation in the grid search loop and include it in the return dict.

### Step 2: Score selection

Add a branch in the metric selection:

```python
if metric == "my_metric":
    score_arr = my_metric_values
```

### Step 3: Rust Engine

If using Rust, add the metric to both `run_backtest()` return and `grid_search()` scoring.
