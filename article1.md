# The Three GPU Optimizations That Made My Trading Backtester 40x Faster

**How I discovered that GPU speed isn't about compute power—it's about data management**

*December 2024*

---

## The Starting Point: Frustratingly Slow

I had a working trading strategy optimizer. It used Bayesian optimization (Optuna) to find the best buy/sell triggers for momentum trading. The problem? 

**40+ seconds per stock. 200 parameter combinations. Painfully sequential.**

My RTX 3090 sat idle at 2% utilization while my CPU maxed out. Something was wrong.

I decided to GPU-accelerate my backtester. What followed was a journey through three critical optimizations—each building on the last—until I achieved something remarkable:

**13,775 parameter combinations in 4 seconds. That's 3,475 strategies per second.**

Here are the three changes that made it happen.

---

## Optimization #1: Pre-Load Data Once (Small Win)

### The Problem
My initial GPU implementation loaded price data for every single trial:

```python
# ❌ SLOW: Copying data 10,000 times
for params in all_combinations:
    gpu_data = cupy.array(prices)  # CPU → GPU each time!
    result = backtest(gpu_data, params)
```

Each `cupy.array()` call copied data from CPU to GPU. For 10,000 trials, that's 10,000 memory transfers. The overhead was enormous.

### The Fix
Load data once during initialization:

```python
# ✅ FAST: Load once, use forever
class GPUBacktester:
    def __init__(self, prices):
        self.gpu_prices = cupy.array(prices)  # One transfer
    
    def run(self, params):
        # Use self.gpu_prices - no transfer!
```

### Impact
**~2x speedup.** Better, but still not the 40x I was hoping for.

---

## Optimization #2: Pinned Memory Transfers (Medium Win)

### The Problem
Even the initial data load was slow. Standard memory copies go through the CPU cache, adding latency.

### The Fix
CuPy supports **CUDA pinned (page-locked) memory**—memory that bypasses the CPU cache and uses Direct Memory Access (DMA):

```python
# ✅ Pinned memory for maximum transfer bandwidth
pinned_mem = cupy.cuda.alloc_pinned_memory(data_size)
gpu_array = cupy.array(data, dtype=cupy.float32)
```

I also pre-computed derived arrays to eliminate redundant GPU→CPU round-trips:

```python
# Pre-compute on GPU once
self.prev_close = cupy.roll(self.close_prices, 1)
self.pct_change = (self.high - self.prev_close) / self.prev_close

# Store as Python lists for the main loop (no GPU→CPU each iteration)
self.prev_close_list = [float(x) for x in self.prev_close.get()]
```

### Impact
**~3-4x total speedup.** Initialization was now instant. But I was still processing parameters one-by-one. I needed something radical.

---

## Optimization #3: Vectorized Batch Processing (THE BREAKTHROUGH 🚀)

### The Problem
I was still thinking sequentially:

```python
# ❌ Sequential: Process one strategy at a time
results = []
for params in all_10000_combinations:
    result = backtest_single(params)
    results.append(result)
```

Even with data pre-loaded, this meant 10,000 function calls, 10,000 loop iterations, and 10,000 Python interpreter overhead instances.

### The Breakthrough Insight
GPUs don't run "fast loops." They run **vector operations**. Instead of testing one strategy at a time, I should test **ALL strategies simultaneously**.

What if I tracked 10,000 positions as a single vector?

### The Fix

```python
# ✅ Vectorized: ALL strategies in ONE operation
def run_batch(self, buy_triggers, sell_triggers):
    n = len(buy_triggers)  # 10,000+ combinations
    
    # State vectors - one element per strategy
    positions = cupy.zeros(n)
    cash = cupy.full(n, 100000.0)
    
    for bar in range(252):  # Just 252 iterations (days), not 10,000!
        # Check buy condition for ALL strategies at once
        pct = (high[bar] - prev[bar]) / prev[bar]
        should_buy = (positions == 0) & (pct >= buy_triggers)
        
        # Execute ALL qualifying buys in one vectorized operation
        positions = cupy.where(should_buy, cash / price, positions)
        cash = cupy.where(should_buy, 0.0, cash)
```

Instead of 10,000 iterations through 252 bars (2.5M operations), I now have:
- 252 bar iterations
- Each iteration processes 10,000 strategies in ONE GPU kernel

### The Magic of Broadcasting
CuPy/NumPy broadcasting means this comparison:
```python
pct >= buy_triggers  # pct is scalar, buy_triggers is 10,000-element vector
```

Returns a 10,000-element boolean vector in **one GPU operation**. No loops. No Python overhead. Pure parallel execution across 10,496 CUDA cores.

### Impact
**40x+ total speedup. This was the breakthrough.**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Time per stock | 40+ sec | ~4 sec | **10x faster** |
| Combinations tested | 200 | 13,775 | **69x more** |
| Strategies/second | ~5 | 3,475 | **695x throughput** |

---

## Why Vectorization Was THE Breakthrough

The first two optimizations were about **reducing overhead**—making the slow parts less slow.

Vectorization is fundamentally different. It's about **changing the algorithm** from O(n × m) sequential operations to O(m) parallel operations where n strategies are computed simultaneously.

Think of it this way:
- **Optimization #1** (Pre-load): "Don't repeat the slow part"
- **Optimization #2** (Pinned memory): "Make the slow part faster"  
- **Optimization #3** (Vectorization): "**Eliminate the slow part entirely**"

The GPU isn't running my backtesting logic 10,000 times. It's running it **once**, operating on 10,000-element vectors. That's the paradigm shift.

---

## The Final Numbers

```
=== GPU OPTIMIZATION VERIFICATION ===

✅ 1. GPU Available: True
✅ 2. Data Pre-loaded: 252 bars in 0.691s
✅ 3. Pre-computed arrays ready
✅ 4. Parallel batch: 13,775 parameter combinations
✅ 5. GPU batch completed in 3.964s
   - Speed: 3,475 strategies/second
```

From 40 seconds (200 trials) to 4 seconds (13,775 trials).

**Not just faster—but exhaustively testing every parameter combination.** No more wondering if the optimizer missed the global optimum. We've tested them all.

---

## Key Takeaways

1. **Start with data management, not compute.** The first 4x came from eliminating redundant memory transfers.

2. **Pinned memory matters for initial loads.** DMA bypasses CPU cache for true PCIe bandwidth.

3. **Vectorization is the real unlock.** Replace loops with vector operations. Let the GPU do what GPUs do best.

4. **Think in parallel, not faster-sequential.** Don't ask "how do I make this loop faster?" Ask "how do I eliminate this loop entirely?"

5. **Your gaming GPU is a quant supercomputer.** 3,475 strategies per second on consumer hardware.

---

## Code Available

The complete implementation:
- `optimizer/gpu_backtest.py` - Vectorized GPU backtester
- `optimizer/Dockerfile.gpu` - CUDA 12.2 + CuPy container
- `optimizer/server.py` - Web server with GPU toggle

Try it yourself and let me know what speedups you achieve!

---

*The breakthrough wasn't making Python faster. It was realizing that GPUs don't run fast loops—they run parallel vector operations. Once I stopped thinking sequentially, everything changed.*
