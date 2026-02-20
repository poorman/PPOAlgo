# Architectural Decisions Log

## ADR-001: Make buy_trigger a Functional Parameter in VWAP Rust Strategy

**Date:** 2026-02-20
**Status:** Implemented

### Context
When running batch optimizations for 300 stocks with the VWAP Rust algorithm, nearly all results showed `buy_trigger = 0.1%`. Investigation revealed that `buy_trigger_pct` was a dead parameter — it had zero effect on optimization results:

- **Rust binary**: `run_backtest()` only accepted `sell_trig`; `buy_trig` was captured but never passed
- **GPU backtester**: hardcoded `"buy_trigger_pct": 0.1` in results
- **Trade log**: VWAP path used only the 4 VWAP conditions, no buy_trigger gate

Meanwhile, the frontend JavaScript already applied buy_trigger as a pre-filter gate.

### Decision
Add buy_trigger as a 5th entry condition: `price_10am >= prev_close * (1 + buy_trigger_pct)`.

This creates a meaningful pre-filter that requires the stock to have gained at least X% from yesterday's close before the 4 VWAP conditions are evaluated. The optimizer now performs a 2D grid search (buy x sell) instead of 1D (sell only).

### Consequences
- Grid search goes from ~45 combinations to ~1,305 (29 buy x 45 sell)
- Still fast: <1s on GPU, <2s on Rust
- buy_trigger values now vary meaningfully across stocks
- Frontend, optimizer (GPU + Rust), and trade log all use the same 5-condition logic

### Files Modified
- `optimizer/rust_vwap/src/main.rs` — Added `buy_trigger_pct` param to `run_backtest()`
- `optimizer/gpu_backtest.py` — 2D grid search in `GPUVWAPRustBacktester`
- `optimizer/server.py` — Pass `buy_range` to GPU engine, add gate to trade log
- `optimizer/rust_optimizer.py` — Already passed `buy_range` (no change needed)

---

## ADR-002: Hybrid GPU/CPU Dispatch

**Date:** 2025 (pre-existing)
**Status:** Active

### Context
Not all deployment environments have NVIDIA GPUs. The optimizer needs to work on both GPU and CPU-only machines.

### Decision
Implement hybrid dispatch: if CuPy detects a CUDA GPU, use the GPU engine; otherwise, fall back to the Rust binary via subprocess. Both engines produce identical results.

### Consequences
- GPU path: ~3,475 strategies/sec (RTX 3090)
- CPU path: ~345 strategies/sec (32-core i9-13900K via Rayon)
- Both paths must be kept in sync when strategy logic changes

---

## ADR-003: VWAP 4-Condition Entry Signal

**Date:** 2025 (pre-existing)
**Status:** Active (now 5 conditions with ADR-001)

### Decision
Entry requires ALL conditions to be true:
1. `price_10am > vwap` — Price above VWAP
2. `vwap_slope > 0` — VWAP trending up (9:40→10:00)
3. `stretch < 0.50 * or_vol` — Not too extended from VWAP
4. `momentum > 0.25 * or_vol` — Sufficient momentum from open
5. `price_10am >= prev_close * (1 + buy_trigger%)` — Buy trigger gate (ADR-001)

Alpha (0.50) and Gamma (0.25) are hardcoded constants tuned empirically.
