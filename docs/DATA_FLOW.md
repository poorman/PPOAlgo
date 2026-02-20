# Data Flow

## 1. Optimization Pipeline

```
User clicks "Optimize"
        │
        ▼
Frontend (WebSocket) ──▶ Optimizer server.py
        │                       │
        │                 ┌─────┴──────┐
        │                 │ Fetch bars  │◀── Backend API (8010) ◀── Alpaca/Polygon/Widesurf
        │                 │ + VWAP data │
        │                 └─────┬──────┘
        │                       │
        │              ┌────────┴────────┐
        │              │  Hybrid Dispatch │
        │              └───┬─────────┬───┘
        │                  │         │
        │          ┌───────▼──┐  ┌───▼────────┐
        │          │ GPU Path │  │ Rust Path  │
        │          │ (CuPy)   │  │ (Rayon)    │
        │          └───────┬──┘  └───┬────────┘
        │                  │         │
        │              ┌───▼─────────▼───┐
        │              │  Best Params    │
        │              │  buy_trigger_pct│
        │              │  sell_trigger_pct│
        │              └───────┬─────────┘
        │                      │
        │              ┌───────▼─────────┐
        │              │ Trade Log Gen   │
        │              │ (server.py)     │
        │              └───────┬─────────┘
        │                      │
        │              ┌───────▼─────────┐
        │              │ Save to DB      │
        │              │ (PostgreSQL)    │
        │              └───────┬─────────┘
        │                      │
        ◀──────────────────────┘
  Results + Trade Log displayed
```

## 2. Price Data Flow

```
Market Data APIs
    │
    ├── Alpaca (1-min bars, VWAP built-in)
    ├── Polygon (aggregates, requires calculation)
    └── Widesurf V2 (exact-time API, pre-aggregated VWAP)
         │
         ▼
Backend API (server.py:8010)
    │
    ├── /api/prices ──▶ Daily OHLC bars
    └── (intraday)  ──▶ 1-min bars for VWAP calculation
         │
         ▼
Optimizer enrichment (server.py)
    │
    ├── price_10am: Close at 10:00 AM ET
    ├── vwap: VWAP from 9:30-10:00
    ├── vwap_940: VWAP from 9:30-9:40
    ├── or_high: Opening range high (9:30-10:00)
    └── or_low: Opening range low (9:30-10:00)
```

## 3. VWAP Entry Signal Evaluation

```
For each trading day:

  prev_close (yesterday's close)
       │
  ┌────▼─────────────────────────┐
  │ Condition 5: Buy Trigger     │
  │ price_10am >= prev_close     │
  │   * (1 + buy_trigger_pct)    │
  └────┬──────────────────┬──────┘
     PASS              FAIL → skip
       │
  ┌────▼─────────────────────────┐
  │ Condition 1: price > VWAP    │
  │ Condition 2: slope > 0       │
  │ Condition 3: stretch < α*vol │
  │ Condition 4: momentum > γ*vol│
  └────┬──────────────────┬──────┘
    ALL PASS           ANY FAIL → skip
       │
  ┌────▼──────┐
  │ BUY at    │
  │ price_10am│
  └────┬──────┘
       │
  ┌────▼──────────────────────────┐
  │ EXIT:                         │
  │  if high >= TP price → sell TP│
  │  else → sell at close         │
  └───────────────────────────────┘
```

## 4. Parameter Grid Search

```
buy_range:  0.1% to 3.0% (step 0.1%) → 29 values
sell_range: 1.0% to 10%  (step 0.2%) → 45 values
                                        ─────────
                              Total:   1,305 combos

Each combo runs full backtest → returns (total_return, sharpe, win_rate, trades)
Best combo selected by optimization metric (default: total_return)
```

## 5. Result Persistence

```
Optimization Result
    │
    ▼
PostgreSQL (ppoalgo:5433)
    │
    ├── symbol, algo, date range
    ├── best_params (buy_trigger, sell_trigger)
    ├── metrics (return, sharpe, win_rate, trades)
    └── metadata (engine, duration, combos tested)
    │
    ▼
History API (routers/history.py)
    │
    ▼
Frontend History Panel (sortable by date, sharpe, return, etc.)
```
