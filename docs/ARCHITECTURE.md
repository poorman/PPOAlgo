# PPOAlgo Architecture

## System Overview

PPOAlgo is a GPU-accelerated trading strategy backtesting and optimization platform. It finds optimal buy/sell trigger parameters for day-trading strategies using exhaustive grid search.

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────────────┐
│   Frontend      │────▶│   Backend API    │────▶│   Market Data APIs        │
│   (Nginx:8081)  │     │  (FastAPI:8010)  │     │  (Alpaca/Polygon/Widesurf)│
│   HTML/JS UI    │     │  Price fetching   │     └───────────────────────────┘
└────────┬────────┘     └──────────────────┘
         │ WebSocket
         ▼
┌─────────────────────────────────────────────┐
│            Optimizer (FastAPI:8082)          │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │  Rust Engine │  │   GPU Engine (CuPy)  │  │
│  │  (32 cores)  │  │   (RTX 3080/3090)    │  │
│  │  Rayon par.  │  │   Vectorized batch   │  │
│  └─────────────┘  └──────────────────────┘  │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │  PostgreSQL (port 5433)              │   │
│  │  Optimization history & results      │   │
│  └──────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

## Service Stack

| Service | Port | Technology | Role |
|---------|------|------------|------|
| Frontend | 8081 | Nginx + HTML/JS | User interface, client-side backtesting |
| Backend API | 8010 | FastAPI (Python) | Price data fetching from market APIs |
| Optimizer | 8082 | FastAPI + WebSocket | Optimization engine, trade log generation |
| Database | 5433 | PostgreSQL | Persistent result storage |

## Execution Engines

### Rust Engine (CPU)
- Location: `optimizer/rust_vwap/src/main.rs`
- Parallelism: Rayon (auto-detects cores, typically 32 on i9-13900K)
- Communication: subprocess stdin/stdout JSON
- Use case: Default when GPU unavailable, 1 thread per stock in batch mode

### GPU Engine (CuPy)
- Location: `optimizer/gpu_backtest.py`
- Parallelism: Vectorized operations across all parameter combos simultaneously
- Use case: When CUDA GPU available, single-stock deep optimization

### Frontend Engine (JavaScript)
- Location: `frontend/app.js`
- Use case: Client-side backtesting with "Apply These Settings" button
- Mirrors optimizer logic for instant feedback

## Trading Strategies

| Strategy | Entry Signal | Exit Signal |
|----------|-------------|-------------|
| VWAP Rust | 5-condition VWAP gate + buy trigger | Take-profit or close |
| Dipper | High >= prev_close * (1 + buy%) | Target price or close |
| ChatGPT 9AM | 10AM price >= open * (1 + buy%) | Target price or close |

## Docker Topology

```yaml
services:
  frontend:  # Nginx serving static HTML/JS
  api:       # FastAPI price data server
  optimizer: # GPU/CPU optimization engine
  db:        # PostgreSQL 15
```

All services share the `ppoalgo_default` Docker network. The optimizer reaches the API via `http://api:8000` (internal hostname).
