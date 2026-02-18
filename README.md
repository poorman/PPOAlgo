# PPOAlgo - Smart Prop Firm Trading Console

🔗 **[Live Demo](https://pbieda.com/projects/ppoalgo-momentum-algorithm-using-rtx-3090)**

A comprehensive trading backtesting platform with **AI-powered parameter optimization** using Bayesian optimization (Optuna) and GPU acceleration. Supports multiple algorithmic strategies including PPO, LSTM, and advanced Momentum-based day trading with VWAP analysis.

## Screenshots

### Dashboard Console
![Dashboard - Model Configuration, Charts, and Performance Metrics](images/dashboard.jpg)

### Trade Log
![Trade Log - Chronological trade history with detailed metrics](images/trade_log.jpg)

## Stock Parameter Optimizer

The **Stock Parameter Optimizer** uses Bayesian optimization and GPU-accelerated grid search to automatically find the best buy/sell triggers for each stock based on its unique volatility characteristics.

### Optimizer Features

| Feature | Description |
|---------|-------------|
| **Bayesian Optimization** | Uses Optuna's TPE sampler for intelligent parameter search |
| **Volatility Analysis** | Analyzes each stock's volatility to set smart search bounds |
| **GPU Acceleration** | RTX 3080/3090 support via CuPy for massive parallel grid search |
| **Rust Rayon Engine** | Native Rust binary with Rayon parallelism for CPU grid search |
| **Real-time Progress** | Live WebSocket updates with progress bar and ETA |
| **PostgreSQL History** | All results saved to database with full history |
| **Search & Sort** | Filter history by symbol, sort by Date/Sharpe/Max Gain/Win Rate |
| **Duration Tracking** | Per-job wall-clock timing with `completed_at` timestamps |
| **Run Logs Tab** | Persistent job history with live session log and expandable per-stock details |

### Trading Algorithms

| Algorithm | Description |
|-----------|-------------|
| **Momentum (Default)** | Day-trading strategy with configurable buy/sell triggers |
| **ChatGPT 10am** | Buys at 10 AM if price is above previous close trigger |
| **ChatGPT Stop Loss** | ChatGPT 10am variant with configurable stop-loss |
| **Momentum 10am + VWAP** | VWAP-weighted strategy using 10 AM and 9:40 AM price anchors |
| **Momentum 10am + VWAP (Rust)** | Same as above, grid search executed via native Rust/Rayon binary |

### Data Sources

| Source | Description |
|--------|-------------|
| **Widesurf API V1** | Self-hosted Go market data server (port 8020) |
| **Widesurf API V2** | High-throughput Go server with 20 concurrent connections (port 8090) |
| **Alpaca API** | Cloud-based market data (paper trading) |

### Performance Architecture

The optimizer uses a hybrid GPU/CPU worker pipeline:

```
Phase 1: Pre-fetch all price data (batched, connection-pooled)
    └── Widesurf V2: 40 stocks/batch, 20 concurrent connections
    └── 3 API calls per VWAP stock (daily + 10AM + 9:40AM) — parallel for V2

Phase 2: Optimize each stock
    ├── GPU Workers (2x): 40% of stocks → CuPy vectorized grid search
    └── CPU Workers (30x): 60% of stocks → Rust Rayon grid search (1 thread each)
    └── Trade log generation uses pre-enriched bars (no re-download)

Phase 3: Save results to PostgreSQL
```

| Metric | CPU Only (Optuna) | GPU + Rust Hybrid |
|--------|-------------------|-------------------|
| Combinations tested | 200 | 1,305+ per stock |
| Time per stock | ~40s | ~1-2s |
| 372 stocks total | Hours | ~2-3 min |
| Strategies/second | ~5 | ~3,000+ |

---

## Trading Models

| Model | Description |
|-------|-------------|
| **PPO** | Proximal Policy Optimization reinforcement learning algorithm |
| **LSTM** | Long Short-Term Memory neural network predictor |
| **Baseline** | Simple rule-based trading strategy |
| **Momentum** | Day-trading momentum strategy with configurable triggers |

## Momentum Strategy

The Momentum model is a day-trading strategy:

### Buy Logic
- **Buy Trigger %**: Enters position when stock's high reaches X% above previous close
- **Buy Time**: Configurable entry time (default: 09:00 AM CDT)

### Sell Logic
- **Sell Trigger %**: Exits when position gains X% from entry (intraday target)
- **End of Day Exit**: If target isn't reached, exits at 15:59

### Compound Investing
When enabled, proceeds from each sell become the buy amount for the next trade, compounding gains over time.

## Architecture

```
PPOAlgo/
├── frontend/
│   ├── index.html          # Main backtester UI
│   └── nginx.conf          # Nginx config for static serving
├── backend/
│   └── server.py           # FastAPI price data API
├── optimizer/
│   ├── server.py           # Optimizer WebSocket server (main engine)
│   ├── config.py           # API keys, DB config, GPU flags
│   ├── database.py         # PostgreSQL connection pool + queries
│   ├── gpu_backtest.py     # CuPy GPU backtester (grid search)
│   ├── gpu_backtest_enhanced.py  # Enhanced GPU backtester with VWAP
│   ├── rust_optimizer.py   # Rust/Rayon subprocess bridge
│   ├── routers/
│   │   ├── history.py      # /api/history + /api/jobs endpoints
│   │   └── ai100.py        # AI-100 stock list endpoint
│   ├── static/
│   │   └── index.html      # Optimizer GUI (single-page app)
│   ├── Dockerfile.turbo    # GPU-enabled container (CUDA 11.8)
│   └── start_optimized.sh  # Container entrypoint
├── docker-compose.yml       # Standard stack
├── docker-compose.turbo.yml # GPU-optimized stack
└── README.md
```

## Running the Application

### Docker Compose (Standard)

```bash
cd PPOAlgo
docker-compose up -d
```

### Docker Compose (GPU Turbo)

```bash
cd PPOAlgo
docker-compose -f docker-compose.turbo.yml up -d
```

### Access Points

| Service | URL |
|---------|-----|
| **Optimizer GUI** | http://localhost:8082 |
| **Frontend** | http://localhost:8081 |
| **Backend API** | http://localhost:8010 |
| **Database** | localhost:5433 |

## API Endpoints

### Optimizer Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `WS` | `/ws` | WebSocket for real-time optimization updates |
| `GET` | `/api/history` | All optimization results with metrics |
| `GET` | `/api/history/{id}` | Single result with full trade log |
| `GET` | `/api/history/count` | Total result count |
| `DELETE` | `/api/history` | Delete all results |
| `GET` | `/api/jobs` | Job-level run logs with aggregated metrics |
| `GET` | `/api/jobs/{job_id}` | Per-symbol breakdown for a job |
| `DELETE` | `/api/jobs` | Clear all job logs |
| `GET` | `/api/keywords` | Keyword stock list configs |
| `GET` | `/api/keys` | API key configuration |

### Backend API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/prices` | Historical OHLC data (Polygon/Alpaca) |
| `POST` | `/api/backtest` | Run single backtest |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ALPACA_API_KEY_ID` | Alpaca API key |
| `ALPACA_API_SECRET_KEY` | Alpaca secret key |
| `ALPACA_BASE_URL` | Alpaca base endpoint |
| `WIDESURF_API_KEY` | Widesurf V1 API key |
| `WIDESURF_API_URL` | Widesurf V1 endpoint (default: port 8020) |
| `WIDESURF_V2_API_KEY` | Widesurf V2 API key |
| `WIDESURF_V2_API_URL` | Widesurf V2 endpoint (default: port 8090) |
| `POSTGRES_HOST` | Database host |
| `POSTGRES_DB` | Database name |
| `POSTGRES_USER` | Database user |
| `POSTGRES_PASSWORD` | Database password |
| `PPOALGO_API` | Backend API URL for optimizer |

## GPU Acceleration

### Performance Benchmarks (RTX 3080/3090)

| Metric | CPU (Optuna) | GPU (Grid Search) | Speedup |
|--------|-------------|-------------------|---------|
| Combinations tested | 200 | 13,775 | 69x more |
| Time per stock | ~40 seconds | <2 seconds | 20x faster |
| Strategies/second | ~5 | ~3,475 | 695x faster |

### Key Optimizations

1. **Pinned Memory Transfers** — CUDA page-locked memory for 2-3x faster CPU-to-GPU data movement
2. **Pre-loaded Data** — Price history loaded once to GPU memory, reused across all parameter combinations
3. **Vectorized Batch Processing** — All parameters evaluated in a single matrix operation via CuPy broadcasting
4. **Hybrid GPU/CPU Dispatch** — GPU handles 40% of stocks (grid search), 30 CPU workers handle 60% (Rust/Rayon)
5. **Connection-Pooled API Fetching** — `requests.Session` with HTTPAdapter for TCP connection reuse
6. **Batched Prefetch** — Widesurf V2 stocks fetched in batches of 40 with inter-batch pauses to prevent TCP exhaustion

## Recent Updates

### v3.0 - Widesurf V2 + Performance Overhaul (February 2026)
- Added Widesurf API V2 data source (Go server, port 8090, high concurrency)
- Fixed critical bug: GPU path's `generate_trade_log` missing `data_source` parameter (was defaulting to Alpaca, downloading 105K+ minute bars per stock instead of using cached Widesurf data)
- Separate V2 semaphore (20 concurrent connections) with batched prefetch (40 stocks/batch)
- Connection-pooled HTTP session (`requests.Session` + HTTPAdapter)
- Parallel prefetch for VWAP algorithms (3 API calls fire simultaneously per stock on V2)
- 2 GPU workers for pipeline overlap (grid search + trade log generation)
- Rebalanced GPU/CPU split from 60/40 to 40/60 for better throughput
- Added **Logs tab** with persistent job history, live session log, expandable per-stock details
- Added `completed_at` timestamp and wall-clock duration tracking per job
- Raised in-memory cache threshold from 1,000 to 2,000 entries (prevents mid-run cache eviction on 300+ stock runs)

### v2.1 - GPU Optimization (December 2024)
- CUDA 12.2 devel image for full CuPy support
- Pinned memory GPU transfers (2-3x faster loading)
- Pre-computed price arrays (eliminate redundant transfers)
- Vectorized batch backtesting (13,000+ strategies/batch)
- Chunked processing for 100K+ parameter grids
- Real-time GPU progress tracking

### v2.0 - Smart Optimizer (December 2024)
- Bayesian optimization with Optuna
- GPU acceleration (CuPy/CUDA)
- Real-time WebSocket progress
- PostgreSQL history persistence
- History search, sort, and delete
- Volatility-based parameter bounds

## License

MIT License
