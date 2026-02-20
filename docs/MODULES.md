# Module Catalog

## Frontend (`frontend/`)

| File | Purpose |
|------|---------|
| `index.html` | Main UI: configuration panel, results display, history sidebar |
| `app.js` | Client-side backtester, WebSocket communication, chart rendering |
| `style.css` | Dark theme styling |
| `nginx.conf` | Nginx reverse proxy configuration |

## Backend API (`backend/`)

| File | Purpose |
|------|---------|
| `server.py` | FastAPI server for fetching price data from Alpaca/Polygon/Widesurf APIs |

## Optimizer (`optimizer/`)

### Core Engine

| File | Purpose |
|------|---------|
| `server.py` | Main optimization server (~3400 lines). Handles WebSocket connections, orchestrates optimization jobs, generates trade logs, manages batch processing |
| `gpu_backtest.py` | GPU-accelerated backtesters using CuPy/NumPy (~1700 lines). Contains `GPUBatchBacktester` (Dipper), `GPUChatGPT9AMBacktester`, `GPUVWAPRustBacktester` |
| `rust_optimizer.py` | Bridge to Rust binary — sends JSON via stdin, reads results from stdout |
| `config.py` | Environment configuration (API keys, database URLs) |

### Database

| File | Purpose |
|------|---------|
| `database.py` | PostgreSQL connection pooling (`ThreadedConnectionPool`) |
| `models.py` | SQLAlchemy ORM models for optimization results |

### Routers

| File | Purpose |
|------|---------|
| `routers/history.py` | REST endpoints for browsing/filtering optimization history |

### Services

| File | Purpose |
|------|---------|
| `services/backtester.py` | Service-layer backtesting utilities |
| `services/price_cache.py` | Price data caching layer |

### Rust Engine (`optimizer/rust_vwap/`)

| File | Purpose |
|------|---------|
| `src/main.rs` | Rayon-parallel VWAP grid search. Reads JSON from stdin, outputs best params to stdout |
| `Cargo.toml` | Rust dependencies (rayon, serde, serde_json) |

## GPU Backtester Classes

| Class | Strategy | Grid Dimensions |
|-------|----------|----------------|
| `GPUBatchBacktester` | Dipper (breakout) | buy_trigger x sell_trigger |
| `GPUChatGPT9AMBacktester` | ChatGPT 9AM | buy_trigger x sell_trigger |
| `GPUVWAPRustBacktester` | VWAP Momentum 10am | buy_trigger x sell_trigger |
