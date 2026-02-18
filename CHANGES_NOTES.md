# PPOAlgo — Recent Changes Notes (Feb 11, 2026)

> Generated on 2026-02-12 by reviewing the last 10 git commits (all from Feb 11, 2026).

---

## Summary

The entire day's work was focused on **performance optimization** and **GPU compatibility** for the stock optimizer service. The changes progressed through several iterations of tuning concurrency and ultimately culminated in a major commit that optimized GPU utilization, database connection pooling, and data fetching efficiency.

---

## Commit Timeline (oldest → newest)

| Commit | Time | Summary |
|--------|------|---------|
| `a0fa2d9` | 00:27 | Added **Rust VWAP algorithm** ("Momentum 10am + VWAP (Rust 🦀)") |
| `17181b1` | 00:37 | Updated README with Rust VWAP algo docs |
| `7e7ff27` | 00:54 | Forced Rust algo to use **Parallel Grid Search** instead of slow Optuna trials |
| `e5e1967` | 01:00 | Parallelized stock optimization loop (16 concurrent) for better data fetching throughput |
| `83e02b7` | 01:05 | Increased concurrency to 12 + thread pools to keep UI responsive |
| `2abb137` | 01:04 | Lowered concurrency back to 4 to prevent event loop starvation |
| `c5d6629` | 01:24 | Scaled to 30 workers for multi-process UI responsiveness |
| `7270989` | 01:26 | Maximized i9-13900K — 30-stock parallel processing (1 core/stock) |
| `cf2e137` | 01:26 | Cleanup |
| **`a5fbd6b`** | **14:42** | **Major commit** — GPU utilization, DB connection pooling, CUDA 11.8 downgrade, GPU reservations |

---

## Key Changes in Detail

### 1. 🚀 GPU & CUDA Compatibility (`Dockerfile.gpu`, `docker-compose.yml`)

- **CUDA downgraded from 12.2.2 → 11.8.0** for maximum host driver compatibility (especially WSL2).
- Changed `cupy-cuda12x` → `cupy-cuda11x` to match CUDA version.
- Added `tzdata` dependency for timezone support.
- Reduced uvicorn from **4 workers → 1 worker** (single worker manages all internal parallelism via thread pools).
- Added **GPU hardware reservations** in `docker-compose.yml` (`deploy.resources.reservations.devices` with `nvidia` driver).
- Optimizer now builds from `Dockerfile.gpu` explicitly (previously used plain `Dockerfile`).
- Startup command changed from direct `uvicorn` to `/bin/bash /app/start.sh` for GPU warmup flow.

### 2. 🗄️ Database Connection Pooling (`database.py`)

- **Upgraded from single connections to `ThreadedConnectionPool`** (5–200 connections) for thread safety with `ThreadPoolExecutor`.
- New functions: `init_db_pool()`, `release_db_conn()`.
- All `conn.close()` calls replaced with `release_db_conn(conn)` to return connections to pool.
- DB logic **moved from `server.py` into `database.py`** (big refactoring — removed ~500 lines of duplicated DB code from server).

### 3. 🔧 Server Refactoring (`server.py` — 972 lines changed)

- Massive restructuring: database operations extracted to `database.py`.
- Progress counting and job tracking improvements.
- UI feedback for GPU vs CPU engine dispatching.
- **`generate_trade_log` optimized** to reuse enriched price data instead of redundant re-downloads.

### 4. 📊 GPU Backtesting (`gpu_backtest.py`)

- Improved CuPy initialization: now checks `cp.cuda.runtime.getDeviceCount()` before claiming GPU is available.
- Better fallback messaging when CUDA devices aren't found vs CuPy not installed.
- Fixed typo: `RTX 3089` → `RTX 3080` in comments.
- Added **`GPUChatGPTVWAPBacktester`** class — full GPU-accelerated backtester for the ChatGPT VWAP Strategy:
  - Entry at 10 AM if price ≥ open × (1 + entry_trigger) AND price > VWAP.
  - Take profit / stop loss / protect winner logic.
  - Vectorized grid search across parameter combinations.

### 5. 🔑 Config Updates (`config.py`)

- Updated **Widesurf API key** and **URL** (port changed from `8020` → `1020`).

### 6. 📦 New Files

- `optimizer/import_history.py` — 158 lines, new history import module.
- `optimizer/start.sh` — GPU warmup startup script (checks `nvidia-smi`, runs `gpu_warmup.py`, then starts uvicorn).
- `.dockerignore` — New file to reduce Docker build context.

### 7. 🖥️ Frontend (`optimizer/static/index.html`)

- 118 lines of UI changes — likely GPU/CPU feedback indicators and layout improvements.

---

## Current Docker Status

| Service | Status |
|---------|--------|
| `api` | ✅ Up (port 8010) |
| `db` | ✅ Up (port 5433) |
| `frontend` | ❌ Exit 128 |
| `optimizer` | ✅ Up (port 8082) |

**Note:** Frontend is crashed (Exit 128). This will be addressed with the rebuild.

---

## Architecture

```
docker-compose.yml
├── api          — FastAPI backend (uvicorn, port 8010→8000)
├── frontend     — Nginx serving static HTML (port 8083→80)
├── optimizer    — GPU-enabled optimizer (Dockerfile.gpu, CUDA 11.8, port 8082→8000)
└── db           — PostgreSQL 15 (port 5433→5432)
```
