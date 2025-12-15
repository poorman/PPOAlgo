# 🚀 Smart Stock Parameter Optimizer

GPU-accelerated Bayesian optimization to find the best buy/sell triggers for any stock.

## Features

| Feature | Description |
|---------|-------------|
| **🧠 Bayesian Optimization** | Uses Optuna's TPE sampler for intelligent parameter search |
| **📊 Volatility Analysis** | Analyzes each stock to set smart search bounds |
| **🔥 GPU Acceleration** | RTX 3090 support via CuPy for 10,000+ parallel backtests |
| **📈 Real-time Progress** | Live WebSocket progress bar with ETA |
| **💾 PostgreSQL History** | All results saved with full history |
| **🔍 Search & Sort** | Filter by symbol, sort by Date/Sharpe/Max Gain |
| **🗑️ Delete** | Remove unwanted history entries |
| **⏱️ Duration** | Track how long each stock took to optimize |

## Quick Start

### CPU Mode (Standard)

```bash
cd optimizer
docker build -t ppoalgo-optimizer .
docker run -d --name ppoalgo_optimizer \
  --network ppoalgo_default -p 8082:8000 \
  -e PPOALGO_API=http://ppoalgo_api_1:8000 \
  -e POSTGRES_HOST=ppoalgo_db_1 \
  ppoalgo-optimizer
```

Access at: **http://localhost:8082**

---

## 🔥 GPU Mode (RTX 3090)

For maximum performance with your RTX 3090:

### Prerequisites
- NVIDIA Drivers installed on host
- NVIDIA Container Toolkit
  - Windows: Docker Desktop (WSL 2 backend with GPU)
  - Linux: `sudo apt-get install -y nvidia-container-toolkit`

### Build GPU Image

```bash
cd optimizer
docker build -f Dockerfile.gpu -t ppoalgo-optimizer-gpu .
```

### Run with GPU

```bash
# Stop CPU version
docker stop ppoalgo_optimizer
docker rm ppoalgo_optimizer

# Run GPU version
docker run -d \
  --name ppoalgo_optimizer \
  --gpus all \
  --network ppoalgo_default \
  -p 8082:8000 \
  -e PPOALGO_API=http://ppoalgo_api_1:8000 \
  -e POSTGRES_HOST=ppoalgo_db_1 \
  ppoalgo-optimizer-gpu
```

### Verify GPU

```bash
docker logs ppoalgo_optimizer | grep -i gpu
```

Expected: `INFO:root:CuPy loaded - GPU acceleration enabled`

### Usage

1. Go to http://localhost:8082
2. Check **"Available GPU: RTX 3090"** box
3. Click **Start Optimization**
4. Watch 10,000+ parameter combinations tested in seconds!

---

## Performance Comparison

| Mode | Combinations | Time (1 stock) |
|------|--------------|----------------|
| **CPU (Optuna)** | 200 trials | ~40 seconds |
| **GPU (Grid Search)** | 10,000+ | ~2 seconds |

---

## GPU Optimizations

The GPU backtester includes several optimizations:

1. **Vectorized Operations** - All strategies computed in parallel using matrix ops
2. **Pinned Memory** - Fast CPU→GPU data transfer using CUDA pinned memory
3. **Pre-computed Arrays** - Price data extracted once, reused for all combinations
4. **Chunked Processing** - Large grids (100K+) processed in memory-safe chunks
5. **No Python Loops** - Backtest logic uses pure CuPy/NumPy vectorization

---

## API Endpoints

### WebSocket `/ws`
Real-time optimization progress updates.

### POST `/api/optimize`
Start optimization job.

```json
{
  "symbols": ["TSLA", "NVDA"],
  "start_date": "2024-01-01",
  "end_date": "2024-12-01",
  "capital": 100000,
  "n_trials": 200,
  "optimization_metric": "sharpe",
  "use_gpu": true
}
```

### GET `/api/history`
Get optimization history (default limit: 500).

### DELETE `/api/history/{id}`
Delete a specific history entry.

---

## Files

| File | Description |
|------|-------------|
| `server.py` | FastAPI WebSocket server |
| `gpu_backtest.py` | CuPy GPU-accelerated batch backtester |
| `stock_optimizer.py` | Standalone CLI optimizer |
| `static/index.html` | Optimizer GUI |
| `Dockerfile` | CPU container |
| `Dockerfile.gpu` | GPU container (CUDA 12.2) |

---

## Troubleshooting

### "CuPy not available"
GPU container failed to load CuPy. Check NVIDIA drivers and container toolkit.

### WebSocket disconnects
Increase timeout in your proxy/firewall for long-running optimizations.

### History not showing
Click "Refresh History" or check database connection.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PPOALGO_API` | Backend API URL (e.g., `http://ppoalgo_api_1:8000`) |
| `POSTGRES_HOST` | PostgreSQL host |
| `POSTGRES_DB` | Database name (default: `ppoalgo`) |
| `POSTGRES_USER` | Database user (default: `ppoalgo`) |
| `POSTGRES_PASSWORD` | Database password |
