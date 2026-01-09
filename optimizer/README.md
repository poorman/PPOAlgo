# 🚀 Stock Parameter Optimizer

GPU-accelerated parameter optimization for finding optimal buy/sell triggers using advanced VWAP-based momentum strategies.

---

## PPOAlgo vs. Optimizer: What's the Difference?

This **Optimizer** is a specialized evolution that I developed based on learnings from the original PPOAlgo project. Here's how they differ:

| Aspect | PPOAlgo (Original) | Optimizer (This Tool) |
|--------|-------------------|----------------------|
| **Primary Purpose** | Reinforcement learning for trading | Parameter optimization for strategies |
| **Core Algorithm** | PPO (Proximal Policy Optimization) | Bayesian + GPU Grid Search |
| **Strategy Type** | AI learns patterns autonomously | Human-defined rules, AI optimizes parameters |
| **Entry Logic** | Neural network decisions | **VWAP-based momentum conditions** |
| **Training Time** | Hours/Days | Seconds (GPU) |
| **Explainability** | Black box | Fully transparent (see why each trade was skipped) |
| **Data Needs** | Massive historical data | Works with 1 year of data |

### The Evolution Story

1. **PPOAlgo Started As:** A reinforcement learning experiment using OpenAI's PPO algorithm to learn trading patterns from raw price data. While fascinating, it required massive GPU resources, days of training, and produced "black box" decisions.

2. **The Learning:** Through PPOAlgo development, I discovered that simpler rule-based strategies with optimized parameters often outperform complex ML models when:
   - Markets are efficient (hard to find arbitrage)
   - Explainability matters (you need to trust your system)
   - Quick iteration is needed (test ideas in seconds, not days)

3. **Optimizer Was Born:** I pivoted to a parameter optimization approach where:
   - **Humans define the strategy** (VWAP conditions, momentum filters)
   - **GPU optimizes the parameters** (entry %, exit %, stop-loss levels)
   - **Results are explainable** (see exactly why each trade was taken or skipped)

### Why VWAP-Based Momentum?

The Optimizer uses an **Adaptive VWAP Strategy** that I developed and refined:

```
Entry Conditions (ALL must pass):
1. price_10am > VWAP         → Price above institutional benchmark
2. vwap_slope > 0            → VWAP trending upward
3. stretch < 0.5 × or_vol    → Not overextended
4. momentum > 0.25 × or_vol  → Sufficient buying pressure
```

This approach combines:
- **VWAP** – Used by institutional traders as a benchmark
- **Slope analysis** – Confirms trend direction
- **Stretch control** – Prevents buying at tops
- **Momentum confirmation** – Ensures real buying pressure

**Key Insight:** By using VWAP as an anchor, we filter for setups where institutional money is likely flowing in the same direction—dramatically improving win rates.

---

## Features

| Feature | Description |
|---------|-------------|
| **🧠 Bayesian Optimization** | Uses Optuna's TPE sampler for intelligent parameter search |
| **📊 VWAP Momentum Strategy** | Advanced 4-condition adaptive entry system |
| **🔥 GPU Acceleration** | RTX 3090 support via CuPy for 100,000+ parallel backtests |
| **📈 Real-time Progress** | Live WebSocket progress bar with ETA |
| **💾 PostgreSQL History** | All results saved with full trade logs |
| **🔍 Trade Log Analysis** | See exactly why each trade was taken or skipped |
| **📡 Multi-API Support** | Alpaca, Widesurf, Massive.com data sources |
| **🗑️ Delete** | Remove unwanted history entries |

---

## Trading Algorithms

### 1. Dipper Strategy
Buy on dips below previous close, sell on profit target.

### 2. 10AM Momentum
Buy at 10AM if price is up from open, sell at target.

### 3. Adaptive VWAP ⭐
Our most sophisticated strategy using four momentum conditions:

| Condition | Formula | Purpose |
|-----------|---------|---------|
| **Price > VWAP** | `price_10am > vwap` | Bullish positioning |
| **VWAP Slope > 0** | `(vwap_10am - vwap_940) / vwap_940` | Trend confirmation |
| **Stretch < 0.5×σ** | `(price - vwap) / vwap < threshold` | Not overextended |
| **Momentum > 0.25×σ** | `(price_10am - open) / open` | Buying pressure |

The trade log shows skip reasons like:
- `$ < VWAP` – Price below VWAP (bearish)
- `slope↓ (-0.15%)` – VWAP trending down
- `stretch↑ (0.8%)` – Too extended above VWAP
- `mom↓ (0.1%)` – Insufficient momentum

---

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

For maximum performance:

### Prerequisites
- NVIDIA Drivers installed on host
- NVIDIA Container Toolkit:
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

---

## Performance Comparison

| Mode | Combinations | Time | Speedup |
|------|--------------|------|---------|
| **CPU (Optuna)** | 200 trials | ~40 sec | 1x |
| **GPU (Grid Search)** | 10,000 | ~2 sec | **20x** |
| **GPU (Grid Search)** | 100,000 | ~10 sec | **200x** |

---

## Data Sources

The Optimizer supports multiple market data APIs:

| API | Description | Auth Method |
|-----|-------------|-------------|
| **Alpaca** | Default, via PPOALGO_API | Internal |
| **Widesurf** | Alternative provider | X-API-KEY header |
| **Massive.com** | Institutional data | apiKey query param |

Switch between sources using the dropdown in the UI. The trade log header shows which API was used for each run.

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
  "use_gpu": true,
  "algo": "chatgpt_vwap",
  "data_source": "alpaca"
}
```

### GET `/api/history`
Get optimization history with full trade logs.

### GET `/api/price-compare`
Compare price data across all 3 APIs (Alpaca, Widesurf, Massive.com).

### DELETE `/api/history/{id}`
Delete a specific history entry.

---

## Files

| File | Description |
|------|-------------|
| `server.py` | FastAPI WebSocket server with all algorithms |
| `gpu_backtester.py` | CuPy GPU-accelerated batch backtester |
| `config.py` | API keys and database configuration |
| `database.py` | PostgreSQL connection management |
| `routers/history.py` | History API endpoints |
| `routers/api_tester.py` | Multi-API price comparison tool |
| `static/index.html` | Optimizer GUI |
| `Dockerfile` | CPU container |
| `Dockerfile.gpu` | GPU container (CUDA 12.2) |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PPOALGO_API` | Backend API URL (e.g., `http://ppoalgo_api_1:8000`) |
| `POSTGRES_HOST` | PostgreSQL host |
| `POSTGRES_DB` | Database name (default: `ppoalgo`) |
| `POSTGRES_USER` | Database user (default: `ppoalgo`) |
| `POSTGRES_PASSWORD` | Database password |
| `WIDESURF_API_KEY` | Widesurf.com API key |
| `MASSIVE_API_KEY` | Massive.com API key |

---

## Troubleshooting

### "CuPy not available"
GPU container failed to load CuPy. Check NVIDIA drivers and container toolkit.

### WebSocket disconnects
Increase timeout in your proxy/firewall for long-running optimizations.

### History not showing
Click "Refresh History" or check database connection.

### Trade log shows "--" for profit
When a trade wasn't taken, the profit column shows potential profit in gray (what you would have made).

---

## The Journey: From PPO to VWAP

This optimizer represents a philosophical shift in algorithmic trading:

> **"Instead of training a black-box AI to make decisions, give humans the strategy and let AI find the optimal parameters."**

What I learned from PPOAlgo:
1. RL algorithms are powerful but require massive compute and data
2. Explainability matters more than marginal performance gains
3. Simple strategies with optimized parameters often beat complex ML
4. Fast iteration (seconds vs days) leads to better final strategies

The Optimizer embodies these learnings—combining the computational power of GPU acceleration with the transparency of rule-based trading.

---

**Built with learnings from PPOAlgo, optimized for practical trading.**
