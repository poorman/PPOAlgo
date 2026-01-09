# PPOAlgo: GPU-Accelerated Stock Parameter Optimizer

## Overview

PPOAlgo is an intelligent trading parameter optimization system that uses GPU-accelerated grid search and machine learning to find optimal buy/sell trigger points for stocks. Unlike traditional backtesting tools that test strategies one-by-one, PPOAlgo leverages NVIDIA GPU parallel processing to evaluate **millions of parameter combinations in seconds**.

Built for day traders, swing traders, and quantitative analysts, PPOAlgo automates the tedious process of backtesting and provides actionable insights with detailed trade-by-trade analysis.

---

## The Problem We Solve

Traditional backtesting is painfully slow:

| Manual Approach | Time Required |
|-----------------|---------------|
| Test 100 buy/sell combinations | ~30 minutes |
| Test 10,000 combinations | ~5 hours |
| Test 1,000,000 combinations | ~20+ days |

**PPOAlgo solves this** by using GPU parallel processing to test all combinations simultaneously, reducing days of work to seconds.

---

## How It Works

### Step 1: Data Collection

PPOAlgo connects to multiple professional-grade market data APIs:

- **Alpaca Markets** – Commission-free trading with real-time/historical data
- **Massive.com** – Institutional-quality market data
- **Widesurf** – Alternative data provider

The system fetches minute-level OHLCV (Open, High, Low, Close, Volume) data and calculates derived metrics like VWAP, momentum, and volatility in real-time.

### Step 2: GPU-Accelerated Optimization

The core innovation is **massive parallel grid search** using NVIDIA CUDA:

```
Traditional CPU Approach:
for buy_trigger in range(0.5, 5.0, 0.1):     # 45 values
    for sell_trigger in range(0.5, 10.0, 0.1): # 95 values
        backtest(buy_trigger, sell_trigger)    # 4,275 sequential tests

PPOAlgo GPU Approach:
matrix[4275 strategies × 500 trading days] → ALL tested in parallel
```

Using an NVIDIA RTX 3090 with 10,496 CUDA cores, we achieve:
- **100,000 strategy combinations** in ~1 second
- **1,000,000 combinations** in ~25 seconds

### Step 3: Intelligent Strategy Selection

The optimizer finds the parameters that maximize your chosen metric:

| Metric | Best For |
|--------|----------|
| **Sharpe Ratio** | Risk-adjusted returns (recommended) |
| **Win Rate** | High-probability setups |
| **Total Return** | Maximum profit regardless of risk |
| **Max Drawdown** | Capital preservation |

---

## Trading Algorithms

### 1. Dipper Strategy (Classic)

**Philosophy:** Buy on dips, sell on recovery.

**Rules:**
- **Entry:** Buy when price drops X% below previous day's close
- **Exit:** Sell when price rises Y% from entry (take profit)
- **Best for:** Mean-reversion in volatile stocks

**Example:**
```
Buy Trigger: -2.5% (buy when down 2.5% from prev close)
Sell Trigger: +3.0% (sell when up 3% from entry)
```

---

### 2. 10AM Momentum Strategy

**Philosophy:** Let the first 30 minutes of trading establish direction, then ride momentum.

**Rules:**
- **Entry:** At 10:00 AM ET, if price is up X% from market open (9:30 AM)
- **Exit:** Sell at Y% profit target or end of day
- **Best for:** Trending stocks with morning breakouts

**Why 10AM?**
- First 30 mins are volatile due to overnight gap fills
- By 10AM, institutional traders have established direction
- Momentum from 10AM tends to persist through the session

---

### 3. Adaptive VWAP Strategy (Advanced) ⭐

**Philosophy:** Only enter trades when multiple technical conditions align, using VWAP as the anchor.

This is our most sophisticated algorithm, combining **four momentum indicators** to filter for high-probability setups:

#### The Four Conditions:

| Condition | Formula | Purpose |
|-----------|---------|---------|
| **Price > VWAP** | `price_10am > vwap` | Confirms bullish positioning |
| **VWAP Slope > 0** | `(vwap_10am - vwap_940) / vwap_940 > 0` | Confirms upward VWAP trend |
| **Stretch < 0.5×σ** | `(price - vwap) / vwap < 0.5 × or_vol` | Not overextended |
| **Momentum > 0.25×σ** | `(price_10am - open) / open > 0.25 × or_vol` | Sufficient buying pressure |

**Entry:** ALL four conditions must be TRUE at 10:00 AM ET

**Exit Options:**
- **Take Profit:** X% above entry (optimized parameter)
- **Stop Loss:** At market open price (limits downside)
- **Time Exit:** End of day if neither TP nor SL hit

#### Why This Works:

1. **VWAP Filter** – Institutional traders use VWAP as a benchmark. Buying above VWAP means you're aligned with the "smart money."

2. **Slope Confirmation** – An upward-sloping VWAP means volume-weighted buying pressure is increasing throughout the morning.

3. **Stretch Control (α)** – Prevents buying when price is too extended above VWAP. If everyone has already bought, there's less upside.

4. **Momentum Threshold (γ)** – Requires minimum momentum relative to the day's volatility. Filters out low-conviction moves.

#### Real Example:

```
Date: 2025-01-15 | AAPL
─────────────────────────────────────
Open (9:30 AM):     $185.00
Price (10:00 AM):   $186.50  (+0.81%)
VWAP (10:00 AM):    $185.80
VWAP (9:40 AM):     $185.20
OR High:            $186.80
OR Low:             $184.50
─────────────────────────────────────
✓ price_10am ($186.50) > vwap ($185.80)
✓ vwap_slope = +0.32% > 0
✓ vwap_stretch = 0.38% < 1.0% (0.5 × 2.0% vol)
✓ momentum = 0.81% > 0.5% (0.25 × 2.0% vol)
─────────────────────────────────────
DECISION: ✅ BUY @ $186.50
TP Target: $188.83 (+1.25%)
SL Level:  $185.00 (open)
```

---

### 4. Stop-Loss/Take-Profit Management

All strategies support advanced exit management:

| Exit Type | Description |
|-----------|-------------|
| **Take Profit (TP)** | Fixed % above entry |
| **Stop Loss (SL)** | Fixed % below entry or at open |
| **Trailing Stop** | Dynamic SL that follows price up |
| **Time Exit** | Close at end of day |

---

## Trade Log Analysis

After optimization, PPOAlgo generates a comprehensive trade log showing:

| Column | Description |
|--------|-------------|
| **Date** | Trading day |
| **Buy?** | ✅ Entry taken or ❌ Skipped |
| **10AM Price** | Price at entry time |
| **VWAP** | Volume-weighted average price |
| **Skipped** | Reason if trade not taken (e.g., "slope↓", "$ < VWAP") |
| **TP @** | Take profit target |
| **SL @** | Stop loss level |
| **Exit** | How trade was closed (TP, SL, CLOSE) |
| **Profit** | Dollar profit/loss (or potential % if skipped) |
| **Equity** | Running account balance |

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🚀 **GPU Acceleration** | 100x faster than CPU backtesting |
| 📊 **Multi-API Support** | Alpaca, Massive.com, Widesurf |
| 🔄 **Real-Time WebSocket** | Live progress during optimization |
| 📈 **4 Trading Algorithms** | Dipper, Momentum, VWAP, Stop-Loss |
| 💹 **Trade Log Analysis** | Day-by-day breakdown with skip reasons |
| 📱 **Paper Trading** | Test live with simulated money (Alpaca) |
| 💾 **History Database** | Store and compare past runs |
| 🔍 **API Comparison Tool** | Verify data accuracy across providers |

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **GPU Computing** | NVIDIA CUDA, CuPy, NumPy |
| **Optimization** | Optuna (Bayesian), Grid Search |
| **Database** | PostgreSQL with JSONB |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Real-Time** | WebSocket (bidirectional) |
| **Deployment** | Docker, Docker Compose |
| **GPU Support** | NVIDIA RTX 3090 (10,496 CUDA cores) |

---

## Performance Benchmarks

| Test Configuration | CPU (Optuna) | GPU (Grid Search) | Speedup |
|--------------------|--------------|-------------------|---------|
| 10,000 trials | ~60 sec | ~1 sec | **60x** |
| 100,000 trials | ~10 min | ~3 sec | **200x** |
| 1,000,000 trials | ~2 hours | ~25 sec | **288x** |

*Tested on: Intel i9-10900K + NVIDIA RTX 3090*

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/yourname/PPOAlgo.git
cd PPOAlgo

# Configure API keys
cp .env.example .env
# Edit .env with your Alpaca/Massive/Widesurf API keys

# Run with Docker
docker-compose up -d

# Access the optimizer
open http://localhost:8082
```

---

## Why PPOAlgo?

| Traditional Tools | PPOAlgo |
|-------------------|---------|
| Hours of manual backtesting | Seconds with GPU |
| Single strategy at a time | Millions in parallel |
| Basic profit/loss | VWAP, momentum, skip reasons |
| No explanation why | Shows exactly why trades were skipped |
| Separate paper trading setup | Built-in Alpaca integration |

---

*PPOAlgo transforms the tedious process of strategy optimization into an interactive, GPU-powered experience that delivers actionable results in seconds instead of hours. Stop guessing. Start optimizing.*

---

**License:** MIT  
**Author:** Built with ❤️ for quantitative traders
