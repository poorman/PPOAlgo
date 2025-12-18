# Stock Parameter Optimizer - Algorithm Documentation

## Overview

This document explains the two main trading algorithms available in the Stock Parameter Optimizer.

---

## 1. Default Algorithm: "Buy the Dip"

**Strategy:** Buy on dip, sell on profit target

### How It Works

The Default algorithm is a **"buy the dip"** strategy that:

1. **Waits for a price DROP** from the market open price
2. **Buys when the price drops by at least X%** (Buy Trigger)
3. **Sells when the price rises by Y%** from the buy price (Sell Trigger)
4. **If target not hit, sells at market close**

### Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| **Buy Trigger** | Minimum % drop from open to trigger a buy | 13.34% |
| **Sell Trigger** | % profit target from buy price | 1.85% |
| **Compound** | Whether to reinvest profits | Yes/No |

### Example Trade

```
Day: 2025-11-20
Open Price: $7.92
Intraday Low: $6.27

Buy Trigger: 13.34%
Threshold = $7.92 × (1 - 0.1334) = $6.86

Since price dropped to $6.86 (≤ threshold):
  → BUY at $6.86 (-13.34% from open)

Sell Trigger: 1.85%
Sell Target = $6.86 × (1 + 0.0185) = $6.99

If price reaches $6.99 during the day:
  → SELL at $6.99 (+1.85% profit) ✓

Result: +$3,514 profit on this trade
```

### Visual Representation

```
Price
  ↑
$7.92 ─────────── Open
  │
  │    ╲
  │     ╲
  │      ╲ Price drops
  │       ╲
$6.86 ─────────── Buy Trigger (-13.34%)
  │        │
  │        │ BUY HERE
  │        │
  │       ╱
  │      ╱ Price rises
  │     ╱
$6.99 ─────────── Sell Target (+1.85% from buy)
  │    │
  │    │ SELL HERE ✓
  │
```

### When to Use

- **Best for:** Volatile stocks with intraday swings
- **Works well when:** Stock tends to dip early and recover
- **Risk:** If price keeps dropping, you buy and lose money

---

## 2. ChatGPT Algorithm: "9AM Momentum"

**Strategy:** Buy at 9AM if price >= open + trigger%, sell before close

### How It Works

The ChatGPT algorithm is a **momentum strategy** that:

1. **Checks the 9AM price** (Eastern Time)
2. **Buys if 9AM price is UP by at least X%** from market open (Buy Trigger)
3. **Sells when the price rises by Y%** from buy price (Sell Trigger)
4. **If target not hit, sells at market close**

### Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| **Buy Trigger** | Minimum % increase at 9AM to trigger buy | 0.72% |
| **Sell Trigger** | % profit target from 9AM buy price | 6.7% |
| **Compound** | Whether to reinvest profits | Yes/No |

### Example Trade

```
Day: 2021-03-01
Open Price: $230.04
9AM Price: $232.50

Buy Trigger: 0.72%
Threshold = $230.04 × (1 + 0.0072) = $231.70

Since 9AM price $232.50 ≥ $231.70:
  → BUY at $232.50 (+1.07% from open)

Sell Trigger: 6.7%
Sell Target = $232.50 × (1 + 0.067) = $248.08

If price reaches $248.08 during the day:
  → SELL at $248.08 (+6.7% profit) ✓

If not, sell at close price.
```

### Visual Representation

```
Price
  ↑
        ╱ 9AM price above threshold
$232.50 ─────────── 9AM Price → BUY HERE
  │    ╱
$231.70 ─────────── Buy Threshold (+0.72%)
  │   ╱
$230.04 ─────────── Market Open
  │
  │        Later in day:
  │             ╱
  │            ╱
$248.08 ─────────── Sell Target (+6.7%)
  │           │
  │           │ SELL HERE ✓
```

### When to Use

- **Best for:** Stocks with morning momentum
- **Works well when:** Stocks that gap up tend to continue rising
- **Risk:** False breakouts - price rises at 9AM then drops

---

## 3. ChatGPT with VWAP Algorithm

**Strategy:** Buy at 10AM if price >= open + trigger% AND price > VWAP

### How It Works

This algorithm combines the ChatGPT momentum strategy with a **VWAP (Volume Weighted Average Price) filter**:

1. **Optimization Phase:** Uses SAME logic as ChatGPT 9AM Strategy (no VWAP filter)
2. **Backtesting Phase:** Applies VWAP as additional entry filter

### Entry Conditions (Backtesting)

| Condition | Description |
|-----------|-------------|
| Price_10AM >= Open × (1 + Buy Trigger%) | Price must be UP from open |
| Price_10AM > VWAP | Price must be ABOVE the VWAP |

Both conditions must be true to enter a trade.

### Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| **Buy Trigger** | Minimum % increase at 10AM to trigger buy | 2.0% |
| **Sell Trigger** | % profit target from 10AM buy price | 10.0% |
| **Compound** | Whether to reinvest profits | Yes/No |

### Example Trade

```
Day: 2025-01-15
Open Price: $130.00
10AM Price: $133.00 (+2.3% from open)
VWAP: $131.50

Buy Trigger: 2.0%
Threshold = $130.00 × (1 + 0.02) = $132.60

Check conditions:
  1. 10AM Price $133.00 >= $132.60 ✓
  2. 10AM Price $133.00 > VWAP $131.50 ✓

Both true → BUY at $133.00

Sell Trigger: 10%
Sell Target = $133.00 × (1 + 0.10) = $146.30

If price reaches $146.30 → SELL at $146.30 (+10% profit) ✓
If not, sell at close price.
```

### Visual Representation

```
Price
  ↑
            10AM price above threshold AND VWAP
$133.00 ─────────── 10AM Price → BUY HERE
  │      ↑
$132.60 ─────────── Buy Threshold (+2%)
  │     ╱
$131.50 ─────────── VWAP (must be below 10AM price)
  │    ╱
$130.00 ─────────── Market Open
  │
  │        Later in day:
  │             ╱
$146.30 ─────────── Sell Target (+10%)
  │           │
  │           │ SELL HERE ✓
```

### Why Use VWAP Filter?

| Benefit | Description |
|---------|-------------|
| **Quality Filter** | VWAP represents fair value; trading above = strength |
| **Reduces False Signals** | Filters out weak momentum moves |
| **Fewer Trades** | More selective entries = higher quality |
| **Better Win Rate** | Potential for improved accuracy |

### When to Use

- **Best for:** Stocks with institutional buying (VWAP respect)
- **Works well when:** Strong momentum with volume confirmation
- **Trade-off:** Fewer trades but potentially higher quality

---

## Key Differences

| Aspect | Default (Dip) | ChatGPT (Momentum) | ChatGPT + VWAP |
|--------|---------------|-------------------|----------------|
| **Buy condition** | Price DROPS by X% | Price RISES by X% at 9AM | Price RISES by X% AND > VWAP |
| **Strategy type** | Mean reversion | Momentum | Momentum + Volume filter |
| **When to buy** | Intraday dip | 9AM Eastern Time | 10AM Eastern Time |
| **Best market** | Choppy/Range-bound | Trending/Breakout | Strong trend with volume |
| **Filter** | None | None | VWAP |

---

## Optimization Process

The optimizer tests thousands of parameter combinations to find:
- **Best Buy Trigger %** - The optimal dip/rise percentage
- **Best Sell Trigger %** - The optimal profit target
- **Compound setting** - Whether reinvesting helps

### Optimization Metrics

| Metric | Description |
|--------|-------------|
| **Total Return** | Overall profit/loss percentage |
| **Win Rate** | % of profitable trades |
| **Sharpe Ratio** | Risk-adjusted return |
| **Max Drawdown** | Largest peak-to-trough decline |

---

## Table Columns Explained

| Column | Description |
|--------|-------------|
| **Date** | Trading day |
| **Buy?** | ✓ = Bought, ✗ = No buy |
| **Open** | Market open price |
| **Buy @** | Actual buy price (Default algo) |
| **9AM Price** | Price at 9AM Eastern (ChatGPT algo) |
| **% vs Open** | % change from open to buy price |
| **9AM Δ %** | % change from open to 9AM (ChatGPT) |
| **Close** | Market close price |
| **% @ Close** | % change from buy price to close |
| **Sell @** | Actual sell price (✓ = hit target) |
| **Profit** | Dollar profit/loss on trade |
| **Equity** | Running total portfolio value |

---

## Tips for Best Results

1. **Use longer date ranges** - More data = more reliable optimization
2. **Check Win Rate** - High total return with low win rate is risky
3. **Consider volatility** - High volatility stocks work better for dip buying
4. **Backtest different periods** - Ensure strategy works in various markets
5. **Cache data** - Look for 💾 DB badge for faster optimizations

---

*Last updated: December 2025*

