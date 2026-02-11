// ============================================================================
// rust_vwap – Rayon-parallel VWAP Momentum 10am backtester
// ============================================================================
//
// Reads JSON from stdin, runs a massively-parallel grid search using Rayon
// (one work-unit per parameter combination, NOT per stock), and writes JSON
// results to stdout.
//
// Design principles:
//   1. Zero shared mutable state – each combo gets its own immutable view of bars
//   2. Parallel over population (parameter combos), sequential inside backtest
//   3. Rayon auto-tunes to 32 logical processors on the i9-13900K
//
// Input JSON schema (via stdin):
// {
//   "bars": [ { "o": f64, "h": f64, "l": f64, "c": f64,
//               "price_10am": f64, "vwap": f64, "vwap_940": f64,
//               "or_high": f64, "or_low": f64 } ... ],
//   "capital": f64,
//   "buy_range": [min, max, step],
//   "sell_range": [min, max, step],
//   "metric": "total_return" | "sharpe" | "win_rate"
// }
//
// Output JSON schema (via stdout):
// {
//   "best_params": { "buy_trigger_pct": f64, "sell_trigger_pct": f64 },
//   "metrics": { "total_return": f64, "sharpe": f64, "win_rate": f64,
//                "max_drawdown": f64, "total_trades": i64 },
//   "combinations_tested": i64,
//   "threads_used": usize
// }

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::{self, Read};

// ============================================================================
// Data structures  (all immutable / Clone – no shared mutable state)
// ============================================================================

#[derive(Debug, Deserialize, Clone)]
struct Bar {
    o: f64,
    h: f64,
    l: f64,
    c: f64,
    #[serde(default)]
    price_10am: f64,
    #[serde(default)]
    vwap: f64,
    #[serde(default)]
    vwap_940: f64,
    #[serde(default)]
    or_high: f64,
    #[serde(default)]
    or_low: f64,
}

#[derive(Debug, Deserialize)]
struct Input {
    bars: Vec<Bar>,
    #[serde(default = "default_capital")]
    capital: f64,
    buy_range: [f64; 3],  // [min, max, step]
    sell_range: [f64; 3],
    #[serde(default = "default_metric")]
    metric: String,
}

fn default_capital() -> f64 { 100_000.0 }
fn default_metric() -> String { "total_return".into() }

#[derive(Debug, Serialize, Clone)]
struct BestParams {
    buy_trigger_pct: f64,
    sell_trigger_pct: f64,
}

#[derive(Debug, Serialize, Clone)]
struct Metrics {
    total_return: f64,
    sharpe: f64,
    win_rate: f64,
    max_drawdown: f64,
    total_trades: i64,
}

#[derive(Debug, Serialize)]
struct Output {
    best_params: BestParams,
    metrics: Metrics,
    combinations_tested: i64,
    threads_used: usize,
}

// ============================================================================
// VWAP entry signal (pure function, no mutation)
// ============================================================================

#[inline]
fn vwap_entry_signal(bar: &Bar, alpha: f64, gamma: f64) -> bool {
    let open  = bar.o;
    let p10   = bar.price_10am;
    let vwap  = bar.vwap;
    let v940  = bar.vwap_940;
    let or_h  = bar.or_high;
    let or_l  = bar.or_low;

    if p10 <= 0.0 || vwap <= 0.0 || v940 <= 0.0 || open <= 0.0 {
        return false;
    }

    // Condition 1: price above VWAP
    if p10 <= vwap { return false; }

    // Condition 2: VWAP slope > 0 (trending up from 9:40 to 10:00)
    let vwap_slope = (vwap - v940) / v940;
    if vwap_slope <= 0.0 { return false; }

    // VWAP-anchored opening range volatility
    let or_vol = if vwap > 0.0 {
        f64::max(or_h - vwap, vwap - or_l) / vwap
    } else {
        0.02
    };

    // Condition 3: not too extended from VWAP
    let vwap_stretch = (p10 - vwap) / vwap;
    let stretch_ok = if or_vol > 0.0 {
        vwap_stretch < alpha * or_vol
    } else {
        vwap_stretch < 0.01
    };
    if !stretch_ok { return false; }

    // Condition 4: sufficient momentum from open
    let momentum = (p10 - open) / open;
    let momentum_ok = if or_vol > 0.0 {
        momentum > gamma * or_vol
    } else {
        momentum > 0.005
    };
    momentum_ok
}

// ============================================================================
// Single backtest run (purely sequential, reads bars immutably)
// ============================================================================

fn run_backtest(bars: &[Bar], capital: f64, take_profit_pct: f64) -> Metrics {
    let alpha = 0.50_f64;
    let gamma = 0.25_f64;

    let mut equity = capital;
    let mut total_trades: i64 = 0;
    let mut winning_trades: i64 = 0;
    let mut total_profit = 0.0_f64;
    let mut total_loss = 0.0_f64;
    let mut max_equity = equity;
    let mut max_drawdown = 0.0_f64;

    // For Sharpe: collect per-trade returns
    let mut trade_returns: Vec<f64> = Vec::new();

    for bar in bars {
        if bar.o <= 0.0 { continue; }

        if !vwap_entry_signal(bar, alpha, gamma) { continue; }

        let buy_price = bar.price_10am;
        if buy_price <= 0.0 { continue; }

        let shares = (equity / buy_price).floor() as i64;
        if shares <= 0 { continue; }

        total_trades += 1;

        let tp_price = buy_price * (1.0 + take_profit_pct);

        // Exit: TP if high hits it, otherwise exit at close
        let sell_price = if bar.h >= tp_price { tp_price } else { bar.c };

        let profit = (shares as f64) * (sell_price - buy_price);
        let pct_return = (sell_price - buy_price) / buy_price;

        trade_returns.push(pct_return);

        if profit > 0.0 {
            winning_trades += 1;
            total_profit += profit;
        } else {
            total_loss += profit.abs();
        }

        // Compound
        equity += profit;

        if equity > max_equity { max_equity = equity; }
        if max_equity > 0.0 {
            let dd = (max_equity - equity) / max_equity;
            if dd > max_drawdown { max_drawdown = dd; }
        }
    }

    let total_return = (equity - capital) / capital;
    let win_rate = if total_trades > 0 {
        winning_trades as f64 / total_trades as f64
    } else {
        0.0
    };

    // Sharpe ratio (annualized, assuming ~252 trading days)
    let sharpe = if trade_returns.len() >= 2 {
        let n = trade_returns.len() as f64;
        let mean = trade_returns.iter().sum::<f64>() / n;
        let variance = trade_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (n - 1.0);
        let std = variance.sqrt();
        if std > 1e-12 {
            (mean / std) * (252.0_f64).sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };

    Metrics {
        total_return,
        sharpe,
        win_rate,
        max_drawdown,
        total_trades,
    }
}

// ============================================================================
// Parallel grid search  (Rayon splits combos across 32 cores)
// ============================================================================

fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let mut v = Vec::new();
    let mut x = start;
    while x < stop {
        v.push((x * 100_000.0).round() / 100_000.0); // avoid FP drift
        x += step;
    }
    v
}

fn grid_search(bars: &[Bar], capital: f64, buy_range: [f64; 3], sell_range: [f64; 3], metric: &str) -> Output {
    let buy_values  = arange(buy_range[0], buy_range[1], buy_range[2]);
    let sell_values = arange(sell_range[0], sell_range[1], sell_range[2]);

    // Build all parameter combinations (owned, no shared state)
    let combos: Vec<(f64, f64)> = buy_values.iter()
        .flat_map(|&b| sell_values.iter().map(move |&s| (b, s)))
        .collect();

    let total = combos.len() as i64;
    let threads = rayon::current_num_threads();

    // Parallel map over combos → each gets an immutable borrow of `bars`
    let results: Vec<(f64, f64, Metrics)> = combos
        .par_iter()
        .map(|&(buy_trig, sell_trig)| {
            let m = run_backtest(bars, capital, sell_trig);
            // Note: buy_trig is used only for the ChatGPT 9AM entry filter
            // which is already embedded in the VWAP signal.  The sell_trig
            // maps to take_profit_pct.
            //
            // For VWAP strategy: buy trigger isn't used separately since
            // entry is controlled by the VWAP conditions.
            // We still grid-search over buy triggers for compatibility
            // and store the best combo.
            (buy_trig, sell_trig, m)
        })
        .collect();

    // Find best result (sequential reduction, no mutation during parallel phase)
    let mut best_score = f64::NEG_INFINITY;
    let mut best_buy = 0.0_f64;
    let mut best_sell = 0.0_f64;
    let mut best_metrics = Metrics {
        total_return: 0.0, sharpe: 0.0, win_rate: 0.0,
        max_drawdown: 0.0, total_trades: 0,
    };

    for (buy_trig, sell_trig, m) in &results {
        let score = match metric {
            "sharpe"     => m.sharpe,
            "win_rate"   => m.win_rate,
            _            => m.total_return,
        };
        if score > best_score {
            best_score  = score;
            best_buy    = *buy_trig;
            best_sell   = *sell_trig;
            best_metrics = m.clone();
        }
    }

    Output {
        best_params: BestParams {
            buy_trigger_pct: (best_buy * 10000.0).round() / 100.0,   // decimal → %
            sell_trigger_pct: (best_sell * 10000.0).round() / 100.0,
        },
        metrics: best_metrics,
        combinations_tested: total,
        threads_used: threads,
    }
}

// ============================================================================
// Main: read JSON from stdin → grid search → write JSON to stdout
// ============================================================================

fn main() {
    // Set Rayon thread pool (auto-detects 32 logical processors on i9-13900K)
    // Can be overridden with RAYON_NUM_THREADS env var
    let mut input_str = String::new();
    io::stdin().read_to_string(&mut input_str)
        .expect("Failed to read stdin");

    let input: Input = serde_json::from_str(&input_str)
        .expect("Failed to parse input JSON");

    let output = grid_search(
        &input.bars,
        input.capital,
        input.buy_range,
        input.sell_range,
        &input.metric,
    );

    let json = serde_json::to_string(&output)
        .expect("Failed to serialize output");
    println!("{}", json);
}
