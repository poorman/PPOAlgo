#!/usr/bin/env python3
"""
Smart Per-Stock Parameter Optimizer
Uses Bayesian Optimization (Optuna) to find optimal buy/sell triggers
for each stock based on its unique volatility characteristics.

Designed for RTX 3089 EVGA GPU acceleration.
"""

import os
import json
import requests
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
import logging

# GPU acceleration
try:
    import optuna
    from optuna.samplers import TPESampler  # Tree-structured Parzen Estimator
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Install optuna: pip install optuna")

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
except ImportError:
    GPU_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration
API_BASE = os.getenv("PPOALGO_API", "http://localhost:8010")
RESULTS_DIR = "optimizer_results"


class StockVolatilityAnalyzer:
    """Analyze stock volatility to set intelligent parameter bounds."""
    
    def __init__(self, bars: List[dict]):
        self.bars = bars
        self.analyze()
    
    def analyze(self):
        """Calculate volatility metrics from price bars."""
        if not self.bars:
            self.avg_daily_range = 5.0
            self.max_daily_gain = 10.0
            self.avg_gap = 2.0
            return
        
        daily_ranges = []
        daily_gains = []
        gaps = []
        
        for i, bar in enumerate(self.bars):
            high = bar.get("h", bar.get("c", 0))
            low = bar.get("l", bar.get("c", 0))
            open_price = bar.get("o", bar.get("c", 0))
            close = bar.get("c", 0)
            
            if open_price > 0:
                # Daily range as % of open
                daily_range = ((high - low) / open_price) * 100
                daily_ranges.append(daily_range)
                
                # Max gain from open to high
                daily_gain = ((high - open_price) / open_price) * 100
                daily_gains.append(daily_gain)
                
                # Gap from previous close
                if i > 0:
                    prev_close = self.bars[i-1].get("c", 0)
                    if prev_close > 0:
                        gap = ((high - prev_close) / prev_close) * 100
                        gaps.append(gap)
        
        self.avg_daily_range = np.mean(daily_ranges) if daily_ranges else 5.0
        self.max_daily_gain = np.percentile(daily_gains, 95) if daily_gains else 10.0
        self.avg_gap = np.mean(gaps) if gaps else 2.0
        self.volatility_score = np.std(daily_gains) if daily_gains else 5.0
        
        logger.info(f"Volatility Analysis: avg_range={self.avg_daily_range:.2f}%, "
                   f"max_gain_95th={self.max_daily_gain:.2f}%, volatility={self.volatility_score:.2f}")
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Get intelligent parameter search bounds based on volatility."""
        # Buy trigger should be lower than typical daily gains
        buy_low = max(0.5, self.avg_gap * 0.3)
        buy_high = min(self.max_daily_gain * 0.8, 20.0)
        
        # Sell trigger should be achievable based on volatility
        sell_low = max(1.0, self.avg_daily_range * 0.3)
        sell_high = min(self.max_daily_gain * 1.2, 30.0)
        
        return {
            "buy_trigger": (buy_low, buy_high),
            "sell_trigger": (sell_low, sell_high),
        }


class StockOptimizer:
    """Optimize trading parameters for a specific stock using Bayesian optimization."""
    
    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        capital: float = 100000,
        n_trials: int = 500,
        optimization_metric: str = "sharpe",  # or "total_return", "win_rate"
    ):
        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.capital = capital
        self.n_trials = n_trials
        self.optimization_metric = optimization_metric
        self.best_params = None
        self.best_score = None
        self.volatility_analyzer = None
        
    def fetch_and_analyze(self) -> bool:
        """Fetch price data and analyze volatility."""
        try:
            resp = requests.get(
                f"{API_BASE}/api/prices",
                params={
                    "symbol": self.symbol,
                    "start": self.start_date,
                    "end": self.end_date,
                    "timeframe": "1Day"
                },
                timeout=30
            )
            resp.raise_for_status()
            bars = resp.json()
            
            if not bars:
                logger.warning(f"No data for {self.symbol}")
                return False
            
            self.volatility_analyzer = StockVolatilityAnalyzer(bars)
            logger.info(f"Loaded {len(bars)} bars for {self.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {self.symbol}: {e}")
            return False
    
    def run_backtest(self, params: dict) -> dict:
        """Run a single backtest with given parameters."""
        try:
            resp = requests.post(
                f"{API_BASE}/api/backtest",
                json={
                    "symbol": self.symbol,
                    "start": self.start_date,
                    "end": self.end_date,
                    "capital": self.capital,
                    "buy_trigger_pct": params["buy_trigger"],
                    "buy_time": params.get("buy_time", "09:00"),
                    "sell_trigger_pct": params["sell_trigger"],
                    "buy_amount": params.get("buy_amount", 0),  # 0 = all capital
                    "compound": params.get("compound", True),
                },
                timeout=30
            )
            resp.raise_for_status()
            return resp.json()["metrics"]
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {}
    
    def objective(self, trial: "optuna.Trial") -> float:
        """Optuna objective function."""
        bounds = self.volatility_analyzer.get_parameter_bounds()
        
        params = {
            "buy_trigger": trial.suggest_float(
                "buy_trigger", 
                bounds["buy_trigger"][0], 
                bounds["buy_trigger"][1],
                step=0.1
            ),
            "sell_trigger": trial.suggest_float(
                "sell_trigger", 
                bounds["sell_trigger"][0], 
                bounds["sell_trigger"][1],
                step=0.1
            ),
            "compound": trial.suggest_categorical("compound", [True, False]),
        }
        
        metrics = self.run_backtest(params)
        
        if not metrics:
            return float("-inf")
        
        # Get optimization target
        score = metrics.get(self.optimization_metric, 0)
        
        # Also track other important metrics
        trial.set_user_attr("total_return", metrics.get("total_return", 0))
        trial.set_user_attr("win_rate", metrics.get("win_rate", 0))
        trial.set_user_attr("max_drawdown", metrics.get("max_drawdown", 0))
        trial.set_user_attr("total_trades", metrics.get("total_trades", 0))
        
        return score if score else float("-inf")
    
    def optimize(self) -> Optional[dict]:
        """Run Bayesian optimization to find best parameters."""
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna not installed")
            return None
        
        if not self.fetch_and_analyze():
            return None
        
        # Create study with TPE sampler (smart Bayesian optimization)
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            study_name=f"{self.symbol}_optimization"
        )
        
        logger.info(f"Starting optimization for {self.symbol} with {self.n_trials} trials...")
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            n_jobs=4,  # Parallel trials
        )
        
        # Get best results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Get additional metrics from best trial
        best_trial = study.best_trial
        
        result = {
            "symbol": self.symbol,
            "optimized_for": self.optimization_metric,
            "best_params": {
                "buy_trigger_pct": round(self.best_params["buy_trigger"], 2),
                "sell_trigger_pct": round(self.best_params["sell_trigger"], 2),
                "compound": self.best_params["compound"],
            },
            "metrics": {
                self.optimization_metric: round(self.best_score, 2),
                "total_return": best_trial.user_attrs.get("total_return", 0),
                "win_rate": best_trial.user_attrs.get("win_rate", 0),
                "max_drawdown": best_trial.user_attrs.get("max_drawdown", 0),
                "total_trades": best_trial.user_attrs.get("total_trades", 0),
            },
            "volatility_profile": {
                "avg_daily_range": round(self.volatility_analyzer.avg_daily_range, 2),
                "max_daily_gain_95th": round(self.volatility_analyzer.max_daily_gain, 2),
                "volatility_score": round(self.volatility_analyzer.volatility_score, 2),
            },
            "search_bounds": self.volatility_analyzer.get_parameter_bounds(),
            "n_trials": self.n_trials,
            "timestamp": datetime.now().isoformat(),
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"OPTIMIZATION COMPLETE: {self.symbol}")
        logger.info(f"Best {self.optimization_metric}: {self.best_score:.2f}")
        logger.info(f"Optimal Buy Trigger: {self.best_params['buy_trigger']:.2f}%")
        logger.info(f"Optimal Sell Trigger: {self.best_params['sell_trigger']:.2f}%")
        logger.info(f"Compound: {self.best_params['compound']}")
        logger.info(f"{'='*60}\n")
        
        return result


class MultiStockOptimizer:
    """Optimize parameters for multiple stocks in parallel."""
    
    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        n_trials_per_stock: int = 200,
        optimization_metric: str = "sharpe",
    ):
        self.symbols = [s.upper() for s in symbols]
        self.start_date = start_date
        self.end_date = end_date
        self.n_trials = n_trials_per_stock
        self.optimization_metric = optimization_metric
        self.results = {}
    
    def optimize_stock(self, symbol: str) -> dict:
        """Optimize a single stock."""
        optimizer = StockOptimizer(
            symbol=symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            n_trials=self.n_trials,
            optimization_metric=self.optimization_metric,
        )
        return optimizer.optimize()
    
    def optimize_all(self, max_workers: int = 2) -> Dict[str, dict]:
        """Optimize all stocks (sequentially to avoid API overload)."""
        for symbol in self.symbols:
            logger.info(f"\n{'#'*60}")
            logger.info(f"Optimizing {symbol}...")
            logger.info(f"{'#'*60}\n")
            
            result = self.optimize_stock(symbol)
            if result:
                self.results[symbol] = result
        
        return self.results
    
    def save_results(self, filename: str = None):
        """Save all results to JSON file."""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        if filename is None:
            filename = f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(RESULTS_DIR, filename)
        
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return filepath
    
    def print_summary(self):
        """Print summary of all optimized stocks."""
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        print(f"{'Symbol':<10} {'Buy%':<8} {'Sell%':<8} {'Sharpe':<10} {'Return%':<10} {'WinRate%':<10}")
        print("-"*80)
        
        for symbol, result in self.results.items():
            params = result["best_params"]
            metrics = result["metrics"]
            print(f"{symbol:<10} "
                  f"{params['buy_trigger_pct']:<8.1f} "
                  f"{params['sell_trigger_pct']:<8.1f} "
                  f"{metrics.get('sharpe', 0):<10.2f} "
                  f"{metrics.get('total_return', 0):<10.2f} "
                  f"{metrics.get('win_rate', 0):<10.2f}")
        
        print("="*80 + "\n")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Per-Stock Parameter Optimizer")
    parser.add_argument("--symbols", nargs="+", default=["TSLA"], help="Stock symbols")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--trials", type=int, default=200, help="Optimization trials per stock")
    parser.add_argument("--metric", default="sharpe", choices=["sharpe", "total_return", "win_rate"])
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("SMART PER-STOCK PARAMETER OPTIMIZER")
    print(f"{'='*60}")
    print(f"Symbols: {args.symbols}")
    print(f"Date Range: {args.start} to {args.end}")
    print(f"Trials per stock: {args.trials}")
    print(f"Optimizing for: {args.metric}")
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"{'='*60}\n")
    
    # Run optimization
    multi_optimizer = MultiStockOptimizer(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        n_trials_per_stock=args.trials,
        optimization_metric=args.metric,
    )
    
    results = multi_optimizer.optimize_all()
    multi_optimizer.save_results()
    multi_optimizer.print_summary()
    
    return results


if __name__ == "__main__":
    main()

