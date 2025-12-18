#!/usr/bin/env python3
"""
Backtester Service
Utility functions for backtesting and trade log generation.

NOTE: The main generate_trade_log function is still in server.py due to its
complexity and tight integration with the optimization loop. This module
contains helper utilities that can be shared.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class StockVolatilityAnalyzer:
    """Analyze stock volatility to set intelligent parameter bounds."""
    
    def __init__(self, bars: List[dict]):
        self.bars = bars
        self.analyze()
    
    def analyze(self):
        """Calculate volatility metrics from price bars."""
        if not self.bars:
            self.avg_daily_range = 0.02
            self.max_daily_gain = 0.05
            self.volatility_score = 5
            return
        
        daily_ranges = []
        daily_gains = []
        
        for bar in self.bars:
            o = bar.get("o", 0)
            h = bar.get("h", 0)
            l = bar.get("l", 0)
            c = bar.get("c", 0)
            
            if o > 0:
                daily_range = (h - l) / o
                daily_gain = (h - o) / o
                daily_ranges.append(daily_range)
                daily_gains.append(daily_gain)
        
        if daily_ranges:
            self.avg_daily_range = sum(daily_ranges) / len(daily_ranges)
            self.max_daily_gain = max(daily_gains) if daily_gains else 0.05
        else:
            self.avg_daily_range = 0.02
            self.max_daily_gain = 0.05
        
        # Score from 1-10 based on volatility
        if self.avg_daily_range < 0.01:
            self.volatility_score = 2
        elif self.avg_daily_range < 0.02:
            self.volatility_score = 4
        elif self.avg_daily_range < 0.03:
            self.volatility_score = 6
        elif self.avg_daily_range < 0.05:
            self.volatility_score = 8
        else:
            self.volatility_score = 10
    
    def get_parameter_bounds(self) -> dict:
        """Get suggested parameter bounds based on volatility."""
        return {
            "buy_trigger_min": max(0.5, self.avg_daily_range * 20),
            "buy_trigger_max": min(10, self.avg_daily_range * 200),
            "sell_trigger_min": max(1.0, self.avg_daily_range * 30),
            "sell_trigger_max": min(15, self.avg_daily_range * 300),
        }


def calculate_metrics(trade_log: List[dict], initial_capital: float = 100000) -> dict:
    """
    Calculate performance metrics from a trade log.
    
    Args:
        trade_log: List of trade entries
        initial_capital: Starting capital
        
    Returns:
        Dict with metrics: total_return, win_rate, sharpe, etc.
    """
    if not trade_log:
        return {
            "total_return": 0,
            "win_rate": 0,
            "sharpe": 0,
            "max_drawdown": 0,
            "num_trades": 0
        }
    
    trades_with_action = [t for t in trade_log if t.get("bought")]
    winning_trades = [t for t in trades_with_action if t.get("profit", 0) > 0]
    
    final_equity = trade_log[-1].get("equity", initial_capital) if trade_log else initial_capital
    total_return = ((final_equity - initial_capital) / initial_capital) * 100
    
    num_trades = len(trades_with_action)
    win_rate = (len(winning_trades) / num_trades * 100) if num_trades > 0 else 0
    
    # Simple Sharpe approximation
    returns = []
    for i in range(1, len(trade_log)):
        prev_eq = trade_log[i-1].get("equity", initial_capital)
        curr_eq = trade_log[i].get("equity", initial_capital)
        if prev_eq > 0:
            returns.append((curr_eq - prev_eq) / prev_eq)
    
    if returns and len(returns) > 1:
        import statistics
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        sharpe = (avg_return / std_return * (252 ** 0.5)) if std_return > 0 else 0
    else:
        sharpe = 0
    
    # Max drawdown
    peak = initial_capital
    max_dd = 0
    for t in trade_log:
        eq = t.get("equity", initial_capital)
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    
    return {
        "total_return": round(total_return, 2),
        "win_rate": round(win_rate, 2),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd * 100, 2),
        "num_trades": num_trades
    }


def get_optimal_buy_time_simple(symbol: str, optimization_metric: str) -> str:
    """
    Simple buy time determination based on market patterns.
    Returns optimal buy time in HH:MM format (CDT).
    """
    # Default: 10:00 AM CDT for most stocks
    # Morning gap fill patterns typically complete by 10:00
    
    if optimization_metric in ["win_rate", "sharpe"]:
        # For risk-adjusted metrics, slightly later is better
        return "10:15"
    elif optimization_metric == "total_return":
        # For max return, earlier can capture bigger moves
        return "09:45"
    else:
        return "10:00"


def sanitize_metric(value, decimals: int = 2):
    """Sanitize metric values, replacing invalid values with None."""
    import math
    if value is None:
        return None
    try:
        if math.isnan(value) or math.isinf(value):
            return None
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return None
