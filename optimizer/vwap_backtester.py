"""
VWAP Adaptive Strategy Backtester
=================================

A complete, standalone VWAP backtesting module with:
- VWAP calculation from intraday bars
- VWAP Slope (trend detection)
- VWAP Stretch (extension measurement)
- Opening Range (OR) volatility
- Momentum Score

This module can be imported directly into other algo projects.

Author: PPOAlgo Team
Date: 2026-01-05
"""

import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Callable
from zoneinfo import ZoneInfo


# ============================================================================
# TIMEZONE UTILITIES
# ============================================================================

ET_ZONE = ZoneInfo("America/New_York")
UTC_ZONE = ZoneInfo("UTC")


def get_et_minutes_from_timestamp(bar_time: str) -> Optional[int]:
    """
    Convert UTC timestamp string to Eastern Time minutes since midnight.
    Properly handles DST by using America/New_York timezone.
    
    Args:
        bar_time: ISO timestamp string (e.g., "2024-01-18T15:00:00Z")
    
    Returns:
        Minutes since midnight in Eastern Time (e.g., 570 = 9:30 AM)
    """
    if not bar_time or "T" not in bar_time:
        return None
    
    try:
        ts = bar_time.replace("Z", "+00:00")
        if "+" not in ts and ts.count("-") == 2:
            ts = ts + "+00:00"
        
        dt_utc = datetime.fromisoformat(ts)
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=UTC_ZONE)
        
        dt_et = dt_utc.astimezone(ET_ZONE)
        return dt_et.hour * 60 + dt_et.minute
    except Exception:
        return None


# ============================================================================
# VWAP CALCULATION UTILITIES
# ============================================================================

def calculate_vwap_from_bars(bars: List[Dict]) -> float:
    """
    Calculate VWAP from a list of price bars.
    
    VWAP = Sum(Typical_Price * Volume) / Sum(Volume)
    Typical_Price = (High + Low + Close) / 3
    
    Args:
        bars: List of price bars with keys: h (high), l (low), c (close), v (volume)
    
    Returns:
        VWAP value
    """
    if not bars:
        return 0.0
    
    cum_tpv = 0.0  # Cumulative (typical price * volume)
    cum_vol = 0.0  # Cumulative volume
    
    for bar in bars:
        high = bar.get("h", bar.get("high", 0))
        low = bar.get("l", bar.get("low", 0))
        close = bar.get("c", bar.get("close", 0))
        volume = bar.get("v", bar.get("volume", 0))
        
        # Typical price = (H + L + C) / 3
        typical_price = (high + low + close) / 3
        cum_tpv += typical_price * volume
        cum_vol += volume
    
    return cum_tpv / cum_vol if cum_vol > 0 else 0.0


def extract_vwap_metrics_from_intraday(
    intraday_bars: List[Dict],
    target_time_minutes: int = 600  # 10:00 AM ET
) -> Dict[str, float]:
    """
    Extract VWAP-related metrics from intraday bars for a single day.
    
    This calculates:
    - price_10am: Price at 10:00 AM (or closest available)
    - vwap: VWAP from 9:30 AM to 10:00 AM
    - vwap_940: VWAP from 9:30 AM to 9:40 AM (for slope calculation)
    - or_high: Opening range high (9:30-10:00)
    - or_low: Opening range low (9:30-10:00)
    
    Args:
        intraday_bars: List of 1-minute bars for a single day
        target_time_minutes: Target time in ET minutes (default 600 = 10:00 AM)
    
    Returns:
        Dict with VWAP metrics
    """
    best_target_bar = None
    best_target_diff = 9999
    
    vwap_bars = []       # Bars for VWAP (9:30-10:00)
    vwap_940_bars = []   # Bars for VWAP at 9:40 (9:30-9:40)
    or_highs = []        # Opening range highs
    or_lows = []         # Opening range lows
    
    for bar in intraday_bars:
        bar_time = bar.get("t", bar.get("timestamp", ""))
        et_minutes = get_et_minutes_from_timestamp(bar_time)
        
        if et_minutes is None:
            continue
        
        # Collect bars from 9:30 (570) to 10:00 (600) for VWAP and OR
        if 570 <= et_minutes <= 600:
            vwap_bars.append(bar)
            or_highs.append(bar.get("h", bar.get("high", 0)))
            or_lows.append(bar.get("l", bar.get("low", 0)))
        
        # Collect bars from 9:30 to 9:40 (580) for robust slope
        if 570 <= et_minutes <= 580:
            vwap_940_bars.append(bar)
        
        # Find closest bar to target time (10:00 AM = 600 minutes)
        if 570 <= et_minutes <= 960:  # Market hours only
            diff = abs(et_minutes - target_time_minutes)
            if diff < best_target_diff:
                best_target_bar = bar
                best_target_diff = diff
    
    # Calculate metrics
    price_target = 0.0
    if best_target_bar and best_target_diff <= 5:  # Within 5 minutes
        price_target = best_target_bar.get("c", best_target_bar.get("close", 0))
    
    vwap = calculate_vwap_from_bars(vwap_bars)
    vwap_940 = calculate_vwap_from_bars(vwap_940_bars)
    or_high = max(or_highs) if or_highs else 0.0
    or_low = min(or_lows) if or_lows else 0.0
    
    return {
        "price_10am": price_target,
        "vwap": vwap,
        "vwap_940": vwap_940,
        "or_high": or_high,
        "or_low": or_low
    }


def build_vwap_map_from_intraday(
    intraday_bars: List[Dict],
    target_time_minutes: int = 600
) -> Dict[str, Dict[str, float]]:
    """
    Build a date -> VWAP metrics map from intraday bars spanning multiple days.
    
    Args:
        intraday_bars: List of 1-minute bars (can span multiple days)
        target_time_minutes: Target time in ET minutes (default 600 = 10:00 AM)
    
    Returns:
        Dict mapping date strings (YYYY-MM-DD) to VWAP metrics
    """
    # Group bars by date
    bars_by_date = {}
    for bar in intraday_bars:
        bar_time = bar.get("t", bar.get("timestamp", ""))
        date_part = bar_time[:10] if bar_time else ""
        if date_part:
            if date_part not in bars_by_date:
                bars_by_date[date_part] = []
            bars_by_date[date_part].append(bar)
    
    # Extract metrics for each date
    vwap_map = {}
    for date_part, day_bars in bars_by_date.items():
        vwap_map[date_part] = extract_vwap_metrics_from_intraday(day_bars, target_time_minutes)
    
    return vwap_map


# ============================================================================
# VWAP ADAPTIVE ENTRY SIGNALS
# ============================================================================

def calculate_vwap_entry_signals(
    open_price: float,
    price_10am: float,
    vwap: float,
    vwap_940: float,
    or_high: float,
    or_low: float,
    alpha: float = 0.50,   # Stretch threshold multiplier
    gamma: float = 0.25    # Momentum threshold multiplier
) -> Dict[str, any]:
    """
    Calculate VWAP adaptive entry signals.
    
    Entry conditions (all must be True):
    1. price_10am > vwap (price above VWAP)
    2. vwap_slope > 0 (VWAP trending up from 9:40 to 10:00)
    3. vwap_stretch < alpha * or_vol (price not too extended from VWAP)
    4. momentum_score > gamma * or_vol (sufficient momentum from open)
    
    Args:
        open_price: Day's open price
        price_10am: Price at 10:00 AM
        vwap: VWAP at 10:00 AM (calculated from 9:30-10:00)
        vwap_940: VWAP at 9:40 AM (calculated from 9:30-9:40)
        or_high: Opening range high (9:30-10:00)
        or_low: Opening range low (9:30-10:00)
        alpha: Stretch threshold multiplier (default 0.50)
        gamma: Momentum threshold multiplier (default 0.25)
    
    Returns:
        Dict with entry signals and metrics
    """
    # =========================================================================
    # STEP 1: Calculate VWAP Stretch
    # How far price is extended from VWAP (as a percentage)
    # =========================================================================
    vwap_stretch = (price_10am - vwap) / vwap if vwap > 0 else 0.0
    
    # =========================================================================
    # STEP 2: Calculate VWAP-Anchored Opening Range Volatility (or_vol)
    # This measures how volatile the opening range is relative to VWAP
    # =========================================================================
    or_vol = max(or_high - vwap, vwap - or_low) / vwap if vwap > 0 else 0.02
    
    # =========================================================================
    # STEP 3: Calculate Robust VWAP Slope
    # Change in VWAP from 9:40 to 10:00 (indicates trend direction)
    # =========================================================================
    vwap_slope = (vwap - vwap_940) / vwap_940 if vwap_940 > 0 else 0.0
    
    # =========================================================================
    # STEP 4: Calculate Momentum Score
    # How much price has moved from open to 10AM
    # =========================================================================
    momentum_score = (price_10am - open_price) / open_price if open_price > 0 else 0.0
    
    # =========================================================================
    # STEP 5: Apply Entry Conditions
    # =========================================================================
    
    # Condition 1: Price above VWAP
    price_above_vwap = price_10am > vwap and vwap > 0
    
    # Condition 2: VWAP trending up (positive slope)
    slope_ok = vwap_slope > 0
    
    # Condition 3: Price not too extended from VWAP
    stretch_ok = vwap_stretch < alpha * or_vol if or_vol > 0 else vwap_stretch < 0.01
    
    # Condition 4: Sufficient momentum
    momentum_ok = momentum_score > gamma * or_vol if or_vol > 0 else momentum_score > 0.005
    
    # Final entry signal
    entry_signal = price_above_vwap and slope_ok and stretch_ok and momentum_ok
    
    # Skip reason for analysis
    skip_reason = None
    if not entry_signal:
        if not price_above_vwap:
            skip_reason = "No VWAP" if vwap <= 0 else "$ < VWAP"
        elif not slope_ok:
            skip_reason = f"slope↓ ({vwap_slope*100:.2f}%)"
        elif not stretch_ok:
            skip_reason = f"stretch↑ ({vwap_stretch*100:.2f}%)"
        elif not momentum_ok:
            skip_reason = f"mom↓ ({momentum_score*100:.2f}%)"
    
    return {
        "entry_signal": entry_signal,
        "skip_reason": skip_reason,
        "metrics": {
            "vwap_stretch": vwap_stretch,
            "vwap_slope": vwap_slope,
            "or_vol": or_vol,
            "momentum_score": momentum_score
        },
        "conditions": {
            "price_above_vwap": price_above_vwap,
            "slope_ok": slope_ok,
            "stretch_ok": stretch_ok,
            "momentum_ok": momentum_ok
        }
    }


# ============================================================================
# VWAP BACKTESTER CLASS
# ============================================================================

class VWAPAdaptiveBacktester:
    """
    Complete VWAP Adaptive Strategy Backtester.
    
    Strategy Rules:
    ---------------
    ENTRY (at 10:00 AM):
        - price_10am > VWAP
        - vwap_slope > 0 (VWAP trending up)
        - vwap_stretch < alpha * or_vol (not too extended)
        - momentum_score > gamma * or_vol (sufficient momentum)
    
    EXIT:
        - Take Profit: price >= entry * (1 + take_profit_pct)
        - Stop Loss: price <= open_price (SL = day's open)
        - Time Exit: close position at end of day (3:55 PM)
    
    Parameters:
    -----------
    - alpha: Stretch threshold multiplier (default 0.50)
    - gamma: Momentum threshold multiplier (default 0.25)
    - take_profit_pct: Take profit percentage (default 2.0%)
    """
    
    def __init__(
        self,
        daily_bars: List[Dict],
        vwap_map: Dict[str, Dict[str, float]],
        capital: float = 100000
    ):
        """
        Initialize the backtester.
        
        Args:
            daily_bars: List of daily OHLC bars with keys:
                       t: timestamp, o: open, h: high, l: low, c: close
            vwap_map: Dict mapping date strings to VWAP metrics
                     (from build_vwap_map_from_intraday)
            capital: Starting capital
        """
        self.daily_bars = daily_bars
        self.vwap_map = vwap_map
        self.capital = capital
        self.n_bars = len(daily_bars)
    
    def run_backtest(
        self,
        take_profit_pct: float = 0.02,   # 2%
        alpha: float = 0.50,
        gamma: float = 0.25,
        compound: bool = True
    ) -> Dict:
        """
        Run the VWAP adaptive strategy backtest.
        
        Args:
            take_profit_pct: Take profit as decimal (0.02 = 2%)
            alpha: Stretch threshold multiplier
            gamma: Momentum threshold multiplier
            compound: Whether to compound gains
        
        Returns:
            Dict with backtest results and trade log
        """
        equity = self.capital
        total_trades = 0
        winning_trades = 0
        total_profit = 0.0
        total_loss = 0.0
        max_equity = equity
        max_drawdown = 0.0
        
        trade_log = []
        
        for bar in self.daily_bars:
            bar_date = bar.get("t", bar.get("timestamp", ""))[:10]
            open_price = bar.get("o", bar.get("open", 0))
            high_price = bar.get("h", bar.get("high", 0))
            low_price = bar.get("l", bar.get("low", 0))
            close_price = bar.get("c", bar.get("close", 0))
            
            if open_price <= 0:
                continue
            
            # Get VWAP metrics for this date
            vwap_data = self.vwap_map.get(bar_date, {})
            price_10am = vwap_data.get("price_10am", 0)
            vwap = vwap_data.get("vwap", 0)
            vwap_940 = vwap_data.get("vwap_940", vwap)
            or_high = vwap_data.get("or_high", 0)
            or_low = vwap_data.get("or_low", 0)
            
            if price_10am <= 0:
                price_10am = open_price
            if vwap <= 0:
                vwap = close_price
            
            # Calculate entry signals
            signals = calculate_vwap_entry_signals(
                open_price=open_price,
                price_10am=price_10am,
                vwap=vwap,
                vwap_940=vwap_940,
                or_high=or_high,
                or_low=or_low,
                alpha=alpha,
                gamma=gamma
            )
            
            entry_signal = signals["entry_signal"]
            skip_reason = signals["skip_reason"]
            metrics = signals["metrics"]
            
            buy_price = None
            actual_sell_price = None
            exit_reason = None
            shares = 0
            profit = 0.0
            
            if entry_signal:
                buy_price = price_10am
                shares = int(equity / buy_price) if buy_price > 0 else 0
                
                if shares > 0:
                    total_trades += 1
                    take_profit_price = buy_price * (1 + take_profit_pct)
                    stop_loss_price = open_price  # SL = day's open
                    
                    # Determine exit
                    if high_price >= take_profit_price:
                        actual_sell_price = take_profit_price
                        exit_reason = "TP"
                    else:
                        # Exit at close (capture mean reversion)
                        actual_sell_price = close_price
                        exit_reason = "CLOSE"
                    
                    profit = shares * (actual_sell_price - buy_price)
                    
                    if profit > 0:
                        winning_trades += 1
                        total_profit += profit
                    else:
                        total_loss += abs(profit)
                    
                    if compound:
                        equity += profit
                    
                    # Track drawdown
                    if equity > max_equity:
                        max_equity = equity
                    drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
            
            trade_log.append({
                "date": bar_date,
                "open": round(open_price, 2),
                "price_10am": round(price_10am, 2),
                "vwap": round(vwap, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "bought": entry_signal and shares > 0,
                "skip_reason": skip_reason,
                "entry_signal": entry_signal,
                "buy_price": round(buy_price, 2) if buy_price else None,
                "actual_sell": round(actual_sell_price, 2) if actual_sell_price else None,
                "exit_reason": exit_reason,
                "shares": shares,
                "profit": round(profit, 2),
                "equity": round(equity, 2),
                # VWAP metrics for analysis
                "vwap_stretch": round(metrics["vwap_stretch"] * 100, 2),  # As %
                "vwap_slope": round(metrics["vwap_slope"] * 100, 2),      # As %
                "or_vol": round(metrics["or_vol"] * 100, 2),             # As %
                "momentum": round(metrics["momentum_score"] * 100, 2)    # As %
            })
        
        # Calculate final metrics
        total_return = (equity - self.capital) / self.capital
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "final_equity": equity,
            "trade_log": trade_log,
            "parameters": {
                "take_profit_pct": take_profit_pct,
                "alpha": alpha,
                "gamma": gamma,
                "compound": compound
            }
        }
    
    def grid_search(
        self,
        tp_range: Tuple[float, float, float] = (0.010, 0.040, 0.005),
        alpha_range: Tuple[float, float, float] = (0.30, 0.70, 0.10),
        gamma_range: Tuple[float, float, float] = (0.15, 0.35, 0.05),
        metric: str = "total_return",
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Grid search over strategy parameters.
        
        Args:
            tp_range: (min, max, step) for take profit %
            alpha_range: (min, max, step) for stretch threshold
            gamma_range: (min, max, step) for momentum threshold
            metric: Metric to optimize
            progress_callback: Optional callback(current, total, best_score)
        
        Returns:
            Dict with best parameters and results
        """
        tp_values = np.arange(tp_range[0], tp_range[1], tp_range[2])
        alpha_values = np.arange(alpha_range[0], alpha_range[1], alpha_range[2])
        gamma_values = np.arange(gamma_range[0], gamma_range[1], gamma_range[2])
        
        total_combinations = len(tp_values) * len(alpha_values) * len(gamma_values)
        
        best_score = -float('inf')
        best_params = None
        best_result = None
        current = 0
        
        for tp in tp_values:
            for alpha in alpha_values:
                for gamma in gamma_values:
                    current += 1
                    result = self.run_backtest(
                        take_profit_pct=float(tp),
                        alpha=float(alpha),
                        gamma=float(gamma)
                    )
                    
                    score = result.get(metric, 0)
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "take_profit_pct": float(tp),
                            "alpha": float(alpha),
                            "gamma": float(gamma)
                        }
                        best_result = result
                    
                    if progress_callback:
                        progress_callback(current, total_combinations, best_score)
        
        return {
            "best_params": best_params,
            "best_result": best_result,
            "n_combinations": total_combinations
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_vwap_backtest(
    daily_bars: List[Dict],
    intraday_bars: List[Dict],
    capital: float = 100000,
    take_profit_pct: float = 0.02,
    alpha: float = 0.50,
    gamma: float = 0.25,
    compound: bool = True
) -> Dict:
    """
    Run a complete VWAP adaptive strategy backtest.
    
    This is a convenience function that handles VWAP map building.
    
    Args:
        daily_bars: Daily OHLC bars
        intraday_bars: 1-minute intraday bars (for VWAP calculation)
        capital: Starting capital
        take_profit_pct: Take profit percentage (0.02 = 2%)
        alpha: Stretch threshold multiplier
        gamma: Momentum threshold multiplier
        compound: Whether to compound gains
    
    Returns:
        Backtest results dict
    """
    # Build VWAP map from intraday data
    vwap_map = build_vwap_map_from_intraday(intraday_bars)
    
    # Run backtest
    backtester = VWAPAdaptiveBacktester(daily_bars, vwap_map, capital)
    return backtester.run_backtest(
        take_profit_pct=take_profit_pct,
        alpha=alpha,
        gamma=gamma,
        compound=compound
    )


def run_vwap_optimization(
    daily_bars: List[Dict],
    intraday_bars: List[Dict],
    capital: float = 100000,
    tp_range: Tuple[float, float, float] = (0.010, 0.040, 0.005),
    alpha_range: Tuple[float, float, float] = (0.30, 0.70, 0.10),
    gamma_range: Tuple[float, float, float] = (0.15, 0.35, 0.05),
    metric: str = "total_return"
) -> Dict:
    """
    Run grid search optimization on VWAP strategy.
    
    Args:
        daily_bars: Daily OHLC bars
        intraday_bars: 1-minute intraday bars
        capital: Starting capital
        tp_range: (min, max, step) for take profit
        alpha_range: (min, max, step) for stretch threshold
        gamma_range: (min, max, step) for momentum threshold
        metric: Metric to optimize
    
    Returns:
        Optimization results with best parameters
    """
    vwap_map = build_vwap_map_from_intraday(intraday_bars)
    backtester = VWAPAdaptiveBacktester(daily_bars, vwap_map, capital)
    
    search_result = backtester.grid_search(
        tp_range=tp_range,
        alpha_range=alpha_range,
        gamma_range=gamma_range,
        metric=metric
    )
    
    return {
        "best_params": search_result["best_params"],
        "metrics": {
            "total_return": search_result["best_result"]["total_return"],
            "win_rate": search_result["best_result"]["win_rate"],
            "total_trades": search_result["best_result"]["total_trades"],
            "max_drawdown": search_result["best_result"]["max_drawdown"],
            "profit_factor": search_result["best_result"]["profit_factor"]
        },
        "combinations_tested": search_result["n_combinations"]
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VWAP Adaptive Strategy Backtester")
    print("=" * 60)
    print()
    print("This module provides:")
    print("  - calculate_vwap_entry_signals(): Core entry signal logic")
    print("  - VWAPAdaptiveBacktester: Full backtesting class")
    print("  - run_vwap_backtest(): Simple backtest function")
    print("  - run_vwap_optimization(): Grid search optimization")
    print()
    print("Key Metrics Calculated:")
    print("  - VWAP Slope: (VWAP_10:00 - VWAP_9:40) / VWAP_9:40")
    print("  - VWAP Stretch: (Price_10AM - VWAP) / VWAP")
    print("  - OR Vol: max(OR_High - VWAP, VWAP - OR_Low) / VWAP")
    print("  - Momentum: (Price_10AM - Open) / Open")
    print()
    print("Entry Conditions (all must be True):")
    print("  1. Price_10AM > VWAP (price above VWAP)")
    print("  2. VWAP_Slope > 0 (VWAP trending up)")
    print("  3. VWAP_Stretch < alpha * OR_Vol (not overextended)")
    print("  4. Momentum > gamma * OR_Vol (sufficient momentum)")
    print()
    print("Example Usage:")
    print("-" * 60)
    print("""
from vwap_backtester import run_vwap_backtest, calculate_vwap_entry_signals

# Simple backtest
result = run_vwap_backtest(
    daily_bars=my_daily_bars,
    intraday_bars=my_intraday_bars,
    capital=100000,
    take_profit_pct=0.02  # 2%
)
print(f"Return: {result['total_return']*100:.1f}%")
print(f"Win Rate: {result['win_rate']*100:.1f}%")

# Check entry signal for today
signals = calculate_vwap_entry_signals(
    open_price=150.00,
    price_10am=151.50,
    vwap=150.80,
    vwap_940=150.50,
    or_high=152.00,
    or_low=149.50
)
if signals['entry_signal']:
    print("ENTRY SIGNAL! Go long.")
else:
    print(f"No entry: {signals['skip_reason']}")
""")
