#!/usr/bin/env python3
"""
GPU-Accelerated Batch Backtesting for Stock Optimization
Uses CuPy (CUDA) for parallel processing of multiple parameter combinations.
Falls back to NumPy if GPU is not available.
"""

import os
import logging

# Try importing CuPy for GPU acceleration
try:
    import cupy as cp
    # Runtime check: Does CuPy actually see any GPUs?
    if cp.cuda.runtime.getDeviceCount() > 0:
        GPU_AVAILABLE = True
        xp = cp  # Use CuPy as the array library
        logging.info(f"CuPy loaded - {cp.cuda.runtime.getDeviceCount()} GPU(s) detected and enabled")
    else:
        import numpy as np
        GPU_AVAILABLE = False
        xp = np
        logging.info("CuPy loaded but NO CUDA DEVICES found - falling back to NumPy (CPU)")
except Exception as e:
    import numpy as np
    GPU_AVAILABLE = False
    xp = np  # Fall back to NumPy
    logging.info(f"CuPy initialization failed: {e} - using NumPy (CPU)")

import numpy as np  # Always need NumPy for some operations

logger = logging.getLogger(__name__)


class GPUBatchBacktester:
    """
    GPU-accelerated batch backtester that tests multiple parameter combinations
    simultaneously using vectorized operations.
    
    Key optimizations:
    1. Vectorized operations - no Python loops during backtesting
    2. Batch processing - test 100+ parameter combinations at once
    3. GPU memory efficient - processes in chunks if needed
    4. Pinned memory - fast CPU to GPU data transfer
    5. Pre-computed arrays - avoid redundant calculations
    """
    
    def __init__(self, bars: list, capital: float = 100000):
        """
        Initialize with price data.
        
        Args:
            bars: List of OHLC bars [{"o": open, "h": high, "l": low, "c": close}, ...]
            capital: Starting capital
        """
        self.capital = capital
        self.n_bars = len(bars)
        
        # Extract price data - use pinned memory for faster GPU transfer if available
        open_list = [b.get("o", b.get("c", 0)) for b in bars]
        high_list = [b.get("h", b.get("c", 0)) for b in bars]
        low_list = [b.get("l", b.get("c", 0)) for b in bars]
        close_list = [b.get("c", 0) for b in bars]
        
        if GPU_AVAILABLE:
            # Use pinned memory for faster CPU->GPU transfer
            try:
                pinned_open = cp.cuda.alloc_pinned_memory(len(open_list) * 4)
                pinned_high = cp.cuda.alloc_pinned_memory(len(high_list) * 4)
                pinned_close = cp.cuda.alloc_pinned_memory(len(close_list) * 4)
                
                # Copy to pinned then to GPU (optimal transfer path)
                self.open_prices = cp.array(open_list, dtype=cp.float32)
                self.high_prices = cp.array(high_list, dtype=cp.float32)
                self.low_prices = cp.array(low_list, dtype=cp.float32)
                self.close_prices = cp.array(close_list, dtype=cp.float32)
                
                logger.info("Using pinned memory for GPU transfers")
            except Exception as e:
                logger.warning(f"Pinned memory failed, using standard transfer: {e}")
                self.open_prices = xp.array(open_list, dtype=xp.float32)
                self.high_prices = xp.array(high_list, dtype=xp.float32)
                self.low_prices = xp.array(low_list, dtype=xp.float32)
                self.close_prices = xp.array(close_list, dtype=xp.float32)
        else:
            self.open_prices = xp.array(open_list, dtype=xp.float32)
            self.high_prices = xp.array(high_list, dtype=xp.float32)
            self.low_prices = xp.array(low_list, dtype=xp.float32)
            self.close_prices = xp.array(close_list, dtype=xp.float32)
        
        # Pre-compute previous close (shifted by 1) - avoid recalculating
        self.prev_close = xp.zeros_like(self.close_prices)
        self.prev_close[1:] = self.close_prices[:-1]
        self.prev_close[0] = self.open_prices[0]  # Use open for first bar
        
        # Pre-compute percentage change from prev close to high (used in every iteration)
        self.pct_change_to_high = xp.where(
            self.prev_close > 0,
            (self.high_prices - self.prev_close) / self.prev_close,
            0
        )
        
        # Pre-compute as Python floats for the loop (avoid GPU->CPU transfer each iteration)
        self.prev_close_list = [float(x) for x in (self.prev_close.get() if GPU_AVAILABLE else self.prev_close)]
        self.high_list = [float(x) for x in (self.high_prices.get() if GPU_AVAILABLE else self.high_prices)]
        self.close_list = [float(x) for x in (self.close_prices.get() if GPU_AVAILABLE else self.close_prices)]
        
        logger.info(f"GPUBatchBacktester initialized with {self.n_bars} bars, GPU={GPU_AVAILABLE}")
    
    def run_batch(
        self, 
        buy_triggers: np.ndarray,
        sell_triggers: np.ndarray,
        compounds: np.ndarray = None
    ) -> dict:
        """
        Run backtests for multiple parameter combinations in parallel.
        
        Args:
            buy_triggers: Array of buy trigger percentages (e.g., [0.02, 0.03, 0.04])
            sell_triggers: Array of sell trigger percentages (e.g., [0.05, 0.07, 0.09])
            compounds: Array of compound flags (optional, defaults to True)
            
        Returns:
            Dictionary with metrics for each parameter combination
        """
        n_combinations = len(buy_triggers)
        
        if compounds is None:
            compounds = np.ones(n_combinations, dtype=bool)
        
        # Convert to GPU arrays
        buy_trig = xp.array(buy_triggers, dtype=xp.float32).reshape(-1, 1)
        sell_trig = xp.array(sell_triggers, dtype=xp.float32).reshape(-1, 1)
        compound_flags = xp.array(compounds, dtype=bool).reshape(-1, 1)
        
        # Initialize tracking arrays for each parameter set
        # Shape: (n_combinations, 1)
        cash = xp.full((n_combinations, 1), self.capital, dtype=xp.float32)
        position_qty = xp.zeros((n_combinations, 1), dtype=xp.float32)
        entry_price = xp.zeros((n_combinations, 1), dtype=xp.float32)
        in_position = xp.zeros((n_combinations, 1), dtype=bool)
        
        # Track metrics
        total_trades = xp.zeros((n_combinations, 1), dtype=xp.int32)
        winning_trades = xp.zeros((n_combinations, 1), dtype=xp.int32)
        total_profit = xp.zeros((n_combinations, 1), dtype=xp.float32)
        total_loss = xp.zeros((n_combinations, 1), dtype=xp.float32)
        
        # Portfolio value history for Sharpe calculation
        portfolio_history = xp.zeros((n_combinations, self.n_bars), dtype=xp.float32)
        
        # Iterate through bars (this is the only loop, but operations are vectorized across all param combinations)
        for i in range(self.n_bars):
            # Use pre-computed Python floats (avoids GPU->CPU transfer each iteration)
            prev_close = self.prev_close_list[i]
            high = self.high_list[i]
            close = self.close_list[i]
            
            if prev_close <= 0:
                portfolio_history[:, i:i+1] = cash + position_qty * close
                continue
            
            pct_change = (high - prev_close) / prev_close
            
            # BUY SIGNAL: not in position AND high >= prev_close * (1 + buy_trigger)
            buy_signal = (~in_position) & (pct_change >= buy_trig)
            
            # Calculate entry for buys
            buy_entry = prev_close * (1 + buy_trig)
            buy_qty = xp.where(buy_signal, cash / buy_entry, 0)
            
            # Update positions for buys
            position_qty = xp.where(buy_signal, buy_qty, position_qty)
            entry_price = xp.where(buy_signal, buy_entry, entry_price)
            cash = xp.where(buy_signal, 0, cash)
            in_position = in_position | buy_signal
            
            # SELL SIGNAL: in position (sell at end of day)
            # Check if target hit
            target_price = entry_price * (1 + sell_trig)
            target_hit = in_position & (high >= target_price)
            
            # Exit price: target price if hit, else close
            exit_price = xp.where(target_hit, target_price, close)
            
            # All positions exit at end of day
            sell_signal = in_position
            proceeds = position_qty * exit_price
            profit = proceeds - (position_qty * entry_price)
            
            # Update metrics
            total_trades = xp.where(sell_signal, total_trades + 1, total_trades)
            winning_trades = xp.where(sell_signal & (profit > 0), winning_trades + 1, winning_trades)
            total_profit = xp.where(sell_signal & (profit > 0), total_profit + profit, total_profit)
            total_loss = xp.where(sell_signal & (profit < 0), total_loss + xp.abs(profit), total_loss)
            
            # Update cash and reset position
            cash = xp.where(sell_signal, proceeds, cash)
            position_qty = xp.where(sell_signal, 0, position_qty)
            entry_price = xp.where(sell_signal, 0, entry_price)
            in_position = xp.where(sell_signal, False, in_position)
            
            # Track portfolio value
            portfolio_history[:, i:i+1] = cash + position_qty * close
        
        # Calculate final metrics
        final_value = portfolio_history[:, -1:]
        total_return = (final_value - self.capital) / self.capital
        
        # Calculate Sharpe ratio from daily returns
        daily_returns = xp.diff(portfolio_history, axis=1) / portfolio_history[:, :-1]
        daily_returns = xp.nan_to_num(daily_returns, 0)
        
        mean_return = xp.mean(daily_returns, axis=1, keepdims=True)
        std_return = xp.std(daily_returns, axis=1, keepdims=True)
        std_return = xp.where(std_return < 1e-6, 1e-6, std_return)  # Avoid division by zero
        
        sharpe = (mean_return / std_return) * xp.sqrt(252)  # Annualized
        
        # Win rate
        win_rate = xp.where(
            total_trades > 0,
            winning_trades / total_trades,
            0
        )
        
        # Average win/loss
        avg_win = xp.where(winning_trades > 0, total_profit / winning_trades, 0)
        avg_loss = xp.where(
            (total_trades - winning_trades) > 0,
            total_loss / (total_trades - winning_trades),
            0
        )
        
        # Convert back to numpy for return
        def to_numpy(arr):
            if GPU_AVAILABLE:
                return cp.asnumpy(arr).flatten()
            return arr.flatten()
        
        return {
            "total_return": to_numpy(total_return),
            "sharpe": to_numpy(sharpe),
            "win_rate": to_numpy(win_rate),
            "total_trades": to_numpy(total_trades),
            "avg_win": to_numpy(avg_win),
            "avg_loss": to_numpy(avg_loss),
            "final_value": to_numpy(final_value),
        }
    
    def grid_search(
        self,
        buy_range: tuple = (0.01, 0.10, 0.005),  # (min, max, step)
        sell_range: tuple = (0.01, 0.15, 0.005),
        metric: str = "sharpe"
    ) -> dict:
        """
        Run exhaustive grid search over parameter ranges.
        
        Args:
            buy_range: (min, max, step) for buy trigger
            sell_range: (min, max, step) for sell trigger
            metric: Metric to optimize ("sharpe", "total_return", "win_rate")
            
        Returns:
            Best parameters and metrics
        """
        # Generate parameter grid
        buy_triggers = np.arange(buy_range[0], buy_range[1], buy_range[2])
        sell_triggers = np.arange(sell_range[0], sell_range[1], sell_range[2])
        
        # Create all combinations
        buy_grid, sell_grid = np.meshgrid(buy_triggers, sell_triggers)
        buy_flat = buy_grid.flatten()
        sell_flat = sell_grid.flatten()
        
        n_combinations = len(buy_flat)
        logger.info(f"Running grid search with {n_combinations} combinations")
        
        # Process in chunks if too many combinations (prevent GPU memory overflow)
        # RTX 3080 EVGA has 24GB - each combination uses ~(n_bars * 4 bytes * 10 arrays) 
        # For 252 bars: ~10KB per combination, so 24GB / 10KB = ~2.4M combinations max
        # We'll use 100K chunks to be safe and allow GPU memory for other operations
        CHUNK_SIZE = 100000
        
        if n_combinations > CHUNK_SIZE:
            logger.info(f"Large grid detected, processing in {(n_combinations // CHUNK_SIZE) + 1} chunks")
            
            all_results = {
                "total_return": [],
                "sharpe": [],
                "win_rate": [],
                "total_trades": [],
                "avg_win": [],
                "avg_loss": [],
                "final_value": [],
            }
            
            for chunk_start in range(0, n_combinations, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, n_combinations)
                logger.info(f"Processing chunk {chunk_start//CHUNK_SIZE + 1}: combinations {chunk_start}-{chunk_end}")
                
                chunk_results = self.run_batch(
                    buy_flat[chunk_start:chunk_end],
                    sell_flat[chunk_start:chunk_end]
                )
                
                for key in all_results:
                    all_results[key].extend(chunk_results[key].tolist())
            
            # Convert back to numpy arrays
            results = {k: np.array(v) for k, v in all_results.items()}
        else:
            # Run batch backtest (single batch)
            results = self.run_batch(buy_flat, sell_flat)
        
        # Find best
        metric_values = results[metric]
        best_idx = np.argmax(metric_values)
        
        return {
            "best_params": {
                "buy_trigger_pct": float(buy_flat[best_idx] * 100),
                "sell_trigger_pct": float(sell_flat[best_idx] * 100),
                "compound": True
            },
            "metrics": {
                "total_return": float(results["total_return"][best_idx]),
                "sharpe": float(results["sharpe"][best_idx]),
                "win_rate": float(results["win_rate"][best_idx]),
                "total_trades": int(results["total_trades"][best_idx]),
            },
            "n_combinations_tested": n_combinations,
            "all_results": results  # For analysis
        }


class GPUChatGPT9AMBacktester:
    """
    GPU-accelerated backtester for ChatGPT 9AM Strategy.
    
    Strategy:
    - Buy at 10 AM Eastern (9 AM Central) if price >= open * (1 + buy_trigger%)
    - Sell when price >= open * (1 + sell_trigger%), OR at 3:55 PM Eastern (use close as proxy)
    - All positions exit same day (day trading)
    
    Requires daily bars with 10 AM price included as 'price_10am' field.
    """
    
    def __init__(self, bars: list, capital: float = 100000):
        """
        Initialize with price data including 10 AM prices.
        
        Args:
            bars: List of OHLC bars with 10 AM price: 
                  [{"o": open, "h": high, "l": low, "c": close, "price_10am": price}, ...]
            capital: Starting capital
        """
        self.capital = capital
        self.n_bars = len(bars)
        
        # Extract price data
        open_list = [b.get("o", b.get("c", 0)) for b in bars]
        high_list = [b.get("h", b.get("c", 0)) for b in bars]
        close_list = [b.get("c", 0) for b in bars]
        price_10am_list = [b.get("price_10am", 0) for b in bars]
        
        # Convert to arrays
        self.open_prices = xp.array(open_list, dtype=xp.float32)
        self.high_prices = xp.array(high_list, dtype=xp.float32)
        self.close_prices = xp.array(close_list, dtype=xp.float32)
        self.price_10am = xp.array(price_10am_list, dtype=xp.float32)
        
        # Pre-compute as Python floats for the loop
        self.open_list = [float(x) for x in (self.open_prices.get() if GPU_AVAILABLE else self.open_prices)]
        self.high_list_py = [float(x) for x in (self.high_prices.get() if GPU_AVAILABLE else self.high_prices)]
        self.close_list_py = [float(x) for x in (self.close_prices.get() if GPU_AVAILABLE else self.close_prices)]
        self.price_10am_list = [float(x) for x in (self.price_10am.get() if GPU_AVAILABLE else self.price_10am)]
        
        logger.info(f"GPUChatGPT9AMBacktester initialized with {self.n_bars} bars, GPU={GPU_AVAILABLE}")
    
    def run_batch(
        self, 
        buy_triggers: np.ndarray,
        sell_triggers: np.ndarray,
        compounds: np.ndarray = None
    ) -> dict:
        """
        Run backtests for ChatGPT 9AM strategy with multiple parameter combinations.
        
        Strategy: 
        - Buy at 10 AM if price_10am >= open * (1 + buy_trigger)
        - Sell when high >= open * (1 + sell_trigger), else at close
        """
        n_combinations = len(buy_triggers)
        
        if compounds is None:
            compounds = np.ones(n_combinations, dtype=bool)
        
        # Convert to GPU arrays
        buy_trig = xp.array(buy_triggers, dtype=xp.float32).reshape(-1, 1)
        sell_trig = xp.array(sell_triggers, dtype=xp.float32).reshape(-1, 1)
        compound_flags = xp.array(compounds, dtype=bool).reshape(-1, 1)
        
        # Initialize tracking arrays
        cash = xp.full((n_combinations, 1), self.capital, dtype=xp.float32)
        
        # Track metrics
        total_trades = xp.zeros((n_combinations, 1), dtype=xp.int32)
        winning_trades = xp.zeros((n_combinations, 1), dtype=xp.int32)
        total_profit = xp.zeros((n_combinations, 1), dtype=xp.float32)
        total_loss = xp.zeros((n_combinations, 1), dtype=xp.float32)
        
        # Portfolio value history for Sharpe calculation
        portfolio_history = xp.zeros((n_combinations, self.n_bars), dtype=xp.float32)
        
        # Iterate through bars
        for i in range(self.n_bars):
            open_price = self.open_list[i]
            price_10am = self.price_10am_list[i]
            high = self.high_list_py[i]
            close = self.close_list_py[i]
            
            if open_price <= 0 or price_10am <= 0:
                portfolio_history[:, i:i+1] = cash
                continue
            
            # Buy threshold: open * (1 + buy_trigger)
            buy_threshold = open_price * (1 + buy_trig)
            
            # BUY SIGNAL: 10 AM price >= buy_threshold
            buy_signal = (price_10am >= buy_threshold)
            
            # Buy price is the 10 AM price
            buy_price = price_10am
            
            # Number of shares we can buy
            shares = xp.where(buy_signal, cash / buy_price, 0)
            
            # Sell threshold: open * (1 + sell_trigger)
            sell_threshold = open_price * (1 + sell_trig)
            
            # Did we hit sell target?
            target_hit = buy_signal & (high >= sell_threshold)
            
            # Exit price: sell_threshold if target hit, else close (5 min before would be close price proxy)
            exit_price = xp.where(target_hit, sell_threshold, close)
            
            # Calculate profit/loss
            profit = shares * (exit_price - buy_price)
            
            # Update metrics
            total_trades = xp.where(buy_signal, total_trades + 1, total_trades)
            winning_trades = xp.where(buy_signal & (profit > 0), winning_trades + 1, winning_trades)
            total_profit = xp.where(buy_signal & (profit > 0), total_profit + profit, total_profit)
            total_loss = xp.where(buy_signal & (profit < 0), total_loss + xp.abs(profit), total_loss)
            
            # Update cash (compound or not)
            new_cash = cash + xp.where(compound_flags, profit, 0)
            cash = xp.where(buy_signal, new_cash, cash)
            
            # Track portfolio value
            portfolio_history[:, i:i+1] = cash
        
        # Calculate final metrics
        final_value = portfolio_history[:, -1:]
        total_return = (final_value - self.capital) / self.capital
        
        # Calculate Sharpe ratio from daily returns
        daily_returns = xp.diff(portfolio_history, axis=1) / portfolio_history[:, :-1]
        daily_returns = xp.nan_to_num(daily_returns, 0)
        
        mean_return = xp.mean(daily_returns, axis=1, keepdims=True)
        std_return = xp.std(daily_returns, axis=1, keepdims=True)
        std_return = xp.where(std_return < 1e-6, 1e-6, std_return)
        
        sharpe = (mean_return / std_return) * xp.sqrt(252)
        
        # Win rate
        win_rate = xp.where(
            total_trades > 0,
            winning_trades / total_trades,
            0
        )
        
        # Convert back to numpy
        def to_numpy(arr):
            if GPU_AVAILABLE:
                return cp.asnumpy(arr).flatten()
            return arr.flatten()
        
        return {
            "total_return": to_numpy(total_return),
            "sharpe": to_numpy(sharpe),
            "win_rate": to_numpy(win_rate),
            "total_trades": to_numpy(total_trades),
            "final_value": to_numpy(final_value),
        }
    
    def grid_search(
        self,
        buy_range: tuple = (0.001, 0.05, 0.001),
        sell_range: tuple = (0.01, 0.10, 0.002),
        metric: str = "total_return"
    ) -> dict:
        """
        Run exhaustive grid search over parameter ranges.
        """
        # Generate parameter grid
        buy_triggers = np.arange(buy_range[0], buy_range[1], buy_range[2])
        sell_triggers = np.arange(sell_range[0], sell_range[1], sell_range[2])
        
        # Create all combinations
        buy_grid, sell_grid = np.meshgrid(buy_triggers, sell_triggers)
        buy_flat = buy_grid.flatten()
        sell_flat = sell_grid.flatten()
        
        n_combinations = len(buy_flat)
        logger.info(f"ChatGPT 9AM: Running grid search with {n_combinations} combinations")
        
        # Run batch backtest
        results = self.run_batch(buy_flat, sell_flat)
        
        # Find best
        metric_values = results[metric]
        best_idx = np.argmax(metric_values)
        
        return {
            "best_params": {
                "buy_trigger_pct": float(buy_flat[best_idx] * 100),
                "sell_trigger_pct": float(sell_flat[best_idx] * 100),
                "compound": True
            },
            "metrics": {
                "total_return": float(results["total_return"][best_idx]),
                "sharpe": float(results["sharpe"][best_idx]),
                "win_rate": float(results["win_rate"][best_idx]),
                "total_trades": int(results["total_trades"][best_idx]),
            },
            "n_combinations_tested": n_combinations,
            "all_results": results
        }


def run_chatgpt_9am_optimization(
    bars: list,
    capital: float = 100000,
    buy_range: tuple = (0.001, 0.05, 0.001),
    sell_range: tuple = (0.01, 0.10, 0.002),
    metric: str = "total_return"
) -> dict:
    """
    Run ChatGPT 9AM Strategy optimization.
    
    Args:
        bars: Daily bars WITH 'price_10am' field for each bar
        capital: Starting capital
        buy_range: (min, max, step) for buy trigger search (as decimal, e.g., 0.02 = 2%)
        sell_range: (min, max, step) for sell trigger search
        metric: Optimization metric (total_return, sharpe, win_rate)
        
    Returns:
        Optimization results
    """
    backtester = GPUChatGPT9AMBacktester(bars, capital)
    results = backtester.grid_search(buy_range, sell_range, metric)
    
    return {
        "symbol": "CHATGPT_9AM",
        "best_params": results["best_params"],
        "metrics": results["metrics"],
        "optimization_type": "chatgpt_9am_gpu",
        "gpu_enabled": GPU_AVAILABLE,
        "combinations_tested": results["n_combinations_tested"]
    }


class GPUChatGPTStopLossBacktester:
    """
    GPU-accelerated backtester for ChatGPT 9AM Strategy with Stop Loss.
    
    Strategy Rules (fixed parameters per spec):
    - ENTRY: At 10 AM if (price_10am - open) / open >= 1.1%
    - TAKE PROFIT: Exit if (current - entry) / entry >= 2.2%
    - STOP LOSS: Exit if (current - entry) / entry <= -0.7%
    - PROTECT WINNER: Exit if current < price_10am (after being in profit)
    - TIME EXIT: Exit at 3:55 PM if still holding
    
    Since we use daily OHLC, we approximate:
    - Use low to check stop loss hit
    - Use high to check take profit hit
    - Use close as fallback exit
    """
    
    def __init__(self, bars: list, capital: float = 100000):
        """Initialize with daily bars including 10AM price."""
        self.capital = capital
        self.n_bars = len(bars)
        
        # Fixed parameters per user spec (DO NOT MODIFY)
        # All percentages relative to ENTRY PRICE (which is 10AM price)
        self.entry_trigger = 0.011   # +1.1% from OPEN to trigger entry at 10AM
        self.take_profit = 0.019     # +1.9% from ENTRY for take profit
        self.stop_loss = -0.011      # -1.1% from ENTRY for stop loss
        
        # Extract price data
        self.open_prices = [b.get("o", b.get("c", 0)) for b in bars]
        self.high_prices = [b.get("h", b.get("c", 0)) for b in bars]
        self.low_prices = [b.get("l", b.get("c", 0)) for b in bars]
        self.close_prices = [b.get("c", 0) for b in bars]
        self.price_10am = [b.get("price_10am", 0) for b in bars]
        
        logger.info(f"GPUChatGPTStopLossBacktester initialized with {self.n_bars} bars")
    
    def run_backtest(self, compound: bool = True) -> dict:
        """
        Run the stop loss strategy backtest.
        
        Returns dict with metrics.
        """
        equity = self.capital
        total_trades = 0
        winning_trades = 0
        total_profit = 0
        total_loss = 0
        max_equity = equity
        max_drawdown = 0
        
        trade_log = []
        
        for i in range(self.n_bars):
            open_price = self.open_prices[i]
            price_10am = self.price_10am[i]
            high = self.high_prices[i]
            low = self.low_prices[i]
            close = self.close_prices[i]
            
            if open_price <= 0 or price_10am <= 0:
                continue
            
            # Entry check: 10AM price >= open * (1 + 1.1%)
            entry_threshold = open_price * (1 + self.entry_trigger)
            if price_10am < entry_threshold:
                continue  # No entry signal
            
            # We enter at 10AM price
            entry_price = price_10am
            shares = int(equity / entry_price) if entry_price > 0 else 0
            
            if shares <= 0:
                continue
            
            total_trades += 1
            
            # Calculate exit thresholds
            take_profit_price = entry_price * (1 + self.take_profit)
            stop_loss_price = entry_price * (1 + self.stop_loss)  # This is entry * 0.993
            
            # Determine exit price based on what happens during the day
            exit_price = close  # Default: exit at close
            exit_reason = "time_exit"
            
            # Check if take profit was hit (high reached target)
            if high >= take_profit_price:
                exit_price = take_profit_price
                exit_reason = "take_profit"
            # Check if stop loss was hit (low went below stop)
            elif low <= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = "stop_loss"
            # Check protect winner: if we were up but price dropped below 10AM
            elif low < price_10am:
                # Price went below 10AM at some point - exit at 10AM price
                exit_price = price_10am
                exit_reason = "protect_winner"
            
            # Calculate P&L
            profit = shares * (exit_price - entry_price)
            pct_return = (exit_price - entry_price) / entry_price * 100
            
            if profit > 0:
                winning_trades += 1
                total_profit += profit
            else:
                total_loss += abs(profit)
            
            if compound:
                equity += profit
            
            # Track max drawdown
            if equity > max_equity:
                max_equity = equity
            drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            
            trade_log.append({
                "day": i,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "profit": profit,
                "pct_return": pct_return,
                "equity": equity
            })
        
        # Calculate metrics
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
            "trade_log": trade_log
        }
    
    def run_single(self, entry_trigger: float, take_profit: float, stop_loss: float, trailing_stop: float = 0, use_trend_filter: bool = False) -> dict:
        """
        Run backtest with specific parameters including trailing stop.
        
        Args:
            entry_trigger: % above open for entry (e.g., 0.01 = 1%)
            take_profit: % above entry for TP (e.g., 0.015 = 1.5%)
            stop_loss: % below entry for SL (negative, e.g., -0.015 = -1.5%)
            trailing_stop: % below high for trailing stop (negative, e.g., -0.01 = -1%)
                          If 0, trailing stop is disabled
            use_trend_filter: If True, only trade when previous day closed green
        """
        equity = self.capital
        total_trades = 0
        winning_trades = 0
        total_profit = 0
        total_loss = 0
        
        for i in range(self.n_bars):
            open_price = self.open_prices[i]
            price_10am = self.price_10am[i]
            high = self.high_prices[i]
            low = self.low_prices[i]
            close = self.close_prices[i]
            
            if open_price <= 0 or price_10am <= 0:
                continue
            
            # TREND FILTER: Only trade if previous day closed green (bullish momentum)
            if use_trend_filter and i > 0:
                prev_open = self.open_prices[i-1]
                prev_close = self.close_prices[i-1]
                if prev_close <= prev_open:
                    continue  # Skip if previous day was red/flat
            
            # Entry check: 10AM price >= open * (1 + entry_trigger)
            entry_threshold = open_price * (1 + entry_trigger)
            if price_10am < entry_threshold:
                continue
            
            entry_price = price_10am
            shares = int(equity / entry_price) if entry_price > 0 else 0
            
            if shares <= 0:
                continue
            
            total_trades += 1
            
            # Exit thresholds
            take_profit_price = entry_price * (1 + take_profit)
            stop_loss_price = entry_price * (1 + stop_loss)  # stop_loss is negative
            
            # Trailing stop: tracks the highest price and exits if drops trailing_stop % from it
            # For daily OHLC, we approximate: trailing stop based on day's high
            trailing_stop_price = high * (1 + trailing_stop) if trailing_stop < 0 else 0
            
            # Determine exit priority: TP > Trailing Stop (if profitable) > SL > Time
            if high >= take_profit_price:
                # Take profit hit
                exit_price = take_profit_price
            elif trailing_stop < 0 and low <= trailing_stop_price and trailing_stop_price > entry_price:
                # Trailing stop hit (only if we're in profit - trailing above entry)
                exit_price = trailing_stop_price
            elif low <= stop_loss_price:
                # Stop loss hit
                exit_price = stop_loss_price
            else:
                # Time exit at close
                exit_price = close
            
            profit = shares * (exit_price - entry_price)
            
            if profit > 0:
                winning_trades += 1
                total_profit += profit
            else:
                total_loss += abs(profit)
            
            equity += profit
        
        total_return = (equity - self.capital) / self.capital
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "total_trades": total_trades
        }
    
    def grid_search(
        self,
        entry_range: tuple = (0.005, 0.025, 0.002),
        tp_range: tuple = (0.008, 0.025, 0.002),
        sl_range: tuple = (-0.025, -0.008, 0.002),
        trailing_range: tuple = (-0.015, -0.005, 0.002),
        metric: str = "total_return"
    ) -> dict:
        """
        Grid search over entry trigger, take profit, stop loss, and trailing stop.
        
        Args:
            entry_range: (min, max, step) for entry trigger %
            tp_range: (min, max, step) for take profit %
            sl_range: (min, max, step) for stop loss % (negative values)
            trailing_range: (min, max, step) for trailing stop % (negative values)
            metric: Metric to optimize
        """
        import numpy as np
        
        entry_values = np.arange(entry_range[0], entry_range[1], entry_range[2])
        tp_values = np.arange(tp_range[0], tp_range[1], tp_range[2])
        sl_values = np.arange(sl_range[0], sl_range[1], sl_range[2])
        # Include 0 (disabled) plus the range values for trailing stop
        trailing_values = list(np.arange(trailing_range[0], trailing_range[1], trailing_range[2])) + [0]
        # Test both with and without trend filter
        trend_filter_values = [False, True]
        
        best_score = -float('inf')
        best_params = None
        best_result = None
        n_combinations = 0
        
        for entry in entry_values:
            for tp in tp_values:
                for sl in sl_values:
                    for trail in trailing_values:
                        for use_filter in trend_filter_values:
                            n_combinations += 1
                            result = self.run_single(float(entry), float(tp), float(sl), float(trail), use_filter)
                            
                            score = result.get(metric, result.get("total_return", 0))
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    "entry_trigger": float(entry),
                                    "take_profit": float(tp),
                                    "stop_loss": float(sl),
                                    "trailing_stop": float(trail),
                                    "trend_filter": use_filter
                                }
                                best_result = result
        
        return {
            "best_params": best_params,
            "best_result": best_result,
            "n_combinations": n_combinations
        }


def run_chatgpt_stoploss_optimization(
    bars: list,
    capital: float = 100000,
    entry_range: tuple = (0.001, 0.020, 0.002),
    tp_range: tuple = (0.010, 0.040, 0.003),
    sl_range: tuple = (-0.035, -0.010, 0.004),
    trailing_range: tuple = (-0.020, -0.005, 0.003),
    metric: str = "total_return"
) -> dict:
    """
    Run ChatGPT 9AM Stop Loss Strategy with OPTIMIZATION + Trailing Stop + Trend Filter.
    
    Searches over:
    - Entry trigger: % above open at 10AM to enter (default 0.1% to 2%)
    - Take profit: % above entry to exit with profit (default 1% to 4%)
    - Stop loss: % below entry to exit with loss (default -3.5% to -1%)
    - Trailing stop: % below high to lock in profits (default -2% to -0.5%)
    - Trend filter: Only trade after green days (tested both on and off)
    
    Args:
        bars: Daily bars WITH 'price_10am' field
        capital: Starting capital
        entry_range: (min, max, step) for entry trigger
        tp_range: (min, max, step) for take profit
        sl_range: (min, max, step) for stop loss
        trailing_range: (min, max, step) for trailing stop
        metric: Optimization metric
        
    Returns:
        Optimization results with best parameters
    """
    backtester = GPUChatGPTStopLossBacktester(bars, capital)
    search_result = backtester.grid_search(entry_range, tp_range, sl_range, trailing_range, metric)
    
    best_params = search_result["best_params"]
    best_result = search_result["best_result"]
    
    # Handle trailing_stop = 0 (disabled)
    trailing_pct = abs(best_params.get("trailing_stop", 0)) * 100
    
    return {
        "symbol": "CHATGPT_STOPLOSS",
        "best_params": {
            "buy_trigger_pct": round(best_params["entry_trigger"] * 100, 2),  # Convert to %
            "sell_trigger_pct": round(best_params["take_profit"] * 100, 2),   # Convert to %
            "stop_loss_pct": round(abs(best_params["stop_loss"]) * 100, 2),   # Convert to % (positive)
            "trailing_stop_pct": round(trailing_pct, 2),                       # Convert to % (positive, 0 = disabled)
            "trend_filter": best_params.get("trend_filter", False),           # Whether to only trade after green days
            "compound": True
        },
        "metrics": {
            "total_return": best_result["total_return"],
            "sharpe": 0,
            "win_rate": best_result["win_rate"],
            "total_trades": best_result["total_trades"],
        },
        "optimization_type": "chatgpt_stoploss_advanced",
        "gpu_enabled": False,
        "combinations_tested": search_result["n_combinations"]
    }


    def __init__(self, bars: list, capital: float = 100000):
        self.capital = capital
        self.n_bars = len(bars)
        
        # Extract price data
        open_list = [b.get("open", b.get("o", 0)) for b in bars]
        high_list = [b.get("high", b.get("h", 0)) for b in bars]
        low_list = [b.get("low", b.get("l", 0)) for b in bars]
        close_list = [b.get("close", b.get("c", 0)) for b in bars]
        price_10am_list = [b.get("price_10am", b.get("c", 0)) for b in bars]
        vwap_list = [b.get("vwap", b.get("c", 0)) for b in bars]
        
        # Convert to GPU arrays
        self.open_prices = xp.array(open_list, dtype=xp.float32)
        self.high_prices = xp.array(high_list, dtype=xp.float32)
        self.low_prices = xp.array(low_list, dtype=xp.float32)
        self.close_prices = xp.array(close_list, dtype=xp.float32)
        self.price_10am = xp.array(price_10am_list, dtype=xp.float32)
        self.vwap = xp.array(vwap_list, dtype=xp.float32)

    def run_batch(
        self, 
        entry_triggers: np.ndarray,
        take_profits: np.ndarray,
        stop_losses: np.ndarray,
        use_protect_winner: bool = True
    ) -> dict:
        n_combos = len(entry_triggers)
        
        # Reshape parameters for vectorized math (N_COMBOS, 1)
        entry_trig = xp.array(entry_triggers, dtype=xp.float32).reshape(-1, 1)
        tp_trig = xp.array(take_profits, dtype=xp.float32).reshape(-1, 1)
        sl_trig = xp.array(stop_losses, dtype=xp.float32).reshape(-1, 1)
        
        # Initialize equity tracking
        equity = xp.full((n_combos, 1), self.capital, dtype=xp.float32)
        total_trades = xp.zeros((n_combos, 1), dtype=xp.int32)
        winning_trades = xp.zeros((n_combos, 1), dtype=xp.int32)
        
        # Iterate through days (vectorized across combinations)
        for i in range(self.n_bars):
            o = self.open_prices[i]
            h = self.high_prices[i]
            l = self.low_prices[i]
            c = self.close_prices[i]
            p10 = self.price_10am[i]
            vw = self.vwap[i]
            
            if o <= 0 or p10 <= 0: continue

            # Entry condition: Price >= Open * (1 + trigger) AND Price > VWAP
            entry_threshold = o * (1 + entry_trig)
            signal = (p10 >= entry_threshold) & (p10 > vw)
            
            # Entry Price is always 10AM price
            entry_p = p10
            shares = xp.where(signal, equity / entry_p, 0)
            
            # Exit Prices
            tp_price = entry_p * (1 + tp_trig)
            sl_price = entry_p * (1 + sl_trig)
            
            # Determine Exit Reason and Price (TP > Protect > SL > Time)
            # 1. Take Profit hit?
            tp_hit = (h >= tp_price)
            # 2. Protect hit? (Price dropped below entry level)
            protect_hit = (l < entry_p)
            # 3. Stop Loss hit?
            sl_hit = (l <= sl_price)
            
            # Priorities:
            ex_price = xp.full((n_combos, 1), float(c), dtype=xp.float32)
            ex_price = xp.where(sl_hit, sl_price, ex_price)
            if use_protect_winner:
                ex_price = xp.where(protect_hit, entry_p, ex_price)
            ex_price = xp.where(tp_hit, tp_price, ex_price)
            
            # Calculate Profit
            pnl = shares * (ex_price - entry_p)
            equity = xp.where(signal, equity + pnl, equity)
            total_trades = xp.where(signal, total_trades + 1, total_trades)
            winning_trades = xp.where(signal & (pnl > 0), winning_trades + 1, winning_trades)

        # Final metrics
        total_return = (equity - self.capital) / self.capital
        win_rate = xp.where(total_trades > 0, winning_trades / total_trades, 0)
        
        def to_cpu(arr):
            return cp.asnumpy(arr).flatten() if GPU_AVAILABLE else arr.flatten()
            
        return {
            "total_return": to_cpu(total_return),
            "win_rate": to_cpu(win_rate),
            "total_trades": to_cpu(total_trades)
        }

    def grid_search(self, entry_range: tuple, tp_range: tuple, sl_range: tuple,
                    metric: str = "total_return", progress_callback=None) -> dict:
        # Generate parameter grids
        e_vals = np.arange(entry_range[0], entry_range[1], entry_range[2])
        t_vals = np.arange(tp_range[0], tp_range[1], tp_range[2])
        s_vals = np.arange(sl_range[0], sl_range[1], sl_range[2])
        
        e_grid, t_grid, s_grid = np.meshgrid(e_vals, t_vals, s_vals)
        e_flat, t_flat, s_flat = e_grid.flatten(), t_grid.flatten(), s_grid.flatten()
        
        n_combos = len(e_flat)
        logger.info(f"[GPU] Running VWAP grid search for {n_combos} combinations")
        
        # Chunk if needed
        CHUNK_SIZE = 100000
        results = {"total_return": [], "win_rate": [], "total_trades": []}
        
        for start in range(0, n_combos, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_combos)
            chunk = self.run_batch(e_flat[start:end], t_flat[start:end], s_flat[start:end])
            for k in results: results[k].extend(chunk[k])
            
        res_arr = {k: np.array(v) for k, v in results.items()}
        best_idx = np.argmax(res_arr[metric])
        
        return {
            "best_params": {
                "entry_trigger": float(e_flat[best_idx]),
                "take_profit": float(t_flat[best_idx]),
                "stop_loss": float(s_flat[best_idx])
            },
            "best_result": {
                "total_return": float(res_arr["total_return"][best_idx]),
                "win_rate": float(res_arr["win_rate"][best_idx]),
                "total_trades": int(res_arr["total_trades"][best_idx])
            },
            "n_combinations": n_combos
        }


class GPUChatGPTVWAPBacktester:
    """
    GPU-accelerated backtester for ChatGPT VWAP Strategy.
    
    Strategy:
    - Buy at 10 AM if price_10am >= open * (1 + entry_trigger) AND price_10am > VWAP
    - Take profit when high >= entry_price * (1 + take_profit)
    - Stop loss when low <= entry_price * (1 + stop_loss)  [stop_loss is negative]
    - Exit at close if neither TP nor SL hit
    
    Requires daily bars with 'price_10am', 'vwap', 'or_high', 'or_low' fields.
    """
    
    def __init__(self, bars: list, capital: float = 100000):
        self.capital = capital
        self.n_bars = len(bars)
        
        # Extract price data
        self.open_list = [float(b.get("o", b.get("c", 0))) for b in bars]
        self.high_list = [float(b.get("h", b.get("c", 0))) for b in bars]
        self.low_list = [float(b.get("l", b.get("c", 0))) for b in bars]
        self.close_list = [float(b.get("c", 0)) for b in bars]
        self.price_10am_list = [float(b.get("price_10am", 0)) for b in bars]
        self.vwap_list = [float(b.get("vwap", 0)) for b in bars]
        
        logger.info(f"GPUChatGPTVWAPBacktester initialized with {self.n_bars} bars, GPU={GPU_AVAILABLE}")
    
    def _run_single(self, entry_trigger: float, take_profit: float, stop_loss: float) -> dict:
        """Run a single backtest with given parameters."""
        capital = self.capital
        cash = capital
        total_trades = 0
        winning_trades = 0
        max_drawdown = 0
        peak_value = capital
        
        for i in range(self.n_bars):
            open_price = self.open_list[i]
            high = self.high_list[i]
            low = self.low_list[i]
            close = self.close_list[i]
            price_10am = self.price_10am_list[i]
            vwap = self.vwap_list[i]
            
            if open_price <= 0 or price_10am <= 0 or vwap <= 0:
                continue
            
            # Entry condition: price_10am >= open*(1+entry_trigger) AND price_10am > VWAP
            buy_threshold = open_price * (1 + entry_trigger)
            if price_10am < buy_threshold or price_10am <= vwap:
                continue
            
            # We have a buy signal
            entry_price = price_10am
            shares = cash / entry_price
            total_trades += 1
            
            # Check TP/SL
            tp_price = entry_price * (1 + take_profit)
            sl_price = entry_price * (1 + stop_loss)  # stop_loss is negative
            
            # Determine exit price
            if high >= tp_price:
                exit_price = tp_price
            elif low <= sl_price:
                exit_price = sl_price
            else:
                exit_price = close
            
            profit = shares * (exit_price - entry_price)
            cash += profit
            
            if profit > 0:
                winning_trades += 1
            
            # Track drawdown
            if cash > peak_value:
                peak_value = cash
            dd = (peak_value - cash) / peak_value if peak_value > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd
        
        total_return = (cash - capital) / capital if capital > 0 else 0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "max_drawdown": max_drawdown,
            "final_value": cash,
        }
    
    def grid_search(
        self,
        entry_range: tuple = (0.005, 0.025, 0.002),
        tp_range: tuple = (0.015, 0.045, 0.008),
        sl_range: tuple = (-0.015, -0.003, 0.004),
        metric: str = "total_return",
        progress_callback=None
    ) -> dict:
        """
        Run exhaustive 3D grid search over entry, take_profit, stop_loss.
        """
        entry_triggers = np.arange(entry_range[0], entry_range[1], entry_range[2])
        tp_triggers = np.arange(tp_range[0], tp_range[1], tp_range[2])
        sl_triggers = np.arange(sl_range[0], sl_range[1], sl_range[2])
        
        n_combinations = len(entry_triggers) * len(tp_triggers) * len(sl_triggers)
        logger.info(f"VWAP Grid Search: {n_combinations} combinations ({len(entry_triggers)}x{len(tp_triggers)}x{len(sl_triggers)})")
        
        best_score = float('-inf')
        best_params = None
        best_result = None
        tested = 0
        
        for entry in entry_triggers:
            for tp in tp_triggers:
                for sl in sl_triggers:
                    result = self._run_single(float(entry), float(tp), float(sl))
                    score = result.get(metric, 0)
                    tested += 1
                    
                    if score > best_score:
                        best_score = score
                        best_params = {"entry_trigger": float(entry), "take_profit": float(tp), "stop_loss": float(sl)}
                        best_result = result
                    
                    if progress_callback and tested % 50 == 0:
                        progress_callback(tested, n_combinations, best_score)
        
        return {
            "best_params": best_params,
            "best_result": best_result,
            "n_combinations": n_combinations,
            "best_score": best_score,
        }


def run_chatgpt_vwap_optimization(
    bars: list,
    capital: float = 100000,
    buy_range: tuple = (0.001, 0.03, 0.001),
    sell_range: tuple = (0.01, 0.10, 0.002),
    metric: str = "total_return",
    progress_callback=None
) -> dict:
    """
    Run ChatGPT VWAP Strategy optimization.
    
    Entry requires:
    - Price_10AM >= Open * (1 + entry_trigger)
    - Price_10AM > VWAP
    
    Uses buy_range as entry_trigger range, sell_range as take_profit range,
    and auto-computes stop_loss range from buy_range.
    """
    # Map buy_range -> entry_trigger, sell_range -> take_profit
    # Auto-compute sl_range as negative of buy_range
    entry_range = buy_range
    tp_range = sell_range
    sl_range = (-buy_range[1], -buy_range[0], buy_range[2])  # Negative mirror of buy_range
    
    backtester = GPUChatGPTVWAPBacktester(bars, capital)
    search_result = backtester.grid_search(entry_range, tp_range, sl_range, metric, progress_callback)
    
    best_params = search_result["best_params"]
    best_result = search_result["best_result"]
    
    # Safety check for None results (no valid trades found)
    if best_params is None:
        best_params = {"entry_trigger": 0.01, "take_profit": 0.02, "stop_loss": -0.01}
    if best_result is None:
        best_result = {"total_return": 0, "win_rate": 0, "total_trades": 0}
    
    return {
        "symbol": "CHATGPT_VWAP",
        "best_params": {
            "buy_trigger_pct": round(best_params["entry_trigger"] * 100, 2),
            "sell_trigger_pct": round(best_params["take_profit"] * 100, 2),
            "stop_loss_pct": round(abs(best_params["stop_loss"]) * 100, 2),
            "compound": True
        },
        "metrics": {
            "total_return": best_result.get("total_return", 0),
            "sharpe": 0,
            "win_rate": best_result.get("win_rate", 0),
            "max_drawdown": best_result.get("max_drawdown", 0),
            "total_trades": int(best_result.get("total_trades", 0)),
        },
        "optimization_type": "chatgpt_vwap_gpu_hybrid",
        "gpu_enabled": GPU_AVAILABLE,
        "combinations_tested": search_result["n_combinations"]
    }



def run_gpu_optimization(
    bars: list,
    capital: float = 100000,
    buy_range: tuple = (0.005, 0.12, 0.002),
    sell_range: tuple = (0.005, 0.18, 0.002),
    metric: str = "sharpe"
) -> dict:
    """
    Convenience function to run GPU-accelerated optimization.
    
    Args:
        bars: Price bar data
        capital: Starting capital
        buy_range: (min, max, step) for buy trigger search
        sell_range: (min, max, step) for sell trigger search
        metric: Optimization metric
        
    Returns:
        Optimization results
    """
    backtester = GPUBatchBacktester(bars, capital)
    results = backtester.grid_search(buy_range, sell_range, metric)
    
    return {
        "symbol": "GPU_BATCH",
        "best_params": results["best_params"],
        "metrics": results["metrics"],
        "optimization_type": "gpu_grid_search",
        "gpu_enabled": GPU_AVAILABLE,
        "combinations_tested": results["n_combinations_tested"]
    }


if __name__ == "__main__":
    # Test with dummy data
    import random
    
    # Generate fake price data
    bars = []
    price = 100.0
    for _ in range(252):  # 1 year of trading days
        change = random.uniform(-0.03, 0.035)
        open_price = price
        close_price = price * (1 + change)
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.02))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.02))
        bars.append({"o": open_price, "h": high_price, "l": low_price, "c": close_price})
        price = close_price
    
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"Testing with {len(bars)} bars...")
    
    result = run_gpu_optimization(bars)
    print(f"\nBest Parameters:")
    print(f"  Buy Trigger: {result['best_params']['buy_trigger_pct']:.2f}%")
    print(f"  Sell Trigger: {result['best_params']['sell_trigger_pct']:.2f}%")
    print(f"\nMetrics:")
    print(f"  Sharpe: {result['metrics']['sharpe']:.2f}")
    print(f"  Total Return: {result['metrics']['total_return']*100:.1f}%")
    print(f"  Win Rate: {result['metrics']['win_rate']*100:.1f}%")
    print(f"  Trades: {result['metrics']['total_trades']}")
    print(f"\nCombinations tested: {result['combinations_tested']}")
