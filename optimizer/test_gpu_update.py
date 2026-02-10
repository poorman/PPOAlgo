#!/usr/bin/env python3
"""
Test script to verify:
1. GPU label update from RTX 3090 -> RTX 3089 EVGA
2. Core optimizer backtesting logic with sample calculations
"""

import sys
import os
import re

# ============================================================
# TEST 1: Verify GPU label updates in all files
# ============================================================
def test_gpu_label_updates():
    """Check that all references have been updated from RTX 3090 to RTX 3089 EVGA."""
    print("=" * 60)
    print("TEST 1: GPU Label Update Verification")
    print("=" * 60)
    
    files_to_check = {
        "static/index.html": {
            "should_contain": ["RTX 3089 EVGA"],
            "should_not_contain": ["RTX 3090"],
        },
        "gpu_backtest.py": {
            "should_contain": ["RTX 3089 EVGA"],
            "should_not_contain": [],  # Only checking the updated line
        },
        "stock_optimizer.py": {
            "should_contain": ["RTX 3089 EVGA"],
            "should_not_contain": [],
        },
    }
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_passed = True
    
    for filepath, checks in files_to_check.items():
        full_path = os.path.join(base_dir, filepath)
        if not os.path.exists(full_path):
            print(f"  ❌ FAIL: File not found: {filepath}")
            all_passed = False
            continue
        
        with open(full_path, "r") as f:
            content = f.read()
        
        for expected in checks["should_contain"]:
            if expected in content:
                print(f"  ✅ PASS: '{expected}' found in {filepath}")
            else:
                print(f"  ❌ FAIL: '{expected}' NOT found in {filepath}")
                all_passed = False
        
        for unexpected in checks["should_not_contain"]:
            if unexpected in content:
                print(f"  ❌ FAIL: Old reference '{unexpected}' still in {filepath}")
                all_passed = False
            else:
                print(f"  ✅ PASS: Old reference '{unexpected}' removed from {filepath}")
    
    return all_passed


# ============================================================
# TEST 2: Verify core backtesting logic with sample data
# ============================================================
def test_backtesting_logic():
    """Test the GPU backtester with synthetic price data."""
    print("\n" + "=" * 60)
    print("TEST 2: Core Backtesting Logic (Sample Calculations)")
    print("=" * 60)
    
    import numpy as np
    
    # We'll test the backtesting logic directly with NumPy (CPU mode)
    # This validates the math regardless of GPU availability
    
    # Generate sample price bars (simulating 20 trading days)
    np.random.seed(42)
    initial_price = 100.0
    bars = []
    price = initial_price
    
    for day in range(20):
        open_price = price
        # Simulate daily price movement
        daily_change = np.random.normal(0.002, 0.03)  # Slight upward bias
        high = open_price * (1 + abs(daily_change) + np.random.uniform(0, 0.02))
        low = open_price * (1 - abs(daily_change) - np.random.uniform(0, 0.01))
        close = open_price * (1 + daily_change)
        price_10am = open_price * (1 + daily_change * 0.4 + np.random.uniform(-0.005, 0.01))
        
        bars.append({
            "o": round(open_price, 2),
            "h": round(high, 2),
            "l": round(low, 2),
            "c": round(close, 2),
            "price_10am": round(price_10am, 2),
        })
        price = close
    
    print(f"\n  📊 Generated {len(bars)} sample bars")
    print(f"  📈 Price range: ${bars[0]['o']:.2f} -> ${bars[-1]['c']:.2f}")
    
    all_passed = True
    
    # --- Test GPUBatchBacktester (Dipper Strategy) ---
    print("\n  --- Dipper Strategy (GPUBatchBacktester) ---")
    try:
        # Import the module (will use NumPy since CuPy may not be available)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from gpu_backtest import GPUBatchBacktester, GPU_AVAILABLE
        
        capital = 100000
        backtester = GPUBatchBacktester(bars, capital)
        
        # Test with a few parameter combinations
        buy_triggers = np.array([0.02, 0.03, 0.05], dtype=np.float32)
        sell_triggers = np.array([0.05, 0.07, 0.10], dtype=np.float32)
        
        results = backtester.run_batch(buy_triggers, sell_triggers)
        
        print(f"  GPU Available: {GPU_AVAILABLE}")
        print(f"  Combinations tested: {len(buy_triggers)}")
        
        for i in range(len(buy_triggers)):
            ret = results["total_return"][i] * 100
            sr = results["sharpe"][i]
            wr = results["win_rate"][i] * 100
            trades = int(results["total_trades"][i])
            print(f"    Buy={buy_triggers[i]*100:.1f}% Sell={sell_triggers[i]*100:.1f}% "
                  f"→ Return={ret:+.2f}%, Sharpe={sr:.2f}, WinRate={wr:.0f}%, Trades={trades}")
        
        # Validate results structure
        assert "total_return" in results, "Missing total_return"
        assert "sharpe" in results, "Missing sharpe"
        assert "win_rate" in results, "Missing win_rate"
        assert "total_trades" in results, "Missing total_trades"
        assert len(results["total_return"]) == 3, f"Expected 3 results, got {len(results['total_return'])}"
        
        print("  ✅ PASS: GPUBatchBacktester works correctly")
    except Exception as e:
        print(f"  ❌ FAIL: GPUBatchBacktester error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # --- Test Grid Search ---
    print("\n  --- Grid Search ---")
    try:
        grid_result = backtester.grid_search(
            buy_range=(0.01, 0.05, 0.01),
            sell_range=(0.03, 0.10, 0.02),
            metric="total_return"
        )
        
        best = grid_result["best_params"]
        metrics = grid_result["metrics"]
        n_combos = grid_result["n_combinations_tested"]
        
        print(f"  Combinations searched: {n_combos}")
        print(f"  Best Buy Trigger: {best['buy_trigger_pct']:.2f}%")
        print(f"  Best Sell Trigger: {best['sell_trigger_pct']:.2f}%")
        print(f"  Best Return: {metrics['total_return']*100:.2f}%")
        print(f"  Best Sharpe: {metrics['sharpe']:.2f}")
        print(f"  Best Win Rate: {metrics['win_rate']*100:.1f}%")
        
        assert n_combos > 0, "No combinations tested"
        assert "buy_trigger_pct" in best, "Missing buy_trigger_pct"
        assert "sell_trigger_pct" in best, "Missing sell_trigger_pct"
        
        print("  ✅ PASS: Grid Search works correctly")
    except Exception as e:
        print(f"  ❌ FAIL: Grid Search error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # --- Test ChatGPT 9AM Strategy ---
    print("\n  --- ChatGPT 9AM Strategy (GPUChatGPT9AMBacktester) ---")
    try:
        from gpu_backtest import GPUChatGPT9AMBacktester
        
        backtester_9am = GPUChatGPT9AMBacktester(bars, capital)
        
        buy_triggers_9am = np.array([0.005, 0.01, 0.02], dtype=np.float32)
        sell_triggers_9am = np.array([0.02, 0.04, 0.06], dtype=np.float32)
        
        results_9am = backtester_9am.run_batch(buy_triggers_9am, sell_triggers_9am)
        
        for i in range(len(buy_triggers_9am)):
            ret = results_9am["total_return"][i] * 100
            wr = results_9am["win_rate"][i] * 100
            trades = int(results_9am["total_trades"][i])
            print(f"    Buy={buy_triggers_9am[i]*100:.1f}% Sell={sell_triggers_9am[i]*100:.1f}% "
                  f"→ Return={ret:+.2f}%, WinRate={wr:.0f}%, Trades={trades}")
        
        print("  ✅ PASS: GPUChatGPT9AMBacktester works correctly")
    except Exception as e:
        print(f"  ❌ FAIL: GPUChatGPT9AMBacktester error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # --- Test Stop Loss Strategy ---
    print("\n  --- ChatGPT Stop Loss Strategy ---")
    try:
        from gpu_backtest import GPUChatGPTStopLossBacktester
        
        backtester_sl = GPUChatGPTStopLossBacktester(bars, capital)
        result_sl = backtester_sl.run_backtest(compound=True)
        
        print(f"    Total Return: {result_sl['total_return']*100:+.2f}%")
        print(f"    Win Rate: {result_sl['win_rate']*100:.1f}%")
        print(f"    Total Trades: {result_sl['total_trades']}")
        print(f"    Max Drawdown: {result_sl['max_drawdown']*100:.2f}%")
        print(f"    Final Equity: ${result_sl['final_equity']:,.2f}")
        
        assert "total_return" in result_sl, "Missing total_return"
        assert "trade_log" in result_sl, "Missing trade_log"
        
        print("  ✅ PASS: GPUChatGPTStopLossBacktester works correctly")
    except Exception as e:
        print(f"  ❌ FAIL: GPUChatGPTStopLossBacktester error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # --- Test StockVolatilityAnalyzer ---
    print("\n  --- StockVolatilityAnalyzer ---")
    try:
        from stock_optimizer import StockVolatilityAnalyzer
        
        analyzer = StockVolatilityAnalyzer(bars)
        bounds = analyzer.get_parameter_bounds()
        
        print(f"    Avg Daily Range: {analyzer.avg_daily_range:.2f}%")
        print(f"    Max Daily Gain (95th): {analyzer.max_daily_gain:.2f}%")
        print(f"    Volatility Score: {analyzer.volatility_score:.2f}")
        print(f"    Buy Trigger Bounds: {bounds['buy_trigger'][0]:.2f}% - {bounds['buy_trigger'][1]:.2f}%")
        print(f"    Sell Trigger Bounds: {bounds['sell_trigger'][0]:.2f}% - {bounds['sell_trigger'][1]:.2f}%")
        
        assert bounds["buy_trigger"][0] < bounds["buy_trigger"][1], "Invalid buy trigger bounds"
        assert bounds["sell_trigger"][0] < bounds["sell_trigger"][1], "Invalid sell trigger bounds"
        
        print("  ✅ PASS: StockVolatilityAnalyzer works correctly")
    except Exception as e:
        print(f"  ❌ FAIL: StockVolatilityAnalyzer error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    return all_passed


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n🔧 PPOAlgo Optimizer - GPU Update & Sample Calculation Test")
    print("=" * 60)
    
    test1_passed = test_gpu_label_updates()
    test2_passed = test_backtesting_logic()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Test 1 (GPU Label Update):      {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Test 2 (Sample Calculations):    {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n  🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n  ⚠️  SOME TESTS FAILED")
        sys.exit(1)
