"""
Rust VWAP Optimizer Bridge
==========================
Calls the compiled Rust binary (rust_vwap) via subprocess,
passing bar data as JSON on stdin and reading results from stdout.

The Rust binary uses Rayon to parallelise the grid search across
all 32 logical processors on the i9-13900K.
"""

import json
import logging
import os
import subprocess
import time

logger = logging.getLogger(__name__)

# Path to the compiled Rust binary
RUST_BINARY = os.path.join(os.path.dirname(__file__), "rust_vwap", "target", "release", "rust_vwap")

# Fallback: check if binary is at /app level (inside Docker)
if not os.path.exists(RUST_BINARY):
    RUST_BINARY = os.path.join(os.path.dirname(__file__), "rust_vwap_bin")
    
RUST_AVAILABLE = os.path.exists(RUST_BINARY)


def run_rust_vwap_optimization(
    bars: list,
    capital: float = 100000,
    buy_range: tuple = (0.001, 0.03, 0.001),
    sell_range: tuple = (0.01, 0.10, 0.002),
    metric: str = "total_return",
    threads: int = None
) -> dict:
    """
    Run the VWAP Momentum 10am optimization using the Rust binary.
    
    The Rust binary uses Rayon to parallelise across all CPU cores.
    
    Args:
        bars: Daily bars with price_10am, vwap, vwap_940, or_high, or_low
        capital: Starting capital
        buy_range: (min, max, step) as decimals
        sell_range: (min, max, step) as decimals
        metric: "total_return", "sharpe", or "win_rate"
        threads: Number of threads for Rayon (defaults to all cores)
        
    Returns:
        Dict with best_params, metrics, combinations_tested, threads_used
    """
    if not RUST_AVAILABLE:
        raise RuntimeError(f"Rust binary not found at {RUST_BINARY}")
    
    # Prepare input JSON for the Rust binary
    input_data = {
        "bars": [
            {
                "o": float(b.get("o", b.get("open", 0))),
                "h": float(b.get("h", b.get("high", 0))),
                "l": float(b.get("l", b.get("low", 0))),
                "c": float(b.get("c", b.get("close", 0))),
                "price_10am": float(b.get("price_10am", 0)),
                "vwap": float(b.get("vwap", 0)),
                "vwap_940": float(b.get("vwap_940", 0)),
                "or_high": float(b.get("or_high", 0)),
                "or_low": float(b.get("or_low", 0)),
            }
            for b in bars
        ],
        "capital": capital,
        "buy_range": list(buy_range),
        "sell_range": list(sell_range),
        "metric": metric,
    }
    
    input_json = json.dumps(input_data)
    
    logger.info(f"[RUST] Starting Rayon-parallel VWAP optimization "
                f"({len(bars)} bars, metric={metric}, threads={threads or 'all'})")
    
    start = time.time()
    
    env = os.environ.copy()
    if threads:
        env["RAYON_NUM_THREADS"] = str(threads)
    
    try:
        result = subprocess.run(
            [RUST_BINARY],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            env=env
        )
        
        if result.returncode != 0:
            logger.error(f"[RUST] Binary exited with code {result.returncode}: {result.stderr}")
            raise RuntimeError(f"Rust optimizer failed: {result.stderr}")
        
        output = json.loads(result.stdout)
        elapsed = time.time() - start
        
        logger.info(f"[RUST] Optimization complete in {elapsed:.2f}s "
                     f"({output['combinations_tested']} combos on {output['threads_used']} threads)")
        
        return {
            "symbol": "RUST_VWAP",
            "best_params": output["best_params"],
            "metrics": output["metrics"],
            "optimization_type": "rust_vwap_rayon",
            "gpu_enabled": False,
            "combinations_tested": output["combinations_tested"],
            "threads_used": output["threads_used"],
            "duration_seconds": round(elapsed, 2),
        }
        
    except subprocess.TimeoutExpired:
        logger.error("[RUST] Binary timed out after 120s")
        raise RuntimeError("Rust optimizer timed out")
    except json.JSONDecodeError as e:
        logger.error(f"[RUST] Failed to parse output: {e}")
        raise RuntimeError(f"Failed to parse Rust output: {e}")
