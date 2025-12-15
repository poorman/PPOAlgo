#!/usr/bin/env python3
"""
GPU Warmup Script - Pre-compiles CuPy kernels on container startup
This avoids the 15-20s delay on the first optimization run.
"""

import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def warmup_gpu():
    """Run a minimal GPU computation to trigger kernel compilation."""
    logger.info("Starting GPU warmup...")
    start = time.time()
    
    try:
        from gpu_backtest import GPUBatchBacktester, GPU_AVAILABLE
        
        if not GPU_AVAILABLE:
            logger.warning("GPU not available, skipping warmup")
            return
        
        # Create minimal test data (20 bars is enough to trigger all kernel compilations)
        test_bars = []
        price = 100.0
        for i in range(50):
            change = 0.01 if i % 2 == 0 else -0.01
            open_price = price
            close_price = price * (1 + change)
            high_price = max(open_price, close_price) * 1.01
            low_price = min(open_price, close_price) * 0.99
            test_bars.append({"o": open_price, "h": high_price, "l": low_price, "c": close_price})
            price = close_price
        
        # Run a small grid search to compile all kernels
        logger.info("Compiling GPU kernels with test data...")
        backtester = GPUBatchBacktester(test_bars, capital=10000)
        
        # Run with small parameter space
        results = backtester.grid_search(
            buy_range=(0.01, 0.05, 0.01),
            sell_range=(0.01, 0.05, 0.01),
            metric="sharpe"
        )
        
        elapsed = time.time() - start
        logger.info(f"GPU warmup complete in {elapsed:.1f}s - Kernels compiled and cached!")
        logger.info(f"Test result: sharpe={results['metrics']['sharpe']:.2f}, combinations={results['n_combinations_tested']}")
        
    except Exception as e:
        logger.error(f"GPU warmup failed: {e}")
        logger.info("Warmup failed but server will continue - first optimization may be slow")

if __name__ == "__main__":
    warmup_gpu()
