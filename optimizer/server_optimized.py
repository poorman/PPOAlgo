#!/usr/bin/env python3
"""
Optimized Stock Optimizer GUI Backend
Enhanced for better CPU (32 cores) and GPU (RTX 3080) utilization.
"""

import os
import json
import config
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import numpy as np
import multiprocessing as mp

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests

# Try importing optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Database core imports
from database import (
    get_db_conn, release_db_conn, ensure_tables, init_db_pool,
    save_job_to_db, save_result_to_db, update_job_status,
    get_results_from_db, ensure_keyword_configs_table,
    get_keyword_configs, save_keyword_config, save_all_keyword_configs,
    ensure_api_keys_table, get_api_keys, save_api_key,
    POSTGRES_AVAILABLE
)

# Try importing GPU backtester
try:
    from gpu_backtest import GPUBatchBacktester, GPU_AVAILABLE
    GPU_BACKTEST_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPU_BACKTEST_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PPOALGO_API = os.getenv("PPOALGO_API", "http://ppoalgo_api_1:8000")
RESULTS_DIR = "/app/results"

# OPTIMIZED CONCURRENCY SETTINGS FOR i9-13900K (24 physical cores, 32 logical threads)
# Use all available cores efficiently
CPU_CORES = min(32, mp.cpu_count())  # i9-13900K has 32 threads
CPU_WORKERS = CPU_CORES - 2  # Reserve 2 cores for system/GPU management

# Increase API limits for better throughput
WIDESURF_SEMAPHORE = asyncio.Semaphore(20)  # Increased from 5
DB_SEMAPHORE = asyncio.Semaphore(50)       # Increased from 20

# GPU batch size optimization for RTX 3080 (10GB VRAM)
GPU_BATCH_SIZE = 5000  # Increased from default, RTX 3080 can handle more
GPU_QUEUE_SIZE = 10    # Pre-queue multiple batches for GPU

# Thread pool for I/O operations
thread_executor = ThreadPoolExecutor(max_workers=CPU_WORKERS)
# Process pool for CPU-bound operations
process_executor = ProcessPoolExecutor(max_workers=CPU_WORKERS//2)

# Database configuration
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "ppoalgo"),
    "user": os.getenv("POSTGRES_USER", "ppoalgo"),
    "password": os.getenv("POSTGRES_PASSWORD", "ppoalgo"),
    "host": os.getenv("POSTGRES_HOST", "ppoalgo_db_1"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}

# Timezone constants
ET_ZONE = ZoneInfo("America/New_York")
UTC_ZONE = ZoneInfo("UTC")

# Import original functions we're not changing
from server import (
    get_et_minutes_from_timestamp,
    OptimizationRequest,
    active_connections,
    broadcast_message,
    optimization_results,
    cancelled_jobs,
    CancelRequest,
    is_job_cancelled,
    app
)

# Override the optimization endpoint with optimized version
@app.post("/api/optimize")
async def start_optimization_optimized(request: OptimizationRequest):
    """Optimized version using all CPU cores and better GPU utilization."""
    if not OPTUNA_AVAILABLE:
        raise HTTPException(status_code=500, detail="Optuna not installed")

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save job to database
    save_job_to_db(job_id, request)

    # Run optimization with enhanced parallelism
    async def run_all_optimized():
        results = {}
        request.symbols = list(dict.fromkeys(request.symbols))
        total_count = len(request.symbols)
        completed_count = 0

        # Enhanced worker with batch processing
        async def enhanced_worker(worker_id, work_queue, engine_type='cpu'):
            nonlocal completed_count
            logger.info(f"Enhanced Worker {worker_id} ({engine_type}) started with batch processing")

            while not work_queue.empty():
                if is_job_cancelled(job_id):
                    break

                try:
                    # Process multiple symbols in batch for GPU
                    if engine_type == 'gpu' and GPU_BACKTEST_AVAILABLE:
                        batch = []
                        batch_size = min(5, work_queue.qsize())  # Process up to 5 symbols at once on GPU

                        for _ in range(batch_size):
                            if not work_queue.empty():
                                symbol = await work_queue.get()
                                batch.append(symbol)

                        if batch:
                            logger.info(f"GPU Worker {worker_id} processing batch: {batch}")
                            # Process batch in parallel on GPU
                            batch_tasks = [optimize_stock_gpu_enhanced(s, request, job_id) for s in batch]
                            batch_results = await asyncio.gather(*batch_tasks)

                            for symbol, res in zip(batch, batch_results):
                                if res:
                                    results[symbol] = res
                                    optimization_results[symbol] = res
                                completed_count += len(batch)
                                for _ in batch:
                                    work_queue.task_done()
                    else:
                        # CPU processing - one at a time but with better parallelism
                        symbol = await work_queue.get()
                        start_time = time.time()

                        res = await optimize_stock_cpu_enhanced(symbol, request, job_id)
                        duration = time.time() - start_time

                        if res:
                            results[symbol] = res
                            optimization_results[symbol] = res
                            logger.info(f"CPU Worker {worker_id} finished {symbol} in {duration:.2f}s")

                        completed_count += 1
                        work_queue.task_done()

                    # Progress updates
                    if completed_count % 5 == 0 or completed_count == total_count:
                        progress_pct = (completed_count / total_count) * 100
                        await broadcast_message({
                            "type": "job_progress",
                            "job_id": job_id,
                            "completed": completed_count,
                            "total": total_count,
                            "progress_percentage": progress_pct,
                            "message": f"Progress: {completed_count}/{total_count} stocks ({progress_pct:.1f}%)"
                        })

                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    if not work_queue.empty():
                        work_queue.task_done()

        # Determine optimal work distribution
        workers = []

        if request.use_gpu and GPU_BACKTEST_AVAILABLE:
            # Optimized dual-queue with better GPU utilization
            gpu_queue = asyncio.Queue()
            cpu_queue = asyncio.Queue()

            # GPU should handle more since it's underutilized (was 40%, now 60%)
            gpu_share = max(1, int(total_count * 0.6))
            symbols_list = list(request.symbols)

            # Prioritize high-volume stocks for GPU
            for i, s in enumerate(symbols_list):
                if i < gpu_share:
                    gpu_queue.put_nowait(s)
                else:
                    cpu_queue.put_nowait(s)

            logger.info(f"Optimized dispatch: GPU={gpu_share} stocks, CPU={total_count - gpu_share} stocks")

            # Multiple GPU workers to keep GPU busy
            gpu_workers = min(3, gpu_share // 5 + 1)  # Up to 3 GPU workers
            for i in range(gpu_workers):
                workers.append(asyncio.create_task(enhanced_worker(f"GPU-{i+1}", gpu_queue, engine_type='gpu')))

            # Use all available CPU cores
            cpu_worker_count = CPU_WORKERS  # Use all 30 available workers
            for i in range(cpu_worker_count):
                workers.append(asyncio.create_task(enhanced_worker(f"CPU-{i+1}", cpu_queue, engine_type='cpu')))
        else:
            # CPU-only mode with maximum parallelism
            cpu_queue = asyncio.Queue()
            for s in request.symbols:
                cpu_queue.put_nowait(s)

            # Use all CPU workers
            for i in range(CPU_WORKERS):
                workers.append(asyncio.create_task(enhanced_worker(f"CPU-{i+1}", cpu_queue, engine_type='cpu')))

        # Wait for completion
        if request.use_gpu and GPU_BACKTEST_AVAILABLE:
            await asyncio.gather(gpu_queue.join(), cpu_queue.join())
        else:
            await cpu_queue.join()

        # Cancel workers
        for w in workers:
            w.cancel()

        # Save results
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        results_file = os.path.join(RESULTS_DIR, f"{job_id}_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Update job status
        update_job_status(job_id, 'complete')

        await broadcast_message({
            "type": "optimization_complete",
            "job_id": job_id,
            "message": f"✅ Optimization complete! Processed {len(results)}/{total_count} stocks successfully.",
            "results": results
        })

    # Start optimization in background
    asyncio.create_task(run_all_optimized())

    return {
        "status": "started",
        "job_id": job_id,
        "message": f"🚀 Starting optimized processing for {len(request.symbols)} stocks using {CPU_WORKERS} CPU cores + GPU"
    }

async def optimize_stock_gpu_enhanced(symbol: str, config: OptimizationRequest, job_id: str):
    """Enhanced GPU optimization with better batching and memory management."""
    try:
        # Fetch price data
        response = requests.post(
            f"{PPOALGO_API}/api/get_prices",
            json={"symbol": symbol},
            timeout=30
        )

        if response.status_code != 200:
            logger.error(f"Failed to get prices for {symbol}")
            return None

        price_data = response.json()
        bars = price_data.get("bars", [])

        if len(bars) < 10:
            logger.warning(f"Insufficient data for {symbol}")
            return None

        # Initialize GPU backtester with larger batch size
        gpu_tester = GPUBatchBacktester(bars, capital=config.capital)

        # Generate larger parameter grid for better GPU utilization
        buy_range = np.linspace(0.001, 0.05, 50)  # Increased from default
        sell_range = np.linspace(0.005, 0.10, 50)

        # Create all combinations
        buy_triggers = []
        sell_triggers = []
        for b in buy_range:
            for s in sell_range:
                if s > b:  # Sell must be higher than buy
                    buy_triggers.append(b)
                    sell_triggers.append(s)

        # Process in larger batches for better GPU utilization
        batch_size = min(GPU_BATCH_SIZE, len(buy_triggers))
        best_result = None
        best_sharpe = -999999

        for i in range(0, len(buy_triggers), batch_size):
            batch_buy = buy_triggers[i:i+batch_size]
            batch_sell = sell_triggers[i:i+batch_size]

            results = gpu_tester.run_batch(
                np.array(batch_buy),
                np.array(batch_sell),
                np.ones(len(batch_buy), dtype=bool) if config.compound else np.zeros(len(batch_buy), dtype=bool)
            )

            # Find best in batch
            for idx, sharpe in enumerate(results.get('sharpe_ratios', [])):
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_result = {
                        'symbol': symbol,
                        'buy_trigger': batch_buy[idx],
                        'sell_trigger': batch_sell[idx],
                        'sharpe_ratio': sharpe,
                        'total_return': results['total_returns'][idx],
                        'win_rate': results['win_rates'][idx],
                        'method': 'gpu_grid_enhanced'
                    }

        if best_result:
            # Save to database
            save_result_to_db(job_id, best_result)

            await broadcast_message({
                "type": "stock_complete",
                "symbol": symbol,
                "job_id": job_id,
                "result": best_result,
                "message": f"✅ {symbol} optimization complete (GPU Enhanced)"
            })

        return best_result

    except Exception as e:
        logger.error(f"GPU optimization error for {symbol}: {e}")
        return None

async def optimize_stock_cpu_enhanced(symbol: str, config: OptimizationRequest, job_id: str):
    """Enhanced CPU optimization with better parallelism."""
    try:
        # Use thread pool for I/O operations
        loop = asyncio.get_event_loop()

        # Fetch price data asynchronously
        response = await loop.run_in_executor(
            thread_executor,
            lambda: requests.post(
                f"{PPOALGO_API}/api/get_prices",
                json={"symbol": symbol},
                timeout=30
            )
        )

        if response.status_code != 200:
            logger.error(f"Failed to get prices for {symbol}")
            return None

        price_data = response.json()
        bars = price_data.get("bars", [])

        if len(bars) < 10:
            logger.warning(f"Insufficient data for {symbol}")
            return None

        # Use Optuna with more trials for CPU
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(n_startup_trials=50)  # More startup trials
        )

        def objective(trial):
            buy_trigger = trial.suggest_float("buy_trigger", 0.001, 0.05)
            sell_trigger = trial.suggest_float("sell_trigger", buy_trigger + 0.001, 0.10)

            # Run backtest
            result = run_simple_backtest(
                bars,
                buy_trigger,
                sell_trigger,
                config.capital,
                config.compound
            )

            return result.get('sharpe_ratio', -999999)

        # Run optimization with more trials
        await loop.run_in_executor(
            thread_executor,
            lambda: study.optimize(objective, n_trials=500)  # Increased trials
        )

        if study.best_trial:
            best_params = study.best_params
            best_result = {
                'symbol': symbol,
                'buy_trigger': best_params['buy_trigger'],
                'sell_trigger': best_params['sell_trigger'],
                'sharpe_ratio': study.best_value,
                'method': 'cpu_optuna_enhanced'
            }

            # Save to database
            save_result_to_db(job_id, best_result)

            await broadcast_message({
                "type": "stock_complete",
                "symbol": symbol,
                "job_id": job_id,
                "result": best_result,
                "message": f"✅ {symbol} optimization complete (CPU Enhanced)"
            })

            return best_result

        return None

    except Exception as e:
        logger.error(f"CPU optimization error for {symbol}: {e}")
        return None

def run_simple_backtest(bars, buy_trigger, sell_trigger, capital, compound):
    """Simple backtest implementation for CPU optimization."""
    cash = capital
    position = 0
    trades = 0
    wins = 0
    returns = []

    for i in range(1, len(bars)):
        prev_close = bars[i-1].get('c', 0)
        high = bars[i].get('h', 0)
        close = bars[i].get('c', 0)

        if prev_close > 0:
            pct_change = (high - prev_close) / prev_close

            # Buy signal
            if position == 0 and pct_change >= buy_trigger:
                position = cash / high if compound else capital / high
                entry_price = high
                cash = 0 if compound else cash - capital

            # Sell signal
            elif position > 0:
                pct_gain = (close - entry_price) / entry_price
                if pct_gain >= sell_trigger:
                    cash = position * close if compound else cash + position * close
                    if close > entry_price:
                        wins += 1
                    trades += 1
                    returns.append(pct_gain)
                    position = 0

    # Calculate metrics
    if len(returns) > 0:
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        win_rate = wins / trades if trades > 0 else 0
        total_return = (cash - capital) / capital if capital > 0 else 0
    else:
        sharpe_ratio = -999999
        win_rate = 0
        total_return = 0

    return {
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'total_return': total_return,
        'trades': trades
    }

# Initialize optimized settings on startup
@app.on_event("startup")
async def startup_event_optimized():
    """Initialize optimized settings on startup."""
    logger.info(f"🚀 Starting optimized server with {CPU_WORKERS} CPU workers")
    logger.info(f"📊 GPU Batch Size: {GPU_BATCH_SIZE}, Queue Size: {GPU_QUEUE_SIZE}")
    logger.info(f"🔧 API Limits: Widesurf={20}, Database={50}")

    # Initialize database pool with more connections
    if POSTGRES_AVAILABLE:
        init_db_pool()
        ensure_tables()

    # Warm up GPU if available
    if GPU_AVAILABLE:
        try:
            import cupy as cp
            # Allocate some memory to warm up GPU
            test_array = cp.zeros((1000, 1000), dtype=cp.float32)
            del test_array
            logger.info("✅ GPU warmed up successfully")
        except Exception as e:
            logger.warning(f"GPU warmup failed: {e}")

if __name__ == "__main__":
    import uvicorn
    # Use more workers for better concurrency
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)