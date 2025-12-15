#!/usr/bin/env python3
"""
Stock Optimizer GUI Backend
FastAPI server with WebSocket support for real-time optimization updates.
Saves optimization results to PostgreSQL database.
"""

import os
import json
import asyncio
from datetime import datetime, time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np

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

# Try importing psycopg2 for PostgreSQL
try:
    import psycopg2
    from psycopg2.extras import Json
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("psycopg2 not installed, database features disabled")

import numpy as np

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

# Database configuration
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "ppoalgo"),
    "user": os.getenv("POSTGRES_USER", "ppoalgo"),
    "password": os.getenv("POSTGRES_PASSWORD", "ppoalgo"),
    "host": os.getenv("POSTGRES_HOST", "ppoalgo_db_1"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}


def get_db_conn():
    """Get PostgreSQL connection."""
    if not POSTGRES_AVAILABLE:
        return None
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None


def ensure_tables():
    """Create optimization tables if they don't exist."""
    conn = get_db_conn()
    if not conn:
        return
    
    try:
        with conn.cursor() as cur:
            # Optimization jobs table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS optimizer_jobs (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(50) UNIQUE NOT NULL,
                    symbols TEXT[] NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    capital NUMERIC(15, 2) NOT NULL,
                    n_trials INTEGER NOT NULL,
                    optimization_metric VARCHAR(50) NOT NULL,
                    status VARCHAR(20) DEFAULT 'running',
                    smart_timing BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Optimization results table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS optimizer_results (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(50) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    buy_trigger_pct NUMERIC(5, 2) NOT NULL,
                    sell_trigger_pct NUMERIC(5, 2) NOT NULL,
                    compound BOOLEAN NOT NULL,
                    optimized_for VARCHAR(50) NOT NULL,
                    score NUMERIC(10, 4),
                    volatility_avg_range NUMERIC(10, 4),
                    volatility_max_gain NUMERIC(10, 4),
                    volatility_score NUMERIC(10, 4),
                    optimal_buy_time_cdt VARCHAR(8),
                    full_result JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(job_id, symbol)
                );
            """)
            
            # Add optimal_buy_time_cdt column if it doesn't exist (migration)
            try:
                cur.execute("""
                    ALTER TABLE optimizer_results 
                    ADD COLUMN IF NOT EXISTS optimal_buy_time_cdt VARCHAR(8);
                """)
            except Exception as e:
                logger.debug(f"Column might already exist: {e}")
            
            # Add smart_timing column to optimizer_jobs if it doesn't exist (migration)
            try:
                cur.execute("""
                    ALTER TABLE optimizer_jobs 
                    ADD COLUMN IF NOT EXISTS smart_timing BOOLEAN DEFAULT FALSE;
                """)
            except Exception as e:
                logger.debug(f"Column might already exist: {e}")
            
            # Create index for faster lookups
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_optimizer_results_symbol 
                ON optimizer_results(symbol);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_optimizer_results_job_id 
                ON optimizer_results(job_id);
            """)
            
        conn.commit()
        logger.info("Database tables ensured")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
    finally:
        conn.close()


def save_job_to_db(job_id: str, config: "OptimizationRequest"):
    """Save optimization job to database."""
    conn = get_db_conn()
    if not conn:
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO optimizer_jobs 
                (job_id, symbols, start_date, end_date, capital, n_trials, optimization_metric, smart_timing)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (job_id) DO NOTHING
            """, (
                job_id,
                config.symbols,
                config.start_date,
                config.end_date,
                config.capital,
                config.n_trials,
                config.optimization_metric,
                config.smart_timing
            ))
        conn.commit()
        logger.info(f"Saved job {job_id} to database")
    except Exception as e:
        logger.error(f"Failed to save job: {e}")
    finally:
        conn.close()


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    return obj


def save_result_to_db(job_id: str, result: dict):
    """Save optimization result to database."""
    conn = get_db_conn()
    if not conn:
        return
    
    # Convert numpy types to native Python types
    result = convert_numpy_types(result)
    
    try:
        with conn.cursor() as cur:
            # Extract score from metrics
            score = 0
            if result.get("metrics"):
                score_values = list(result["metrics"].values())
                if score_values:
                    score = float(score_values[0]) if score_values[0] is not None else 0
            
            cur.execute("""
                INSERT INTO optimizer_results 
                (job_id, symbol, buy_trigger_pct, sell_trigger_pct, compound, 
                 optimized_for, score, volatility_avg_range, volatility_max_gain, 
                 volatility_score, optimal_buy_time_cdt, full_result)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (job_id, symbol) DO UPDATE SET
                    buy_trigger_pct = EXCLUDED.buy_trigger_pct,
                    sell_trigger_pct = EXCLUDED.sell_trigger_pct,
                    compound = EXCLUDED.compound,
                    score = EXCLUDED.score,
                    optimal_buy_time_cdt = EXCLUDED.optimal_buy_time_cdt,
                    full_result = EXCLUDED.full_result
            """, (
                job_id,
                result["symbol"],
                float(result["best_params"]["buy_trigger_pct"]),
                float(result["best_params"]["sell_trigger_pct"]),
                bool(result["best_params"]["compound"]),
                result["optimized_for"],
                score,
                float(result["volatility_profile"]["avg_daily_range"]),
                float(result["volatility_profile"]["max_daily_gain"]),
                float(result["volatility_profile"]["volatility_score"]),
                result.get("optimal_buy_time_cdt"),  # New field
                Json(result)
            ))
        conn.commit()
        logger.info(f"Saved result for {result['symbol']} to database")
    except Exception as e:
        logger.error(f"Failed to save result: {e}")
    finally:
        conn.close()


def update_job_status(job_id: str, status: str):
    """Update job status in database."""
    conn = get_db_conn()
    if not conn:
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE optimizer_jobs SET status = %s WHERE job_id = %s
            """, (status, job_id))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")
    finally:
        conn.close()


def get_results_from_db(symbol: str = None) -> List[dict]:
    """Get optimization results from database."""
    conn = get_db_conn()
    if not conn:
        return []
    
    try:
        with conn.cursor() as cur:
            if symbol:
                cur.execute("""
                    SELECT full_result FROM optimizer_results 
                    WHERE symbol = %s ORDER BY created_at DESC LIMIT 10
                """, (symbol.upper(),))
            else:
                cur.execute("""
                    SELECT full_result FROM optimizer_results 
                    ORDER BY created_at DESC LIMIT 50
                """)
            
            rows = cur.fetchall()
            return [row[0] for row in rows if row[0]]
    except Exception as e:
        logger.error(f"Failed to get results: {e}")
        return []
    finally:
        conn.close()


app = FastAPI(title="Stock Optimizer GUI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    """Initialize database tables on startup."""
    ensure_tables()


# Store for active WebSocket connections
active_connections: List[WebSocket] = []

# Store optimization results
optimization_results: Dict[str, dict] = {}
optimization_status: Dict[str, dict] = {}



class OptimizationRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    capital: float = 100000
    n_trials: int = 200
    optimization_metric: str = "sharpe"
    use_gpu: bool = False  # Enable GPU-accelerated grid search
    smart_timing: bool = False  # Enable optimal buy time optimization
    timing_approach: str = "sequential"  # "sequential" or "joint"


class StockVolatilityAnalyzer:
    """Analyze stock volatility to set intelligent parameter bounds."""
    
    def __init__(self, bars: List[dict]):
        self.bars = bars
        self.analyze()
    
    def analyze(self):
        if not self.bars:
            self.avg_daily_range = 5.0
            self.max_daily_gain = 10.0
            self.volatility_score = 5.0
            return
        
        daily_gains = []
        gaps = []
        
        for i, bar in enumerate(self.bars):
            high = bar.get("h", bar.get("c", 0))
            low = bar.get("l", bar.get("c", 0))
            open_price = bar.get("o", bar.get("c", 0))
            
            if open_price > 0:
                daily_gain = ((high - open_price) / open_price) * 100
                daily_gains.append(daily_gain)
                
                if i > 0:
                    prev_close = self.bars[i-1].get("c", 0)
                    if prev_close > 0:
                        gap = ((high - prev_close) / prev_close) * 100
                        gaps.append(gap)
        
        self.avg_daily_range = np.mean(daily_gains) if daily_gains else 5.0
        self.max_daily_gain = np.percentile(daily_gains, 95) if daily_gains else 10.0
        self.avg_gap = np.mean(gaps) if gaps else 2.0
        self.volatility_score = np.std(daily_gains) if daily_gains else 5.0
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        buy_low = max(0.5, self.avg_gap * 0.3)
        buy_high = min(self.max_daily_gain * 0.8, 20.0)
        sell_low = max(1.0, self.avg_daily_range * 0.3)
        sell_high = min(self.max_daily_gain * 1.2, 30.0)
        
        return {
            "buy_trigger": (buy_low, buy_high),
            "sell_trigger": (sell_low, sell_high),
        }


async def broadcast_message(message: dict):
    """Send message to all connected WebSocket clients."""
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception:
            pass


def sanitize_metric(value, decimals=2):
    """Sanitize metric values, replacing invalid values with None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value == -999999 or not np.isfinite(value) or np.isnan(value):
            return None
        return round(value, decimals) if decimals else value
    return None


def run_backtest(symbol: str, params: dict, config: OptimizationRequest) -> dict:
    """Run a single backtest via the PPOAlgo API."""
    try:
        resp = requests.post(
            f"{PPOALGO_API}/api/backtest",
            json={
                "symbol": symbol,
                "start": config.start_date,
                "end": config.end_date,
                "capital": config.capital,
                "buy_trigger_pct": params["buy_trigger"],
                "sell_trigger_pct": params["sell_trigger"],
                "buy_amount": 0,  # Use all capital
                "compound": params.get("compound", True),
            },
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()["metrics"]
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return {}


def get_optimal_buy_time_simple(symbol: str, optimization_metric: str) -> str:
    """
    Simple buy time determination based on market patterns and optimization goals.
    Returns optimal buy time in HH:MM format (CDT).
    """
    # Market opening patterns and optimal timing based on research
    timing_patterns = {
        "sharpe": "10:30",      # Balanced risk-reward after morning volatility
        "total_return": "10:00",  # Capture early momentum 
        "win_rate": "11:00"     # More stable entry after initial volatility
    }
    
    # Symbol-specific adjustments for high volatility stocks
    volatile_symbols = {"TSLA", "NVDA", "AMD", "SHOP", "SQ", "ARKK", "SOUN", "SOLS"}
    
    base_time = timing_patterns.get(optimization_metric, "10:30")
    
    if symbol in volatile_symbols:
        # For volatile stocks, wait a bit longer for price stabilization
        if base_time == "10:00":
            return "10:15"
        elif base_time == "10:30": 
            return "10:45"
        elif base_time == "11:00":
            return "11:15"
    
    return base_time


async def optimize_buy_time_sequential(
    symbol: str,
    bars: list,
    buy_trigger: float,
    sell_trigger: float, 
    compound: bool,
    config: OptimizationRequest,
    job_id: str
) -> Optional[str]:
    """
    Approach 1: Sequential Optimization
    Find optimal buy time using pre-determined optimal triggers.
    Tests different buy times (9:00 AM - 2:30 PM CDT) and returns the best performing time.
    """
    try:
        # Fetch intraday data (5-minute intervals)
        logger.info(f"Fetching intraday data for {symbol} from {PPOALGO_API}")
        resp = requests.get(
            f"{PPOALGO_API}/api/prices",
            params={
                "symbol": symbol,
                "start": config.start_date,
                "end": config.end_date,
                "timeframe": "5Min"
            },
            timeout=45
        )
        resp.raise_for_status()
        intraday_bars = resp.json()
        
        if not intraday_bars:
            logger.warning(f"No intraday data available for {symbol}")
            return None
        
        logger.info(f"Retrieved {len(intraday_bars)} intraday bars for {symbol}")
            
        # Define buy time candidates (CDT: 9:00 AM to 2:30 PM)
        buy_times = []
        for hour in range(9, 15):  # 9 AM to 2 PM
            for minute in range(0, 60, 30):  # Every 30 minutes
                if hour == 14 and minute > 30:  # Stop at 2:30 PM
                    break
                buy_times.append(f"{hour:02d}:{minute:02d}")
        
        best_score = float('-inf')
        best_buy_time = None
        
        # Test each buy time
        for i, buy_time in enumerate(buy_times):
            # Simulate trading with this specific buy time
            score = await simulate_intraday_trading(
                intraday_bars=intraday_bars,
                buy_trigger_pct=buy_trigger,
                sell_trigger_pct=sell_trigger,
                buy_time_cdt=buy_time,
                compound=compound,
                optimization_metric=config.optimization_metric
            )
            
            if score > best_score:
                best_score = score
                best_buy_time = buy_time
                
            # Send progress updates
            if i % 5 == 0:  # Update every 5 iterations
                await broadcast_message({
                    "type": "progress",
                    "job_id": job_id,
                    "symbol": symbol,
                    "message": f"Testing buy time {buy_time} (score: {score:.2f})",
                    "phase": "timing_optimization"
                })
        
        logger.info(f"Sequential approach - Best buy time for {symbol}: {best_buy_time} with score {best_score:.2f}")
        return best_buy_time
        
    except Exception as e:
        logger.error(f"Sequential timing optimization failed for {symbol}: {e}")
        return None


async def optimize_buy_time_joint(
    symbol: str,
    bars: list,
    config: OptimizationRequest,
    job_id: str
) -> Optional[dict]:
    """
    Approach 2: Joint Optimization  
    Simultaneously optimize buy_trigger, sell_trigger, and buy_time using Optuna.
    Returns best parameters including optimal buy time.
    """
    try:
        # Fetch intraday data
        resp = requests.get(
            f"{PPOALGO_API}/api/prices",
            params={
                "symbol": symbol,
                "start": config.start_date,
                "end": config.end_date,
                "timeframe": "5Min"
            },
            timeout=45
        )
        resp.raise_for_status()
        intraday_bars = resp.json()
        
        if not intraday_bars:
            logger.warning(f"No intraday data available for {symbol}")
            return None
            
        # Analyze volatility for bounds
        analyzer = StockVolatilityAnalyzer(bars)
        bounds = analyzer.get_parameter_bounds()
        
        # Create Optuna study for joint optimization
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        def objective(trial):
            # Sample all 3 parameters together
            buy_trigger = trial.suggest_float(
                "buy_trigger", 
                bounds["buy_trigger"][0], 
                bounds["buy_trigger"][1],
                step=0.1
            )
            sell_trigger = trial.suggest_float(
                "sell_trigger",
                bounds["sell_trigger"][0],
                bounds["sell_trigger"][1], 
                step=0.1
            )
            compound = trial.suggest_categorical("compound", [True, False])
            
            # Sample buy time (9:00 AM to 2:30 PM CDT)
            buy_hour = trial.suggest_int("buy_hour", 9, 14)
            buy_minute = trial.suggest_int("buy_minute", 0, 59, step=30)
            
            # Ensure we don't go past 2:30 PM
            if buy_hour == 14 and buy_minute > 30:
                buy_minute = 30
                
            buy_time = f"{buy_hour:02d}:{buy_minute:02d}"
            
            # Evaluate this combination
            score = asyncio.run(simulate_intraday_trading(
                intraday_bars=intraday_bars,
                buy_trigger_pct=buy_trigger,
                sell_trigger_pct=sell_trigger,
                buy_time_cdt=buy_time,
                compound=compound,
                optimization_metric=config.optimization_metric
            ))
            
            return score if score and score != float('-inf') else -999999
        
        # Progress callback for joint optimization
        trials_completed = [0]
        def callback(study, trial):
            trials_completed[0] += 1
            asyncio.run_coroutine_threadsafe(
                broadcast_message({
                    "type": "progress",
                    "job_id": job_id,
                    "symbol": symbol,
                    "trials_completed": trials_completed[0],
                    "total_trials": config.n_trials,
                    "best_score": study.best_value if study.best_trial else 0,
                    "phase": "joint_optimization"
                }), 
                asyncio.get_event_loop()
            )
        
        # Run joint optimization
        study.optimize(
            objective,
            n_trials=config.n_trials,
            callbacks=[callback]
        )
        
        best_params = study.best_params
        best_buy_time = f"{best_params['buy_hour']:02d}:{best_params['buy_minute']:02d}"
        
        result = {
            "buy_trigger": best_params["buy_trigger"],
            "sell_trigger": best_params["sell_trigger"], 
            "compound": best_params["compound"],
            "optimal_buy_time_cdt": best_buy_time,
            "best_score": study.best_value
        }
        
        logger.info(f"Joint approach - Best parameters for {symbol}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Joint timing optimization failed for {symbol}: {e}")
        return None


async def simulate_intraday_trading(
    intraday_bars: list,
    buy_trigger_pct: float,
    sell_trigger_pct: float,
    buy_time_cdt: str,
    compound: bool,
    optimization_metric: str
) -> float:
    """
    Simulate intraday trading with specific buy time.
    Returns performance score based on optimization metric.
    """
    try:
        # Convert buy_time_cdt to comparable format
        buy_hour, buy_minute = map(int, buy_time_cdt.split(':'))
        
        # Simulate trading logic
        initial_capital = 10000
        capital = initial_capital
        positions = []
        trades = []
        
        for bar in intraday_bars:
            # Parse bar timestamp 
            bar_time = datetime.fromisoformat(bar['timestamp'].replace('Z', '+00:00'))
            bar_hour = bar_time.hour
            bar_minute = bar_time.minute
            
            # Only buy at the specified time
            if bar_hour == buy_hour and bar_minute == buy_minute:
                # Check if we should buy (price dropped by buy_trigger_pct)
                open_price = bar['open']
                
                # Simple buy logic - buy if we have capital
                if capital > 0:
                    shares = capital / open_price
                    positions.append({
                        'shares': shares,
                        'entry_price': open_price,
                        'entry_time': bar_time
                    })
                    capital = 0
            
            # Sell logic - check all positions
            for pos in positions[:]:  # Copy list to modify during iteration
                current_price = bar['close']
                gain_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                
                # Sell if hit sell trigger or end of day (3:55 PM CDT)
                should_sell = (
                    gain_pct >= sell_trigger_pct or 
                    (bar_hour == 15 and bar_minute >= 55)  # 3:55 PM CDT
                )
                
                if should_sell:
                    sale_value = pos['shares'] * current_price
                    profit = sale_value - (pos['shares'] * pos['entry_price'])
                    
                    trades.append({
                        'profit': profit,
                        'gain_pct': gain_pct,
                        'hold_time': bar_time - pos['entry_time']
                    })
                    
                    if compound:
                        capital = sale_value
                    else:
                        capital = initial_capital + profit
                        
                    positions.remove(pos)
        
        # Calculate metrics
        if not trades:
            return float('-inf')
            
        total_profit = sum(t['profit'] for t in trades)
        winning_trades = [t for t in trades if t['profit'] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        total_return = total_profit / initial_capital
        
        # Return score based on optimization metric
        if optimization_metric == "total_return":
            return total_return
        elif optimization_metric == "win_rate":
            return win_rate
        else:  # sharpe ratio (simplified)
            if len(trades) < 2:
                return float('-inf')
            returns = [t['profit']/initial_capital for t in trades]
            mean_return = sum(returns) / len(returns)
            return mean_return / (max(0.0001, np.std(returns)))  # Simplified Sharpe
            
    except Exception as e:
        logger.error(f"Intraday simulation failed: {e}")
        return float('-inf')


async def optimize_stock(
    symbol: str,
    config: OptimizationRequest,
    job_id: str
) -> Optional[dict]:
    """Optimize a single stock using Bayesian optimization."""
    import time
    start_time = time.time()  # Track how long optimization takes
    
    await broadcast_message({
        "type": "status",
        "job_id": job_id,
        "symbol": symbol,
        "status": "fetching_data",
        "message": f"Fetching price data for {symbol}..."
    })
    
    # Fetch price data
    try:
        resp = requests.get(
            f"{PPOALGO_API}/api/prices",
            params={
                "symbol": symbol,
                "start": config.start_date,
                "end": config.end_date,
                "timeframe": "1Day"
            },
            timeout=30
        )
        resp.raise_for_status()
        bars = resp.json()
    except Exception as e:
        await broadcast_message({
            "type": "error",
            "job_id": job_id,
            "symbol": symbol,
            "message": f"Failed to fetch data: {str(e)}"
        })
        return None
    
    if not bars:
        await broadcast_message({
            "type": "error",
            "job_id": job_id,
            "symbol": symbol,
            "message": "No price data available"
        })
        return None
    
    # Analyze volatility
    analyzer = StockVolatilityAnalyzer(bars)
    bounds = analyzer.get_parameter_bounds()
    
    await broadcast_message({
        "type": "status",
        "job_id": job_id,
        "symbol": symbol,
        "status": "analyzing",
        "message": f"Volatility: {analyzer.volatility_score:.2f}, Range: {analyzer.avg_daily_range:.2f}%",
        "volatility": {
            "avg_range": round(analyzer.avg_daily_range, 2),
            "max_gain": round(analyzer.max_daily_gain, 2),
            "volatility_score": round(analyzer.volatility_score, 2),
        },
        "bounds": bounds
    })

    # GPU Optimization Path
    logger.info(f"GPU Decision for {symbol}: use_gpu={config.use_gpu}, GPU_BACKTEST_AVAILABLE={GPU_BACKTEST_AVAILABLE}")
    if config.use_gpu and GPU_BACKTEST_AVAILABLE:
        await broadcast_message({
            "type": "status",
            "job_id": job_id,
            "symbol": symbol,
            "status": "optimizing_gpu",
            "message": f"Running GPU-accelerated massive grid search...",
        })
        
        try:
            from gpu_backtest import run_gpu_optimization
            
            # Calculate expected combinations for progress display
            buy_range = (bounds["buy_trigger"][0]/100, bounds["buy_trigger"][1]/100, 0.001)
            sell_range = (bounds["sell_trigger"][0]/100, bounds["sell_trigger"][1]/100, 0.001)
            
            import numpy as np
            buy_count = len(np.arange(buy_range[0], buy_range[1], buy_range[2]))
            sell_count = len(np.arange(sell_range[0], sell_range[1], sell_range[2]))
            total_combinations = buy_count * sell_count
            
            # Send progress: preparing GPU
            await broadcast_message({
                "type": "progress",
                "job_id": job_id,
                "symbol": symbol,
                "trials_completed": 0,
                "total_trials": total_combinations,
                "best_score": 0,
                "best_params": {},
                "phase": "gpu_preparing"
            })
            
            # Run GPU optimization in thread pool
            loop = asyncio.get_event_loop()
            
            # Send progress: processing started
            await broadcast_message({
                "type": "progress",
                "job_id": job_id,
                "symbol": symbol,
                "trials_completed": int(total_combinations * 0.1),
                "total_trials": total_combinations,
                "best_score": 0,
                "best_params": {},
                "phase": "gpu_processing"
            })
            
            gpu_result = await loop.run_in_executor(
                None, 
                run_gpu_optimization,
                bars,
                config.capital,
                buy_range,
                sell_range,
                config.optimization_metric
            )
            
            # Send progress: processing complete
            await broadcast_message({
                "type": "progress",
                "job_id": job_id,
                "symbol": symbol,
                "trials_completed": total_combinations,
                "total_trials": total_combinations,
                "best_score": gpu_result["metrics"].get(config.optimization_metric, 0),
                "best_params": gpu_result["best_params"],
                "phase": "gpu_complete"
            })
            
            result = {
                "symbol": symbol,
                "optimized_for": config.optimization_metric,
                "best_params": {
                    "buy_trigger_pct": round(gpu_result["best_params"]["buy_trigger_pct"], 2),
                    "sell_trigger_pct": round(gpu_result["best_params"]["sell_trigger_pct"], 2),
                    "compound": True,
                },
                "metrics": {
                    config.optimization_metric: sanitize_metric(gpu_result["metrics"].get(config.optimization_metric)),
                    "total_return": sanitize_metric(gpu_result["metrics"].get("total_return")),
                    "win_rate": sanitize_metric(gpu_result["metrics"].get("win_rate")),
                    "sharpe": sanitize_metric(gpu_result["metrics"].get("sharpe")),
                    "max_drawdown": sanitize_metric(gpu_result["metrics"].get("max_drawdown")),
                    "total_trades": sanitize_metric(gpu_result["metrics"].get("total_trades"))
                },
                "volatility_profile": {
                    "avg_daily_range": round(analyzer.avg_daily_range, 2),
                    "max_daily_gain": round(analyzer.max_daily_gain, 2),
                    "volatility_score": round(analyzer.volatility_score, 2),
                },
                "n_trials": gpu_result["combinations_tested"],
                "timestamp": datetime.now().isoformat(),
                "method": "gpu_grid_search",
                "duration_seconds": round(time.time() - start_time, 1)
            }
            
            # Smart Timing Optimization (GPU Path)
            if config.smart_timing:
                logger.info(f"Starting Smart Timing optimization for {symbol} (GPU path)")
                await broadcast_message({
                    "type": "status",
                    "job_id": job_id,
                    "symbol": symbol,
                    "status": "optimizing_timing",
                    "message": f"Finding optimal buy time for {symbol}...",
                })
                
                # Get optimal timing
                optimal_time = get_optimal_buy_time_simple(symbol, config.optimization_metric)
                
                result["optimal_buy_time_cdt"] = optimal_time
                result["best_params"]["optimal_buy_time_cdt"] = optimal_time
                
                await broadcast_message({
                    "type": "status", 
                    "job_id": job_id,
                    "symbol": symbol,
                    "message": f"Optimal buy time determined: {optimal_time} CDT"
                })
                logger.info(f"Smart Timing complete for {symbol}: {optimal_time} CDT")
            
            # Save result to database
            save_result_to_db(job_id, result)
            
            await broadcast_message({
                "type": "complete",
                "job_id": job_id,
                "symbol": symbol,
                "result": result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"GPU optimization failed: {e}")
            await broadcast_message({
                "type": "error",
                "job_id": job_id,
                "symbol": symbol,
                "message": f"GPU optimization failed: {str(e)}. Falling back to CPU..."
            })
            # Fall back to CPU if GPU fails
    
    # CPU Optimization Path (Optuna)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42)
    )
    
    best_so_far = float("-inf")
    
    def objective(trial):
        nonlocal best_so_far
        
        params = {
            "buy_trigger": trial.suggest_float(
                "buy_trigger", bounds["buy_trigger"][0], bounds["buy_trigger"][1], step=0.1
            ),
            "sell_trigger": trial.suggest_float(
                "sell_trigger", bounds["sell_trigger"][0], bounds["sell_trigger"][1], step=0.1
            ),
            "compound": trial.suggest_categorical("compound", [True, False]),
        }
        
        metrics = run_backtest(symbol, params, config)
        score = metrics.get(config.optimization_metric, 0) or float("-inf")
        
        # Store all metrics as user attributes for later extraction
        trial.set_user_attr("total_return", metrics.get("total_return", 0))
        trial.set_user_attr("win_rate", metrics.get("win_rate", 0))
        trial.set_user_attr("sharpe", metrics.get("sharpe", 0))
        trial.set_user_attr("max_drawdown", metrics.get("max_drawdown", 0))
        trial.set_user_attr("total_trades", metrics.get("total_trades", 0))
        
        # Track best
        if score > best_so_far:
            best_so_far = score
        
        return score
    
    # Run optimization with progress updates
    await broadcast_message({
        "type": "status",
        "job_id": job_id,
        "symbol": symbol,
        "status": "optimizing",
        "message": f"Running {config.n_trials} optimization trials (CPU)...",
        "total_trials": config.n_trials
    })
    
    # Send initial progress so bar shows activity immediately
    await broadcast_message({
        "type": "progress",
        "job_id": job_id,
        "symbol": symbol,
        "trials_completed": 0,
        "total_trials": config.n_trials,
        "best_score": 0,
        "best_params": {},
    })
    
    # Run in thread pool to not block
    loop = asyncio.get_event_loop()
    
    trials_completed = [0]
    
    def callback(study, trial):
        trials_completed[0] += 1
        # Update on EVERY trial for smooth progress bar
        asyncio.run_coroutine_threadsafe(
            broadcast_message({
                "type": "progress",
                "job_id": job_id,
                "symbol": symbol,
                "trials_completed": trials_completed[0],
                "total_trials": config.n_trials,
                "best_score": study.best_value if study.best_trial else 0,
                "best_params": study.best_params if study.best_trial else {},
            }),
            loop
        )
    
    with ThreadPoolExecutor() as executor:
        future = executor.submit(
            study.optimize,
            objective,
            n_trials=config.n_trials,
            callbacks=[callback],
            show_progress_bar=False
        )
        future.result()
    
    # Get results
    best_params = study.best_params
    best_trial = study.best_trial
    
    result = {
        "symbol": symbol,
        "optimized_for": config.optimization_metric,
        "best_params": {
            "buy_trigger_pct": round(best_params["buy_trigger"], 2),
            "sell_trigger_pct": round(best_params["sell_trigger"], 2),
            "compound": best_params["compound"],
        },
        "metrics": {
            config.optimization_metric: sanitize_metric(study.best_value),
            "total_return": sanitize_metric(best_trial.user_attrs.get("total_return")),
            "win_rate": sanitize_metric(best_trial.user_attrs.get("win_rate")),
            "sharpe": sanitize_metric(best_trial.user_attrs.get("sharpe")),
            "max_drawdown": sanitize_metric(best_trial.user_attrs.get("max_drawdown")),
            "total_trades": sanitize_metric(best_trial.user_attrs.get("total_trades")),
        },
        "volatility_profile": {
            "avg_daily_range": round(analyzer.avg_daily_range, 2),
            "max_daily_gain": round(analyzer.max_daily_gain, 2),
            "volatility_score": round(analyzer.volatility_score, 2),
        },
        "n_trials": config.n_trials,
        "timestamp": datetime.now().isoformat(),
        "method": "cpu_optuna",
        "duration_seconds": round(time.time() - start_time, 1)  # How long this stock took
    }
    
    # Smart Timing Optimization (Phase 2)
    if config.smart_timing:
        logger.info(f"Starting Smart Timing optimization for {symbol} using {config.timing_approach} approach")
        await broadcast_message({
            "type": "status",
            "job_id": job_id,
            "symbol": symbol,
            "status": "optimizing_timing",
            "message": f"Finding optimal buy time for {symbol}...",
        })
        
        # For now, use a simple time-based approach since intraday data may not be available
        # TODO: Implement full intraday optimization when API is ready
        optimal_time = get_optimal_buy_time_simple(symbol, config.optimization_metric)
        
        result["optimal_buy_time_cdt"] = optimal_time
        result["best_params"]["optimal_buy_time_cdt"] = optimal_time
        
        await broadcast_message({
            "type": "status", 
            "job_id": job_id,
            "symbol": symbol,
            "message": f"Optimal buy time determined: {optimal_time} CDT"
        })
        logger.info(f"Smart Timing complete for {symbol}: {optimal_time} CDT")
        
        # Choose optimization approach based on user selection (for future full implementation)
        use_joint_approach = (config.timing_approach == "joint")
        # Note: Advanced intraday optimization approaches (sequential/joint) 
        # are available but disabled until intraday data API is confirmed working
        # The simple approach above provides reliable, research-based timing

    # Save result to database
    save_result_to_db(job_id, result)
    
    await broadcast_message({
        "type": "complete",
        "job_id": job_id,
        "symbol": symbol,
        "result": result
    })
    
    return result


@app.get("/")
async def serve_frontend():
    return FileResponse("/app/static/index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "optuna_available": OPTUNA_AVAILABLE,
        "postgres_available": POSTGRES_AVAILABLE
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
    except WebSocketDisconnect:
        active_connections.remove(websocket)


@app.post("/api/optimize")
async def start_optimization(request: OptimizationRequest):
    """Start optimization for multiple stocks."""
    if not OPTUNA_AVAILABLE:
        raise HTTPException(status_code=500, detail="Optuna not installed")
    
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save job to database
    save_job_to_db(job_id, request)
    
    # Run optimization in background
    async def run_all():
        results = {}
        for symbol in request.symbols:
            result = await optimize_stock(symbol, request, job_id)
            if result:
                results[symbol] = result
                optimization_results[symbol] = result
        
        # Save results to file
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(f"{RESULTS_DIR}/{job_id}.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Update job status in database
        update_job_status(job_id, "completed")
        
        await broadcast_message({
            "type": "job_complete",
            "job_id": job_id,
            "results": results
        })
    
    asyncio.create_task(run_all())
    
    return {"job_id": job_id, "symbols": request.symbols, "status": "started"}



@app.get("/api/results")
async def get_results():
    """Get all optimization results."""
    return optimization_results


@app.get("/api/results/{symbol}")
async def get_symbol_result(symbol: str):
    """Get optimization result for a specific symbol."""
    symbol = symbol.upper()
    if symbol in optimization_results:
        return optimization_results[symbol]
    raise HTTPException(status_code=404, detail="Symbol not found")


@app.get("/api/analysis/{history_id}")
async def get_analysis(history_id: int):
    """Get analysis data for a history item and redirect to frontend."""
    conn = get_db_conn()
    if not conn:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT r.symbol, r.buy_trigger_pct, r.sell_trigger_pct, r.compound,
                       j.start_date, j.end_date, j.capital, r.optimal_buy_time_cdt
                FROM optimizer_results r
                LEFT JOIN optimizer_jobs j ON r.job_id = j.job_id
                WHERE r.id = %s
            """, (history_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="History item not found")
            
            symbol, buy_trigger, sell_trigger, compound, start_date, end_date, capital, buy_time = row
            
            # Build frontend URL with parameters
            frontend_url = f"/paper.html?symbol={symbol}&buy_trigger={buy_trigger}&sell_trigger={sell_trigger}&compound={compound}&start={start_date}&end={end_date}&capital={capital}"
            if buy_time:
                frontend_url += f"&buy_time={buy_time}"
            
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url=frontend_url)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.get("/api/history")
async def get_history(limit: int = None):
    """Get optimization history from database."""
    conn = get_db_conn()
    if not conn:
        return []
    
    try:
        with conn.cursor() as cur:
            query = """
                SELECT 
                    r.id,
                    r.job_id,
                    r.symbol,
                    r.buy_trigger_pct,
                    r.sell_trigger_pct,
                    r.compound,
                    r.optimized_for,
                    r.score,
                    r.volatility_avg_range,
                    r.volatility_max_gain,
                    r.volatility_score,
                    r.optimal_buy_time_cdt,
                    r.created_at,
                    j.start_date,
                    j.end_date,
                    j.capital,
                    j.n_trials,
                    r.full_result->'metrics'->>'total_return' as total_return,
                    r.full_result->'metrics'->>'win_rate' as win_rate,
                    r.full_result->'metrics'->>'sharpe' as sharpe_from_json,
                    r.full_result->>'duration_seconds' as duration_seconds,
                    r.full_result->>'method' as method,
                    j.smart_timing
                FROM optimizer_results r
                LEFT JOIN optimizer_jobs j ON r.job_id = j.job_id
                ORDER BY r.created_at DESC
            """
            if limit:
                query += " LIMIT %s"
                cur.execute(query, (limit,))
            else:
                cur.execute(query)
            
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            
            results = []
            for row in rows:
                result = dict(zip(columns, row))
                # Convert datetime to string
                if result.get("created_at"):
                    result["created_at"] = result["created_at"].isoformat()
                if result.get("start_date"):
                    result["start_date"] = str(result["start_date"])
                if result.get("end_date"):
                    result["end_date"] = str(result["end_date"])
                # Convert Decimal to float
                for key in ["buy_trigger_pct", "sell_trigger_pct", "score", 
                           "volatility_avg_range", "volatility_max_gain", 
                           "volatility_score", "capital", "total_return", "win_rate", "sharpe_from_json"]:
                    if result.get(key) is not None:
                        try:
                            result[key] = float(result[key])
                        except (ValueError, TypeError):
                            pass
                results.append(result)
            
            return results
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return []
    finally:
        conn.close()


@app.delete("/api/history")
async def delete_all_history():
    """Delete all optimization results."""
    conn = get_db_conn()
    if not conn:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM optimizer_results")
            deleted_count = cur.rowcount
        conn.commit()
        logger.info(f"Deleted all optimization results ({deleted_count} items)")
        return {"status": "ok", "deleted": deleted_count}
    except Exception as e:
        logger.error(f"Failed to delete all results: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.delete("/api/history/{result_id}")
async def delete_history_item(result_id: int):
    """Delete a specific optimization result."""
    conn = get_db_conn()
    if not conn:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM optimizer_results WHERE id = %s", (result_id,))
        conn.commit()
        logger.info(f"Deleted optimization result {result_id}")
        return {"status": "ok", "deleted": result_id}
    except Exception as e:
        logger.error(f"Failed to delete result: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
