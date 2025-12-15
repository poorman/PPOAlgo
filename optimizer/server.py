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
            
            # Price cache table for storing downloaded price data
            cur.execute("""
                CREATE TABLE IF NOT EXISTS price_cache (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    bar_date DATE NOT NULL,
                    bar_time TIMESTAMPTZ,
                    open_price NUMERIC(12, 4) NOT NULL,
                    high_price NUMERIC(12, 4) NOT NULL,
                    low_price NUMERIC(12, 4) NOT NULL,
                    close_price NUMERIC(12, 4) NOT NULL,
                    volume BIGINT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(symbol, timeframe, bar_date, bar_time)
                );
            """)
            
            # Create indexes for price cache
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_cache_symbol_timeframe 
                ON price_cache(symbol, timeframe);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_cache_symbol_date 
                ON price_cache(symbol, timeframe, bar_date);
            """)
            
            # Price cache metadata table to track what data we have
            cur.execute("""
                CREATE TABLE IF NOT EXISTS price_cache_meta (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    min_date DATE NOT NULL,
                    max_date DATE NOT NULL,
                    bar_count INTEGER NOT NULL,
                    last_updated TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(symbol, timeframe)
                );
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
            
            # Get algo from result, default to 'Default'
            algo = result.get("algo", "Default")
            
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
        logger.info(f"Saved result for {result['symbol']} (algo: {algo}) to database")
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


# ============================================================================
# PRICE CACHE FUNCTIONS
# ============================================================================

def get_cached_price_range(symbol: str, timeframe: str) -> tuple:
    """Get the date range of cached prices for a symbol/timeframe."""
    conn = get_db_conn()
    if not conn:
        return None, None
    
    try:
        with conn.cursor() as cur:
            # Check price_cache_meta first
            cur.execute("""
                SELECT min_date, max_date FROM price_cache_meta
                WHERE symbol = %s AND timeframe = %s
            """, (symbol.upper(), timeframe))
            row = cur.fetchone()
            if row and row[0]:
                return row[0], row[1]
            
            # Check legacy market_daily_bars table for daily data
            if timeframe == "1Day":
                cur.execute("""
                    SELECT MIN(bar_date), MAX(bar_date) FROM market_daily_bars
                    WHERE ticker = %s
                """, (symbol.upper(),))
                row = cur.fetchone()
                if row and row[0]:
                    logger.info(f"Found legacy daily data for {symbol}: {row[0]} to {row[1]}")
                    return row[0], row[1]
            
            return None, None
    except Exception as e:
        logger.error(f"Failed to get cached price range: {e}")
        return None, None
    finally:
        conn.close()


def get_cached_prices(symbol: str, timeframe: str, start_date: str, end_date: str) -> list:
    """Get cached prices from database. Checks both price_cache and legacy market_daily_bars tables."""
    conn = get_db_conn()
    if not conn:
        return []
    
    try:
        bars = []
        with conn.cursor() as cur:
            if timeframe == "1Day":
                # First try price_cache table (using actual column names)
                cur.execute("""
                    SELECT bar_date, open, high, low, close, volume
                    FROM price_cache
                    WHERE symbol = %s AND timeframe = %s 
                    AND bar_date >= %s AND bar_date <= %s
                    ORDER BY bar_date
                """, (symbol.upper(), timeframe, start_date, end_date))
                rows = cur.fetchall()
                
                # If no data in price_cache, try legacy market_daily_bars table
                if not rows:
                    cur.execute("""
                        SELECT bar_date, open, high, low, close, volume
                        FROM market_daily_bars
                        WHERE ticker = %s 
                        AND bar_date >= %s AND bar_date <= %s
                        ORDER BY bar_date
                    """, (symbol.upper(), start_date, end_date))
                    rows = cur.fetchall()
                    if rows:
                        logger.info(f"Found {len(rows)} bars in legacy market_daily_bars for {symbol}")
                
                for row in rows:
                    bars.append({
                        "t": row[0].strftime("%Y-%m-%dT05:00:00Z"),
                        "o": float(row[1]),
                        "h": float(row[2]),
                        "l": float(row[3]),
                        "c": float(row[4]),
                        "v": float(row[5]) if row[5] else 0
                    })
            else:
                # For intraday, check price_cache first (using actual column names)
                cur.execute("""
                    SELECT bar_timestamp, open, high, low, close, volume
                    FROM price_cache
                    WHERE symbol = %s AND timeframe = %s 
                    AND bar_date >= %s AND bar_date <= %s
                    ORDER BY bar_timestamp
                """, (symbol.upper(), timeframe, start_date, end_date))
                rows = cur.fetchall()
                
                # Try legacy market_minute_bars if no data
                if not rows:
                    cur.execute("""
                        SELECT bar_date, open, high, low, close, volume
                        FROM market_minute_bars
                        WHERE ticker = %s 
                        AND bar_date >= %s AND bar_date <= %s
                        ORDER BY bar_date
                    """, (symbol.upper(), start_date, end_date))
                    rows = cur.fetchall()
                    if rows:
                        logger.info(f"Found {len(rows)} bars in legacy market_minute_bars for {symbol}")
                
                for row in rows:
                    if isinstance(row[0], str):
                        bars.append({
                            "t": row[0],
                            "o": float(row[1]),
                            "h": float(row[2]),
                            "l": float(row[3]),
                            "c": float(row[4]),
                            "v": float(row[5]) if row[5] else 0
                        })
                    else:
                        bars.append({
                            "t": row[0].isoformat() if row[0] else "",
                            "o": float(row[1]),
                            "h": float(row[2]),
                            "l": float(row[3]),
                            "c": float(row[4]),
                            "v": float(row[5]) if row[5] else 0
                        })
            return bars
    except Exception as e:
        logger.error(f"Failed to get cached prices: {e}")
        return []
    finally:
        conn.close()


def save_prices_to_cache(symbol: str, timeframe: str, bars: list):
    """Save prices to cache database."""
    if not bars:
        return
    
    conn = get_db_conn()
    if not conn:
        return
    
    try:
        with conn.cursor() as cur:
            # Insert bars
            for bar in bars:
                bar_time_str = bar.get("t", "")
                if not bar_time_str:
                    continue
                
                # Parse date from timestamp
                bar_date = bar_time_str[:10]
                
                # Use actual column names from the existing table
                cur.execute("""
                    INSERT INTO price_cache 
                    (symbol, timeframe, bar_date, bar_timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timeframe, bar_date, bar_timestamp) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """, (
                    symbol.upper(),
                    timeframe,
                    bar_date,
                    bar_time_str,
                    bar.get("o", 0),
                    bar.get("h", 0),
                    bar.get("l", 0),
                    bar.get("c", 0),
                    bar.get("v", 0)
                ))
            
            # Update metadata
            cur.execute("""
                SELECT MIN(bar_date), MAX(bar_date), COUNT(*) 
                FROM price_cache 
                WHERE symbol = %s AND timeframe = %s
            """, (symbol.upper(), timeframe))
            row = cur.fetchone()
            if row and row[0]:
                cur.execute("""
                    INSERT INTO price_cache_meta (symbol, timeframe, min_date, max_date, bar_count, last_updated)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (symbol, timeframe) DO UPDATE SET
                    min_date = EXCLUDED.min_date,
                    max_date = EXCLUDED.max_date,
                    bar_count = EXCLUDED.bar_count,
                    last_updated = NOW()
                """, (symbol.upper(), timeframe, row[0], row[1], row[2]))
        
        conn.commit()
        logger.info(f"Saved {len(bars)} bars to cache for {symbol} ({timeframe})")
    except Exception as e:
        logger.error(f"Failed to save prices to cache: {e}")
        conn.rollback()
    finally:
        conn.close()


def fetch_and_cache_prices(symbol: str, start_date: str, end_date: str, timeframe: str = "1Day") -> list:
    """
    Fetch prices with caching. Only downloads missing data.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: "1Day" or "1Min"
    
    Returns:
        List of price bars
    """
    from datetime import datetime, timedelta
    
    symbol = symbol.upper()
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Check what we have cached
    cached_min, cached_max = get_cached_price_range(symbol, timeframe)
    
    logger.info(f"Price cache for {symbol} ({timeframe}): {cached_min} to {cached_max}")
    
    
    # Determine what data we need to fetch
    fetch_ranges = []
    
    if cached_min is None:
        # No cached data, fetch everything
        fetch_ranges.append((start_date, end_date))
    else:
        # Check if we need data before cached range
        if start_dt.date() < cached_min:
            fetch_ranges.append((start_date, (cached_min - timedelta(days=1)).strftime("%Y-%m-%d")))
        
        # Check if we need data after cached range
        if end_dt.date() > cached_max:
            fetch_ranges.append(((cached_max + timedelta(days=1)).strftime("%Y-%m-%d"), end_date))
    
    
    # Fetch missing data
    for fetch_start, fetch_end in fetch_ranges:
        logger.info(f"Fetching {symbol} ({timeframe}) from {fetch_start} to {fetch_end}")
        
        if timeframe == "1Day":
            # Fetch daily data in one request
            try:
                resp = requests.get(
                    f"{PPOALGO_API}/api/prices",
                    params={
                        "symbol": symbol,
                        "start": fetch_start,
                        "end": fetch_end,
                        "timeframe": "1Day"
                    },
                    timeout=60
                )
                if resp.status_code == 200:
                    bars = resp.json()
                    if bars:
                        save_prices_to_cache(symbol, timeframe, bars)
                        logger.info(f"Cached {len(bars)} daily bars for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch daily prices: {e}")
        else:
            # Fetch intraday data in monthly chunks (API limitation)
            current = datetime.strptime(fetch_start, "%Y-%m-%d")
            fetch_end_dt = datetime.strptime(fetch_end, "%Y-%m-%d")
            
            while current <= fetch_end_dt:
                chunk_end = min(current + timedelta(days=30), fetch_end_dt)
                
                try:
                    resp = requests.get(
                        f"{PPOALGO_API}/api/prices",
                        params={
                            "symbol": symbol,
                            "start": current.strftime("%Y-%m-%d"),
                            "end": chunk_end.strftime("%Y-%m-%d"),
                            "timeframe": "1Min"
                        },
                        timeout=60
                    )
                    if resp.status_code == 200:
                        bars = resp.json()
                        if bars and len(bars) > 0:
                            # Only save if we got intraday data (not daily fallback)
                            if "T09:" in bars[0].get("t", "") or "T1" in bars[0].get("t", ""):
                                save_prices_to_cache(symbol, timeframe, bars)
                except Exception as e:
                    logger.error(f"Failed to fetch intraday prices: {e}")
                
                current = chunk_end + timedelta(days=1)
    
    # Return cached data for the requested range
    result = get_cached_prices(symbol, timeframe, start_date, end_date)
    return result


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
    algo: str = "default"  # "default" or "chatgpt"


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


def generate_trade_log(bars: list, buy_trigger_pct: float, sell_trigger_pct: float, compound: bool, capital: float = 100000, algo: str = "default", symbol: str = None, start_date: str = None, end_date: str = None) -> list:
    """
    Generate a day-by-day trade log for the Analysis view with buy/sell guidance.
    
    Args:
        bars: List of daily OHLC bars
        buy_trigger_pct: Buy trigger percentage (e.g., 3.21 for 3.21%)
        sell_trigger_pct: Sell trigger percentage (e.g., 2.0 for 2.0%)
        compound: Whether to compound wins
        capital: Starting capital
        algo: Algorithm type - "default" or "chatgpt"
        symbol: Stock symbol (needed for intraday data)
        start_date: Start date (needed for intraday data)
        end_date: End date (needed for intraday data)
        
    Returns:
        List of trade entries with day-by-day data and guidance
    """
    trade_log = []
    equity = capital
    
    
    # For ChatGPT algo, try to fetch 9AM prices (using cache)
    price_9am_map = {}
    if algo == "chatgpt" and symbol and start_date and end_date:
        try:
            logger.info(f"Fetching 9AM prices for {symbol} (ChatGPT algo) using cache")
            
            # Use cached intraday data
            intraday_bars = fetch_and_cache_prices(symbol, start_date, end_date, "1Min")
            
            
            if intraday_bars:
                # Build map of date -> 9AM price (9 AM Eastern = 14:00 UTC winter, 13:00 UTC summer)
                # Look for first trade between 13:00-15:00 UTC to cover both EST and EDT
                for bar in intraday_bars:
                    bar_time = bar.get("t", "")
                    date_part = bar_time[:10] if bar_time else ""
                    
                    # Skip if we already have a price for this date
                    if date_part in price_9am_map:
                        continue
                    
                    # Extract hour from timestamp (handles both "T14:00:00Z" and "T14:00:00+00:00" formats)
                    try:
                        hour_str = bar_time[11:13] if len(bar_time) > 13 else ""
                        hour = int(hour_str) if hour_str.isdigit() else -1
                        
                        # Accept trades between 13:00-15:00 UTC (9-11 AM Eastern)
                        if 13 <= hour <= 15:
                            price_9am_map[date_part] = bar.get("o", bar.get("c", 0))
                    except:
                        pass
                
                
                logger.info(f"Found 9AM prices for {len(price_9am_map)} days from cache")
            else:
                logger.warning(f"No intraday data available for {symbol}")
        except Exception as e:
            logger.warning(f"Could not fetch 9AM prices: {e}")
    
    for i, bar in enumerate(bars):  # Process all days in the period
        open_price = bar.get("o", bar.get("c", 0))
        close_price = bar.get("c", 0)
        high_price = bar.get("h", close_price)
        low_price = bar.get("l", close_price)
        # Get date from bar timestamp
        bar_date = bar.get("t", "")
        date_key = bar_date[:10] if bar_date else ""
        
        # Get 9AM price if available
        price_9am = price_9am_map.get(date_key)
        
        
        if open_price <= 0:
            continue
        
        # For ChatGPT algo: Buy at 9AM if price increased by at least buy_trigger_pct
        if algo == "chatgpt":
            # Calculate % change from open to 9AM
            pct_change_9am = ((price_9am - open_price) / open_price * 100) if price_9am and open_price > 0 else 0
            
            # ChatGPT strategy: buy at 9AM ONLY if price increased by at least buy_trigger_pct%
            up_threshold = open_price * (1 + buy_trigger_pct / 100)
            if price_9am and price_9am >= up_threshold:
                bought = True
                buy_price_actual = price_9am
            else:
                bought = False
                buy_price_actual = None
            
            
            # Sell target from 9AM price
            sell_price_target = buy_price_actual * (1 + sell_trigger_pct / 100) if buy_price_actual else None
            
            # Did we hit sell target?
            sold_at_target = bought and sell_price_target and high_price >= sell_price_target
            
            shares = 0
            profit = 0
            actual_sell_price = None
            
            if bought and buy_price_actual:
                shares = int(equity / buy_price_actual) if buy_price_actual > 0 else 0
                
                if shares > 0:
                    if sold_at_target:
                        actual_sell_price = sell_price_target
                    else:
                        actual_sell_price = close_price  # Sell at close
                    
                    profit = shares * (actual_sell_price - buy_price_actual)
                    
                    if compound:
                        equity += profit
            
            trade_log.append({
                "date": bar_date,
                "open": round(open_price, 2),
                "price_9am": round(price_9am, 2) if price_9am else None,
                "pct_change_9am": round(pct_change_9am, 2) if price_9am else None,
                "close": round(close_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "pct_change": round(((close_price - open_price) / open_price * 100), 2) if open_price > 0 else 0,
                "bought": bought,
                "sold_at_target": sold_at_target,
                "buy_price": round(buy_price_actual, 2) if buy_price_actual else None,
                "sell_price": round(sell_price_target, 2) if sell_price_target else None,
                "actual_sell": round(actual_sell_price, 2) if actual_sell_price else None,
                "shares": shares,
                "profit": round(profit, 0),
                "equity": round(equity, 0),
                "algo": "chatgpt"
            })
        else:
            # Default algo: buy on dip
            # Calculate % change from open to close
            pct_change_open_close = ((close_price - open_price) / open_price) * 100
            
            # Calculate buy price target (price must drop this much to trigger buy)
            buy_price_target = open_price * (1 - buy_trigger_pct / 100)
            
            # Check if buy was triggered (low went below buy target)
            bought = low_price <= buy_price_target
            
            # Calculate sell price target (from buy price)
            sell_price_target = buy_price_target * (1 + sell_trigger_pct / 100)
            
            # Did we hit sell target?
            sold_at_target = bought and high_price >= sell_price_target
            
            shares = 0
            profit = 0
            actual_sell_price = None
            
            if bought:
                # Calculate how many shares we can buy
                buy_price = buy_price_target  # Assume we buy at trigger price
                shares = int(equity / buy_price) if buy_price > 0 else 0
                
                if shares > 0:
                    if sold_at_target:
                        # Sold at target
                        actual_sell_price = sell_price_target
                    else:
                        # Sold at close
                        actual_sell_price = close_price
                    
                    profit = shares * (actual_sell_price - buy_price)
                    
                    if compound:
                        equity += profit
            
            trade_log.append({
                "date": bar_date,
                "open": round(open_price, 2),
                "price_9am": round(price_9am, 2) if price_9am else None,
                "close": round(close_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "pct_change": round(pct_change_open_close, 2),
                "bought": bought,
                "sold_at_target": sold_at_target,
                "buy_price": round(buy_price_target, 2),
                "sell_price": round(sell_price_target, 2),
                "actual_sell": round(actual_sell_price, 2) if actual_sell_price else None,
                "shares": shares,
                "profit": round(profit, 0),
                "equity": round(equity, 0),
                "algo": "default"
            })
    
    return trade_log


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
    import sys
    start_time = time.time()  # Track how long optimization takes
    
    print(f"[OPTIMIZE] Starting optimization for {symbol}, algo={config.algo}", flush=True)
    sys.stdout.flush()
    
    await broadcast_message({
        "type": "status",
        "job_id": job_id,
        "symbol": symbol,
        "status": "fetching_data",
        "message": f"Fetching price data for {symbol} (using cache)..."
    })
    
    # Fetch price data with caching
    try:
        bars = fetch_and_cache_prices(symbol, config.start_date, config.end_date, "1Day")
        
        # Fallback to direct API if cache fails
        if not bars:
            logger.warning(f"Cache miss for {symbol}, fetching directly")
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
    
    # Extract actual data range from bars
    actual_start_date = bars[0].get("t", "")[:10] if bars else None
    actual_end_date = bars[-1].get("t", "")[:10] if bars else None
    data_range_warning = None
    
    # Check if actual data range differs from requested range
    if actual_start_date and actual_start_date > config.start_date:
        data_range_warning = f"⚠️ Data only available from {actual_start_date}. Requested start {config.start_date} not available (requires Polygon.io paid plan)."
        await broadcast_message({
            "type": "warning",
            "job_id": job_id,
            "symbol": symbol,
            "message": data_range_warning
        })
        logger.warning(f"Data range mismatch for {symbol}: requested {config.start_date} but actual start is {actual_start_date}")
    
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
                "duration_seconds": round(time.time() - start_time, 1),
                "requested_start_date": config.start_date,
                "requested_end_date": config.end_date,
                "actual_start_date": actual_start_date,
                "actual_end_date": actual_end_date,
                "data_range_warning": data_range_warning
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
            
            # Add algo to result
            algo_type = getattr(config, 'algo', 'default') or 'default'
            logger.info(f"[GPU] Generating trade log for {symbol} with algo={algo_type}")
            result["algo"] = algo_type
            
            # Generate trade log for Analysis view
            result["trade_log"] = generate_trade_log(
                bars,
                result["best_params"]["buy_trigger_pct"],
                result["best_params"]["sell_trigger_pct"],
                result["best_params"]["compound"],
                config.capital,
                algo=algo_type,
                symbol=symbol,
                start_date=config.start_date,
                end_date=config.end_date
            )
            
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
        "duration_seconds": round(time.time() - start_time, 1),
        "requested_start_date": config.start_date,
        "requested_end_date": config.end_date,
        "actual_start_date": actual_start_date,
        "actual_end_date": actual_end_date,
        "data_range_warning": data_range_warning
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

    # Add algo to result
    algo_type = getattr(config, 'algo', 'default') or 'default'
    logger.info(f"Generating trade log for {symbol} with algo={algo_type}")
    result["algo"] = algo_type
    
    # Generate trade log for Analysis view
    result["trade_log"] = generate_trade_log(
        bars,
        result["best_params"]["buy_trigger_pct"],
        result["best_params"]["sell_trigger_pct"],
        result["best_params"]["compound"],
        config.capital,
        algo=algo_type,
        symbol=symbol,
        start_date=config.start_date,
        end_date=config.end_date
    )
    
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


@app.get("/api/price-cache/check/{symbol}")
async def check_symbol_cache(symbol: str):
    """Check if a symbol has cached price data."""
    conn = get_db_conn()
    if not conn:
        return {"cached": False, "symbol": symbol.upper()}
    
    try:
        with conn.cursor() as cur:
            # Check price_cache_meta
            cur.execute("""
                SELECT timeframe, min_date, max_date, bar_count
                FROM price_cache_meta
                WHERE symbol = %s
            """, (symbol.upper(),))
            rows = cur.fetchall()
            
            cache_info = []
            for row in rows:
                cache_info.append({
                    "timeframe": row[0],
                    "min_date": row[1].isoformat() if row[1] else None,
                    "max_date": row[2].isoformat() if row[2] else None,
                    "bar_count": row[3]
                })
            
            # Also check legacy market_daily_bars
            cur.execute("""
                SELECT MIN(bar_date), MAX(bar_date), COUNT(*)
                FROM market_daily_bars
                WHERE ticker = %s
            """, (symbol.upper(),))
            legacy_row = cur.fetchone()
            if legacy_row and legacy_row[0]:
                cache_info.append({
                    "timeframe": "1Day (legacy)",
                    "min_date": legacy_row[0].isoformat() if legacy_row[0] else None,
                    "max_date": legacy_row[1].isoformat() if legacy_row[1] else None,
                    "bar_count": legacy_row[2]
                })
            
            return {
                "cached": len(cache_info) > 0,
                "symbol": symbol.upper(),
                "data": cache_info
            }
    except Exception as e:
        logger.error(f"Failed to check cache for {symbol}: {e}")
        return {"cached": False, "symbol": symbol.upper(), "error": str(e)}
    finally:
        conn.close()


@app.get("/api/price-cache")
async def get_price_cache_status():
    """Get price cache status for all symbols."""
    conn = get_db_conn()
    if not conn:
        return {"error": "Database connection failed"}
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, timeframe, min_date, max_date, bar_count, last_updated
                FROM price_cache_meta
                ORDER BY symbol, timeframe
            """)
            rows = cur.fetchall()
            
            cache_info = []
            for row in rows:
                cache_info.append({
                    "symbol": row[0],
                    "timeframe": row[1],
                    "min_date": row[2].isoformat() if row[2] else None,
                    "max_date": row[3].isoformat() if row[3] else None,
                    "bar_count": row[4],
                    "last_updated": row[5].isoformat() if row[5] else None
                })
            
            return {"cache": cache_info}
    except Exception as e:
        logger.error(f"Failed to get cache status: {e}")
        return {"error": str(e)}
    finally:
        conn.close()


@app.delete("/api/price-cache/{symbol}")
async def clear_price_cache(symbol: str, timeframe: str = None):
    """Clear price cache for a symbol."""
    conn = get_db_conn()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor() as cur:
            if timeframe:
                cur.execute("DELETE FROM price_cache WHERE symbol = %s AND timeframe = %s", 
                           (symbol.upper(), timeframe))
                cur.execute("DELETE FROM price_cache_meta WHERE symbol = %s AND timeframe = %s", 
                           (symbol.upper(), timeframe))
            else:
                cur.execute("DELETE FROM price_cache WHERE symbol = %s", (symbol.upper(),))
                cur.execute("DELETE FROM price_cache_meta WHERE symbol = %s", (symbol.upper(),))
        conn.commit()
        return {"status": "ok", "message": f"Cache cleared for {symbol}"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
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
                    r.full_result->>'algo' as algo,
                    r.full_result->'trade_log' as trade_log,
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
