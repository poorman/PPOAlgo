#!/usr/bin/env python3
"""
Stock Optimizer GUI Backend
FastAPI server with WebSocket support for real-time optimization updates.
Saves optimization results to PostgreSQL database.
"""

import os
import json
import config
import asyncio
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
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

# Timezone constants
ET_ZONE = ZoneInfo("America/New_York")
UTC_ZONE = ZoneInfo("UTC")


def get_et_minutes_from_timestamp(bar_time: str) -> int | None:
    """
    Convert UTC timestamp string to Eastern Time minutes since midnight.
    Properly handles DST by using America/New_York timezone.
    
    Args:
        bar_time: ISO timestamp string (e.g., "2024-01-18T15:00:00Z" or "2024-01-18T15:00:00+00:00")
    
    Returns:
        Minutes since midnight in Eastern Time, or None if parsing fails
    """
    if not bar_time or "T" not in bar_time:
        return None
    
    try:
        # Parse the timestamp - handle various formats
        ts = bar_time.replace("Z", "+00:00")
        
        # If no timezone info, assume UTC
        if "+" not in ts and ts.count("-") == 2:
            ts = ts + "+00:00"
        
        # Parse with fromisoformat (handles timezone)
        dt_utc = datetime.fromisoformat(ts)
        
        # If no timezone, assume UTC
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=UTC_ZONE)
        
        # Convert to Eastern Time (handles DST automatically)
        dt_et = dt_utc.astimezone(ET_ZONE)
        
        # Return minutes since midnight
        return dt_et.hour * 60 + dt_et.minute
    except Exception:
        # Fallback to manual parsing (legacy behavior for unusual formats)
        try:
            hour = int(bar_time[11:13])
            minute = int(bar_time[14:16])
            utc_minutes = hour * 60 + minute
            # Use EST offset (this won't handle DST but is a fallback)
            et_minutes = utc_minutes - 300
            if et_minutes < 0:
                et_minutes += 1440
            return et_minutes
        except Exception:
            return None


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
                    bar_timestamp TIMESTAMPTZ,
                    open NUMERIC(12, 4) NOT NULL,
                    high NUMERIC(12, 4) NOT NULL,
                    low NUMERIC(12, 4) NOT NULL,
                    close NUMERIC(12, 4) NOT NULL,
                    volume BIGINT,
                    vwap NUMERIC(12, 4),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(symbol, timeframe, bar_date, bar_timestamp)
                );
            """)
            
            # Add vwap column if it doesn't exist (migration for existing tables)
            cur.execute("""
                ALTER TABLE price_cache ADD COLUMN IF NOT EXISTS vwap NUMERIC(12, 4);
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
    """Save optimization result to database. Returns inserted row id."""
    conn = get_db_conn()
    if not conn:
        return None
    
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
                RETURNING id
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
            row_id = cur.fetchone()[0]
        conn.commit()
        logger.info(f"Saved result for {result['symbol']} (algo: {algo}) to database, id={row_id}")
        return row_id
    except Exception as e:
        logger.error(f"Failed to save result: {e}")
        return None
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
                    SELECT bar_timestamp, open, high, low, close, volume, vwap
                    FROM price_cache
                    WHERE symbol = %s AND timeframe = %s 
                    AND bar_date >= %s AND bar_date <= %s
                    ORDER BY bar_timestamp
                """, (symbol.upper(), timeframe, start_date, end_date))
                rows = cur.fetchall()
                # #region agent log
                import json as _json
                with open("/app/debug.log", "a") as _f:
                    _f.write(_json.dumps({"hypothesisId": "H5", "location": "server.py:447", "message": "get_cached_prices intraday query", "data": {"symbol": symbol, "timeframe": timeframe, "start": start_date, "end": end_date, "rows_from_price_cache": len(rows)}, "timestamp": int(__import__("time").time()*1000)}) + "\n")
                # #endregion
                
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
                            "v": float(row[5]) if row[5] else 0,
                            "vw": float(row[6]) if len(row) > 6 and row[6] else 0
                        })
                    else:
                        bars.append({
                            "t": row[0].isoformat() if row[0] else "",
                            "o": float(row[1]),
                            "h": float(row[2]),
                            "l": float(row[3]),
                            "c": float(row[4]),
                            "v": float(row[5]) if row[5] else 0,
                            "vw": float(row[6]) if len(row) > 6 and row[6] else 0
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
                    (symbol, timeframe, bar_date, bar_timestamp, open, high, low, close, volume, vwap)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timeframe, bar_date, bar_timestamp) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    vwap = EXCLUDED.vwap
                """, (
                    symbol.upper(),
                    timeframe,
                    bar_date,
                    bar_time_str,
                    bar.get("o", 0),
                    bar.get("h", 0),
                    bar.get("l", 0),
                    bar.get("c", 0),
                    bar.get("v", 0),
                    bar.get("vw", None)  # VWAP from Alpaca
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
    
    
    # #region agent log
    import json as _json
    with open("/app/debug.log", "a") as _f:
        _f.write(_json.dumps({"hypothesisId": "H3", "location": "server.py:600", "message": "fetch_and_cache_prices called", "data": {"symbol": symbol, "start": start_date, "end": end_date, "timeframe": timeframe, "cached_min": str(cached_min), "cached_max": str(cached_max), "fetch_ranges": [(str(a), str(b)) for a, b in fetch_ranges]}, "timestamp": int(__import__("time").time()*1000)}) + "\n")
    # #endregion
    
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
                    # #region agent log
                    with open("/app/debug.log", "a") as _f:
                        _f.write(_json.dumps({"hypothesisId": "H4", "location": "server.py:643", "message": "API response for 1Min data", "data": {"symbol": symbol, "chunk_start": current.strftime("%Y-%m-%d"), "chunk_end": chunk_end.strftime("%Y-%m-%d"), "status": resp.status_code, "num_bars": len(resp.json()) if resp.status_code == 200 else 0, "sample_t": resp.json()[0].get("t", "") if resp.status_code == 200 and resp.json() else ""}, "timestamp": int(__import__("time").time()*1000)}) + "\n")
                    # #endregion
                    if resp.status_code == 200:
                        bars = resp.json()
                        if bars and len(bars) > 0:
                            # Only save if we got intraday data (contains time component in timestamp)
                            sample_t = bars[0].get("t", "")
                            # Check for any time component (not just daily bars like "2024-01-01T05:00:00Z")
                            if "T" in sample_t and not sample_t.endswith("T05:00:00Z"):
                                save_prices_to_cache(symbol, timeframe, bars)
                except Exception as e:
                    logger.error(f"Failed to fetch intraday prices: {e}")
                
                current = chunk_end + timedelta(days=1)
    
    # Return cached data for the requested range
    result = get_cached_prices(symbol, timeframe, start_date, end_date)
    return result


# ============================================================================
# WIDESURF API DATA FETCHER
# ============================================================================

# Widesurf API configuration
WIDESURF_API_KEY = os.getenv("WIDESURF_API_KEY", "2vOf7y2q0qrEkRI4NYOIAprAhFaEoHKglZMGrG2GUgE")
WIDESURF_API_URL = os.getenv("WIDESURF_API_URL", "http://10.0.0.94:3000")

def fetch_prices_from_widesurf(symbol: str, start_date: str, end_date: str, timeframe: str = "1Min") -> list:
    """
    Fetch historical price data from Widesurf API.
    
    Args:
        symbol: Stock symbol (e.g., AAPL)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: "1Min" for minute bars, "1Day" for daily bars
    
    Returns:
        List of price bars in standard format
    """
    symbol = symbol.upper()
    bars = []
    
    # Map timeframe to Widesurf format
    if timeframe == "1Day":
        multiplier = 1
        timespan = "day"
    else:  # 1Min
        multiplier = 1
        timespan = "minute"
    
    # Build Widesurf API URL
    # Format: GET https://api.widesurf.com/v1/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
    url = f"{WIDESURF_API_URL}/v1/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000  # Get as many bars as possible
    }
    headers = {
        "X-API-KEY": WIDESURF_API_KEY
    }
    
    logger.info(f"Fetching from Widesurf API: {symbol} {timeframe} {start_date} to {end_date}")
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=60)
        
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            
            for bar in results:
                # Convert Widesurf format to our standard format
                # Widesurf returns: o, h, l, c, v, vw (VWAP), t (timestamp in ms)
                timestamp_ms = bar.get("t", 0)
                
                # Convert timestamp from ms to ISO format
                if timestamp_ms:
                    dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
                    timestamp_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    timestamp_str = ""
                
                bars.append({
                    "t": timestamp_str,
                    "o": bar.get("o", 0),
                    "h": bar.get("h", 0),
                    "l": bar.get("l", 0),
                    "c": bar.get("c", 0),
                    "v": bar.get("v", 0),
                    "vw": bar.get("vw", 0)  # VWAP
                })
            
            logger.info(f"Widesurf returned {len(bars)} bars for {symbol}")
        else:
            logger.error(f"Widesurf API error: status {resp.status_code}, response: {resp.text[:500]}")
            
    except Exception as e:
        logger.error(f"Widesurf API request failed: {e}")
    
    return bars


def fetch_prices_universal(symbol: str, start_date: str, end_date: str, timeframe: str = "1Min", data_source: str = "alpaca") -> list:
    """
    Universal price fetcher that supports multiple data sources.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: "1Min" or "1Day"
        data_source: "alpaca" (default via PPOALGO_API) or "widesurf"
    
    Returns:
        List of price bars
    """
    if data_source == "widesurf":
        return fetch_prices_from_widesurf(symbol, start_date, end_date, timeframe)
    else:
        # Use the existing Alpaca API via fetch_and_cache_prices
        return fetch_and_cache_prices(symbol, start_date, end_date, timeframe)

app = FastAPI(title="Stock Optimizer GUI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and register routers
from routers.api_tester import router as api_tester_router
from routers.history import router as history_router
from routers.ai100 import router as ai100_router
app.include_router(api_tester_router)
app.include_router(history_router)
app.include_router(ai100_router)

# Import database functions
from database import get_db_conn, ensure_tables, save_job_to_db, save_result_to_db, update_job_status, get_results_from_db, convert_numpy_types, ensure_keyword_configs_table, get_keyword_configs, save_all_keyword_configs, ensure_api_keys_table, get_api_keys, save_api_key


@app.on_event("startup")
def startup():
    """Initialize database tables on startup."""
    ensure_tables()
    ensure_keyword_configs_table()
    ensure_api_keys_table()


# ============================================================================
# WIDESURF TICKERS PROXY ENDPOINT
# ============================================================================

@app.get("/api/widesurf/tickers/{keyword}")
async def get_widesurf_tickers(keyword: str):
    """
    Proxy endpoint to fetch stock tickers from Widesurf API.
    Avoids CORS issues when frontend calls external APIs directly.
    Uses keyword configurations from the database.
    
    Args:
        keyword: One of 'SPY500', 'NASDAQ500', or '1000'
    
    Returns:
        List of stock tickers
    """
    keyword_upper = keyword.upper()
    valid_keywords = ['SPY500', 'NASDAQ500', '1000', '10000']
    
    if keyword_upper not in valid_keywords:
        return {"success": False, "error": f"Unknown keyword: {keyword}. Valid options: SPY500, NASDAQ500, 1000"}
    
    # Fetch keyword configurations from the database
    try:
        keyword_configs = get_keyword_configs()
    except Exception as e:
        logger.error(f"Failed to get keyword configs: {e}")
        keyword_configs = {}
    
    # Get the URL from keyword configuration, or use default fallback
    keyword_data = keyword_configs.get(keyword_upper, {})
    url = keyword_data.get("api_url", "")
    
    # Fallback to hardcoded defaults if not configured
    if not url or not url.startswith("http"):
        default_urls = {
            'SPY500': f"http://10.0.0.94:8020/v1/reference/tickers?market=stocks&active=true&order=desc&limit=500&sort=volume&type=limited&apiKey={WIDESURF_API_KEY}",
            'NASDAQ500': f"http://10.0.0.94:8020/v1/reference/tickers?market=stocks&active=true&order=desc&limit=500&sort=volume&type=limited&apiKey={WIDESURF_API_KEY}",
            '1000': f"http://10.0.0.94:8020/v1/reference/tickers?market=stocks&active=true&order=desc&limit=1000&sort=volume&type=limited&apiKey={WIDESURF_API_KEY}",
            '10000': f"http://10.0.0.94:8020/v1/reference/tickers?market=stocks&exchange=XNAS%2C+NYSE&active=true&order=desc&limit=10000&sort=transactions&type=limited&apiKey={WIDESURF_API_KEY}"
        }
        url = default_urls.get(keyword_upper)
        logger.info(f"Using fallback URL for {keyword_upper} (no database config found)")
    
    try:
        logger.info(f"Fetching {keyword_upper} tickers from Widesurf API: {url[:80]}...")
        resp = requests.get(url, timeout=30)
        
        if resp.status_code != 200:
            logger.error(f"Widesurf API error: {resp.status_code} - {resp.text[:500]}")
            return {"success": False, "error": f"API returned status {resp.status_code}"}
        
        data = resp.json()
        
        if not data.get("success") or not data.get("data") or not data["data"].get("stocks"):
            logger.error(f"Invalid API response format: {str(data)[:500]}")
            return {"success": False, "error": "Invalid API response format"}
        
        # Extract stock tickers from the 'pair' field
        stocks = [stock.get("pair") for stock in data["data"]["stocks"] if stock.get("pair")]
        
        logger.info(f"Fetched {len(stocks)} tickers from Widesurf API for {keyword_upper}")
        
        return {
            "success": True,
            "keyword": keyword_upper,
            "count": len(stocks),
            "stocks": stocks
        }
        
    except requests.Timeout:
        logger.error(f"Widesurf API timeout for {keyword_upper}")
        return {"success": False, "error": "API request timed out"}
    except Exception as e:
        logger.error(f"Widesurf API error: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# KEYWORD CONFIGURATIONS API
# ============================================================================

# Stock lists for static keywords
SPY_500_STOCKS = "NVDA,AAPL,GOOG,GOOGL,MSFT,AMZN,AVGO,META,TSLA,BRK.B,WMT,LLY,JPM,V,ORCL,JNJ,MA,XOM,PLTR,NFLX,BAC,ABBV,COST,AMD,HD,PG,CSCO,UNH,GE,CVX,KO,CAT,WFC,IBM,MU,MS,GS,AXP,CRM,MRK,APP,RTX,PM,MCD,TMUS,TMO,AMAT,ABT,LRCX,PEP,C,DIS,ISRG,QCOM,GEV,BX,INTC,LIN,INTU,NOW,BLK,UBER,TJX,SCHW,T,AMGN,APH,BKNG,VZ,ANET,NEE,ACN,TXN,DHR,KLAC,GILD,BA,COF,SPGI,PFE,ADBE,UNP,LOW,ADI,ETN,BSX,PGR,SYK,PANW,CRWD,DE,MDT,KKR,WELL,PLD,HON,CB,COP,CEG,PH,VRTX,HOOD,HCA,LMT,NEM,ADP,BMY,CVS,MCK,CMCSA,NKE,MO,CME,DASH,SBUX,SO,ICE,GD,DELL,CDNS,SNPS,MMC,MMM,TT,DUK,APO,MCO,WM,UPS,AMT,USB,PNC,BK,GLW,SHW,ELV,NOC,MAR,ORLY,HWM,EMR,ABNB,REGN,RCL,CTAS,GM,ITW,AON,WMB,TDG,EQIX,ECL,WBD,CI,COIN,CMI,JCI,TEL,MNST,PWR,MDLZ,CSX,FCX,SPG,FDX,STX,COR,NSC,RSG,HLT,WDC,ADSK,AJG,TFC,TRV,CL,FTNT,MSI,AEP,SLB,KMI,EOG,PCAR,ROST,VST,WDAY,NXPI,SRE,PSX,PYPL,AZO,BDX,AFL,IDXX,MPC,DLR,F,APD,LHX,MET,ALL,NDAQ,URI,O,DDOG,VLO,ZTS,EA,D,GWW,EW,PSA,ROP,FAST,CAH,CBRE,MPWR,AME,BKR,ROK,OKE,AMP,CMG,AXON,DAL,CARR,DHI,FANG,TTWO,LVS,AIG,CTVA,XEL,TGT,EXC,FICO,ETR,MSCI,PAYX,YUM,PRU,OXY,GRMN,CTSH,A,KDP,CCI,KR,TRGP,TKO,VMC,PEG,GEHC,XYZ,IQV,EBAY,NUE,MLM,EL,HIG,CPRT,MCHP,WAB,HSY,RMD,KEYS,FISV,VTR,CCL,STT,SYY,SNDK,UAL,EQT,FIS,ED,EXPE,KMB,OTIS,XYL,ACGL,WEC,ODFL,KVUE,IR,LYV,NRG,PCG,HPE,RJF,HUM,FITB,TER,FOXA,MTB,WTW,CHTR,SYF,VRSK,VICI,FOX,EXR,LEN,IBKR,FSLR,DG,MTD,KHC,ADM,EME,ROL,CSGP,HBAN,DOV,TSCO,EXE,BRO,DTE,BR,ATO,EFX,DXCM,NTRS,WRB,ULTA,AEE,CBOE,STZ,IRM,DLTR,CINF,FE,AWK,ES,OMC,BIIB,TPR,STLD,CFG,JBL,AVB,PHM,STE,PPL,GIS,HUBB,TDY,VLTO,HAL,RF,CNP,LDOS,EQR,NTAP,DVN,WAT,HPQ,PPG,TROW,VRSN,KEY,WSM,ON,RL,EIX,LULU,CPAY,LH,L,NVR,DRI,PTC,CMS,LUV,TSN,PODD,IP,SBAC,EXPD,TPL,SMCI,DGX,CTRA,PFG,CHD,NI,CNC,TRMB,SW,WST,TYL,CDW,GPN,AMCR,JBHT,CHRW,INCY,GPC,PKG,ZBH,SNA,LII,BG,TTD,Q,MKC,FTV,DOW,DD,PNR,APTV,ESS,GEN,GDDY,EVRG,IT,WY,LNT,HOLX,INVH,J,IFF,COO,MAA,ALB,BBY,PSKY,FFIV,TXT,NWS,DECK,DPZ,ERIE,NWSA,LYB,SOLV,ALLE,AVY,UHS,ZBRA,KIM,EG,IEX,JKHY,MAS,VTRS,BALL,NDSN,HRL,UDR,WYNN,HII,BXP,HST,REG,CLX,AKAM,CF,BEN,BLDR,IVZ,SWK,DOC,HAS,RVTY,ALGN,EPAM,MRNA,AIZ,CPT,GL,DAY,FDS,SJM,PNW,MGM,SWKS,AES,GNRC,BAX,CRL,AOS,TECH,NCLH,TAP,APA,PAYC,HSIC,POOL,MOH,FRT,DVA,CPB,CAG,LW,MOS,SOLS,LKQ,ARE,MTCH,MHK"

CAP_STOCK_LIST = "AAPL,MSFT,NVDA,GOOGL,GOOG,AMZN,META,BRK.A,BRK.B,LLY,AVGO,TSLA,JPM,V,UNH,XOM,MA,JNJ,HD,COST,PG,ABBV,MRK,ORCL,CRM,KO,PEP,ADBE,WMT,NFLX,AMD,LIN,CSCO,ACN,MCD,TMO,INTU,DIS,NKE,ABT,CVX,TXN,AMGN,IBM,PM,UPS,LOW,NEE,RTX,MS,GS,SPGI,BLK,ISRG,QCOM,CAT,DE,SBUX,EL,PLD,MDT,AMAT,ADI,LMT,GE,AXP,CB,MMC,CI,MO,BDX,ZTS,ICE,DUK,SO,EQIX,SHW,ITW,PNC,APD,GM,FDX,HUM,AON,EMR,ECL,ETN,EW,ROP,TRV,VRTX,PSA,NSC,PH,SLB,MCO,ORLY,GILD,REGN,BIIB,MRNA,KLAC,LRCX,CDNS,SNPS,FTNT,ADP,PAYX,CTAS,ROST,ODFL,EXC,XEL,WM,CSX,KMB,COF,MSI,ILMN,DXCM,IDXX,FAST,KR,PPG,ALL,PRU,AIG,STZ,PEG,WEC,ED,ES,HPQ,DELL,HPE,ANET,NET,DDOG,SNOW,CRWD,ZS,PANW,TEAM,HUBS,SHOP,EBAY,ETSY,PINS,ROKU,SPOT,UBER,DASH,ABNB,BKNG,EXPE,MAR,HLT,RCL,CCL,NCLH,UAL,DAL,AAL,LUV,ALK,BA,GD,NOC,LHX,TDG,URI,ULTA,YUM,DPZ,CMG,MGM,LVS,WYNN,RSG,AWK,PCAR,CUMM,ALB,FCX,NEM,AA,STLD,NUE,CE,LYB,DOW,EMN,APTV,F,ROK,KEYS,AME,IR,XYL,TT,WAB,EFX,MTB,HBAN,RF,TFC,CFG,STT,NTRS,BK,AMP,IVZ,DFS,SYF,COIN,MSTR,RIOT,MARA,CHTR,CMCSA,VZ,T,TMUS,ZS,OKTA,WDAY,NOW,SQ,PYPL,INTC,TSM,ASML,MPWR,MCHP,SWKS,QRVO,ENTG,OLED,IPGP,SEDG,ENPH,FSLR,PLUG,BLDP,CHPT,QS,ON,STM,NXPI,INFY,CTSH,EPAM,DXC,ADSK,ANSS,PTC,SSNC,TYL,MANH,APPF,SMAR,PSTG,NTNX,VMW,LOGI,EA,TTWO,MTCH,BMBL,PLTR,AI,UPST,AFRM,SOFI,HOOD,CLS,ONTO,FORM,SMCI,GRMN,TRMB,IRDM,VSAT,ASTS,RKLB,JOBY,ACHR,PL,MAXR,SPCE"

FULL_STOCK_LIST = "AAPL,MSFT,GOOGL,GOOG,AMZN,NVDA,META,TSLA,BRK.B,BRK.A,JPM,V,JNJ,UNH,HD,PG,MA,XOM,LLY,AVGO,CVX,MRK,COST,ABBV,PEP,KO,ORCL,CRM,ADBE,AMD,NFLX,TMO,WMT,DIS,ACN,LIN,MCD,CSCO,ABT,TXN,DHR,INTU,CAT,VZ,CMCSA,NEE,BAC,WFC,PM,UPS,RTX,IBM,MS,PFE,INTC,LOW,UNP,AMGN,SPGI,GS,AXP,BLK,ISRG,QCOM,AMAT,DE,GE,NKE,MDT,ADI,LMT,SBUX,PLD,CB,CVS,MMC,TGT,MO,CI,BDX,ZTS,EL,DUK,SO,EQIX,ICE,SHW,CL,ITW,PNC,APD,GM,FDX,HUM,AON,EMR,ECL,ETN,EW,ROP,TRV,VRTX,PSA,NSC,PH,SLB,MCO,ORLY,GILD,REGN,BIIB,MRNA,KLAC,LRCX,CDNS,SNPS,FTNT,PAYX,ADP,CTAS,ROST,ODFL,EXC,XEL,WM,CSX,KMB,COF,MSI,ILMN,DXCM,IDXX,FAST,KR,PPG,ALL,PRU,AIG,STZ,PEG,WEC,ED,ES,HPQ,DELL,HPE,ANET,NET,DDOG,SNOW,CRWD,ZS,PANW,OKTA,TEAM,HUBS,SHOP,EBAY,ETSY,PINS,SNAP,ROKU,SPOT,UBER,LYFT,DASH,ABNB,BKNG,EXPE,MAR,HLT,RCL,CCL,NCLH,UAL,DAL,AAL,LUV,JBLU,ALK,BA,TXT,SPR,GD,NOC,HII,LHX,TDG,HEI,MTZ,PWR,URI,WSM,BBY,GPC,AZO,KMX,AN,SAH,PAG,TSCO,ULTA,YUM,DPZ,CMG,MGM,LVS,WYNN,PENN,CZR,RSG,AWK,PCAR,ALB,FCX,NEM,GOLD,AA,CLF,STLD,NUE,RS,X,MT,CE,LYB,DOW,EMN,APTV,F,LCID,RIVN,FSR,NIO,XPEV,LI,HMC,TM,POR,OGE,NRG,AES,ET,EPD,MPLX,PAA,KMI,WMB,OKE,ENB,TRP,EXAS,IQV,CTLT,CRL,INCY,ALNY,NBIX,UTHR,BMRN,SRPT,FOLD,PTCT,RARE,BLUE,EDIT,NTLA,CRSP,BEAM,AMED,SEM,HCA,UHS,CNC,MOH,ELV,MDGL,AXSM,GH,PACB,TXG,VCYT,NTRA,CDNA,MYGN,BTX,OMCL,CERS,TRMB,GRMN,IRDM,VSAT,ASTS,PLTR,SOUN,AI,BBAI,UPST,AFRM,SOFI,HOOD,COIN,MSTR,RIOT,MARA,CLSK,ONTO,FORM,ASML,TSM,UMC,SMCI,MPWR,MCHP,SWKS,QRVO,ENTG,OLED,IPGP,SEDG,ENPH,CSIQ,FSLR,ARRY,NEP,BEP,BEPC,PLUG,BLDP,FCEL,CHPT,EVGO,QS,FREY,ALGM,ON,STM,NXPI,INFY,WIT,CTSH,EPAM,DXC,TCS,SAP,ADSK,ANSS,PTC,SSNC,TYL,MANH,APPF,WK,BL,SMAR,PSTG,NTNX,VMW,LOGI,SONY,EA,TTWO,MTCH,BMBL,SPCE,RKLB,ASTR,MAXR,PL,JOBY,ACHR,EH"


def get_default_keyword_configs():
    """Get default keyword configurations."""
    return {
        "ALL": {
            "api_url": SPY_500_STOCKS,
            "description": "SPY 500 stocks (~500)"
        },
        "CAP": {
            "api_url": CAP_STOCK_LIST,
            "description": "Large Cap Focus (~280 stocks)"
        },
        "FULL": {
            "api_url": FULL_STOCK_LIST,
            "description": "Extended Sectors (~300 stocks)"
        },
        "SPY500": {
            "api_url": f"https://api.widesurf.com/v1/reference/tickers?market=stocks&active=true&order=desc&limit=500&sort=volume&type=limited&apiKey={WIDESURF_API_KEY}",
            "description": "Top 500 stocks by volume (Widesurf API)"
        },
        "NASDAQ500": {
            "api_url": f"https://api.widesurf.com/v1/reference/tickers?market=stocks&active=true&order=desc&limit=500&sort=volume&type=limited&apiKey={WIDESURF_API_KEY}",
            "description": "Top 500 NASDAQ stocks by volume (Widesurf API)"
        },
        "1000": {
            "api_url": f"https://api.widesurf.com/v1/reference/tickers?market=stocks&active=true&order=desc&limit=1000&sort=volume&type=limited&apiKey={WIDESURF_API_KEY}",
            "description": "Top 1000 stocks by volume (Widesurf API)"
        },
        "10000": {
            "api_url": f"http://10.0.0.94:8020/v1/reference/tickers?market=stocks&exchange=XNAS%2C+NYSE&active=true&order=desc&limit=10000&sort=transactions&type=limited&apiKey={WIDESURF_API_KEY}",
            "description": "Top 10,000 stocks by transactions (NASDAQ + NYSE)"
        }
    }


@app.get("/api/keywords")
async def get_keywords_api():
    """Get all keyword configurations."""
    try:
        # Get saved configs from database
        db_configs = get_keyword_configs()
        defaults = get_default_keyword_configs()
        
        # Merge with defaults (DB takes precedence)
        result = {}
        for keyword in ["ALL", "CAP", "FULL", "SPY500", "NASDAQ500", "1000", "10000"]:
            if keyword in db_configs and db_configs[keyword].get("api_url"):
                result[keyword] = db_configs[keyword]
            else:
                result[keyword] = defaults.get(keyword, {"api_url": "", "description": ""})
        
        return {"success": True, "keywords": result}
    except Exception as e:
        logger.error(f"Failed to get keywords: {e}")
        return {"success": False, "error": str(e)}


@app.put("/api/keywords")
async def save_keywords_api(request: dict):
    """Save keyword configurations."""
    try:
        configs = request.get("keywords", {})
        if not configs:
            return {"success": False, "error": "No keywords provided"}
        
        success = save_all_keyword_configs(configs)
        if success:
            return {"success": True, "message": f"Saved {len(configs)} keyword configurations"}
        else:
            return {"success": False, "error": "Failed to save to database"}
    except Exception as e:
        logger.error(f"Failed to save keywords: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

def mask_secret(secret: str) -> str:
    """Mask a secret key for security."""
    if not secret:
        return ""
    if len(secret) <= 8:
        return "********"
    return secret[:4] + "********" + secret[-4:]


@app.get("/api/keys")
async def get_api_keys_api():
    """Get all stored API keys with masked secrets, falling back to config.py defaults."""
    try:
        db_keys = get_api_keys()
        
        # Default keys from config.py
        defaults = {
            "ALPACA": {
                "key_id": config.ALPACA_API_KEY_ID,
                "secret_key": config.ALPACA_API_SECRET_KEY,
                "base_url": config.ALPACA_BASE_URL
            },
            "MASSIVE": {
                "key_id": "",
                "secret_key": config.MASSIVE_API_KEY,
                "base_url": config.MASSIVE_API_URL
            },
            "WIDESURF": {
                "key_id": "",
                "secret_key": config.WIDESURF_API_KEY,
                "base_url": config.WIDESURF_API_URL
            }
        }
        
        # Merge DB keys with defaults
        final_keys = {}
        providers = ["ALPACA", "MASSIVE", "WIDESURF"]
        
        for provider in providers:
            db_data = db_keys.get(provider, {})
            default_data = defaults.get(provider, {})
            
            final_keys[provider] = {
                "key_id": db_data.get("key_id") or default_data.get("key_id", ""),
                "secret_key": mask_secret(db_data.get("secret_key") or default_data.get("secret_key", "")),
                "base_url": db_data.get("base_url") or default_data.get("base_url", "")
            }
            
        return {"success": True, "keys": final_keys}
    except Exception as e:
        logger.error(f"Failed to get API keys: {e}")
        return {"success": False, "error": str(e)}


@app.put("/api/keys")
async def save_api_keys_api(request: dict):
    """Save API keys to the database."""
    try:
        keys_data = request.get("keys", {})
        if not keys_data:
            return {"success": False, "error": "No keys provided"}
            
        success_count = 0
        for provider, data in keys_data.items():
            key_id = data.get("key_id")
            secret_key = data.get("secret_key")
            base_url = data.get("base_url")
            
            # If the secret_key is masked, don't update it in the DB
            if secret_key and "********" in secret_key:
                secret_key = None
                
            if save_api_key(provider.upper(), key_id, secret_key, base_url):
                success_count += 1
                
        return {"success": True, "message": f"Updated {success_count} API providers"}
    except Exception as e:
        logger.error(f"Failed to save API keys: {e}")
        return {"success": False, "error": str(e)}


# Store for active WebSocket connections
active_connections: List[WebSocket] = []

# Store optimization results
optimization_results: Dict[str, dict] = {}
optimization_status: Dict[str, dict] = {}


class OptimizationRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    backtest_start_date: str = None  # Optional separate backtest range
    backtest_end_date: str = None    # Optional separate backtest range
    capital: float = 100000
    n_trials: int = 200
    optimization_metric: str = "sharpe"
    use_gpu: bool = False  # Enable GPU-accelerated grid search
    smart_timing: bool = False  # Enable optimal buy time optimization
    timing_approach: str = "sequential"  # "sequential" or "joint"
    algo: str = "default"  # "default" or "chatgpt"
    data_source: str = "alpaca"  # "alpaca" or "widesurf"


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


def generate_trade_log(bars: list, buy_trigger_pct: float, sell_trigger_pct: float, compound: bool, capital: float = 100000, algo: str = "default", symbol: str = None, start_date: str = None, end_date: str = None, stop_loss_pct: float = None, trailing_stop_pct: float = None) -> list:
    # #region agent log
    import json as _json
    with open("/app/debug.log", "a") as _f:
        _f.write(_json.dumps({"hypothesisId": "H0", "location": "server.py:764", "message": "generate_trade_log ENTRY", "data": {"symbol": symbol, "algo": algo, "num_bars": len(bars) if bars else 0, "start": start_date, "end": end_date}, "timestamp": int(__import__("time").time()*1000)}) + "\n")
    # #endregion
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
    
    
    # For ChatGPT algos, try to fetch 10AM prices (using cache)
    price_9am_map = {}
    if algo in ("chatgpt", "chatgpt_stoploss", "chatgpt_vwap", "chatgpt_vwap_rust") and symbol and start_date and end_date:
        try:
            logger.info(f"Fetching 9AM prices for {symbol} (ChatGPT algo) using cache")
            
            # Use cached intraday data
            intraday_bars = fetch_and_cache_prices(symbol, start_date, end_date, "1Min")
            
            # #region agent log
            import json as _json
            with open("/app/debug.log", "a") as _f:
                _f.write(_json.dumps({"hypothesisId": "H1", "location": "server.py:776", "message": "fetch_and_cache_prices returned", "data": {"symbol": symbol, "start": start_date, "end": end_date, "num_bars": len(intraday_bars) if intraday_bars else 0, "sample_bars": [str(b.get("t", ""))[:25] for b in (intraday_bars or [])[:5]]}, "timestamp": int(__import__("time").time()*1000)}) + "\n")
            # #endregion
            
            if intraday_bars:
                # Group bars by date first
                bars_by_date = {}
                for bar in intraday_bars:
                    bar_time = bar.get("t", "")
                    date_part = bar_time[:10] if bar_time else ""
                    if date_part:
                        if date_part not in bars_by_date:
                            bars_by_date[date_part] = []
                        bars_by_date[date_part].append(bar)
                
                # For each date, find the exact 10AM bar and calculate VWAP
                # Uses DST-aware timezone conversion
                for date_part, day_bars in bars_by_date.items():
                    best_10am_bar = None
                    best_10am_diff = 9999
                    vwap_bars = []
                    or_highs = []  # Opening range highs (9:30-10:00)
                    or_lows = []   # Opening range lows (9:30-10:00)
                    vwap_945_bars = []  # Bars for VWAP at 9:40
                    
                    for bar in day_bars:
                        bar_time = bar.get("t", "")
                        et_minutes = get_et_minutes_from_timestamp(bar_time)
                        
                        if et_minutes is None:
                            continue
                        
                        # Collect bars from 9:30 to 10:00 for VWAP and OR
                        if 570 <= et_minutes <= 600:
                            vwap_bars.append(bar)
                            or_highs.append(bar.get("h", 0))
                            or_lows.append(bar.get("l", 0))
                        
                        # Collect bars from 9:30 to 9:40 for VWAP_9:40 (robust slope)
                        if 570 <= et_minutes <= 580:
                            vwap_945_bars.append(bar)
                        
                        # Find closest bar to 10:00 AM EST (600 minutes)
                        # Only consider bars within market hours (9:30 AM - 4:00 PM = 570-960 minutes)
                        if 570 <= et_minutes <= 960:
                            diff = abs(et_minutes - 600)
                            if diff < best_10am_diff:
                                best_10am_bar = bar
                                best_10am_diff = diff
                    
                    # Get 10AM price - verify the selected bar is reasonably close to 10:00 AM
                    # (within 5 minutes to avoid using wrong bar)
                    if best_10am_bar and best_10am_diff <= 5:
                        price_10am = best_10am_bar.get("c", best_10am_bar.get("o", 0))
                    else:
                        # No bar close enough to 10:00 AM found
                        price_10am = 0
                        if best_10am_bar:
                            logger.warning(f"Best 10AM bar for {date_part} is {best_10am_diff} minutes away from 10:00 AM")
                    
                    # Calculate VWAP at 10:00 (9:30 - 10:00)
                    cum_tpv = 0
                    cum_vol = 0
                    for bar in vwap_bars:
                        tp = (bar.get("h", 0) + bar.get("l", 0) + bar.get("c", 0)) / 3
                        vol = bar.get("v", 0)
                        cum_tpv += tp * vol
                        cum_vol += vol
                    
                    vwap_to_10am = cum_tpv / cum_vol if cum_vol > 0 else price_10am
                    
                    # Calculate VWAP at 9:40 (9:30 - 9:40) for robust slope
                    cum_tpv_940 = 0
                    cum_vol_940 = 0
                    for bar in vwap_945_bars:
                        tp = (bar.get("h", 0) + bar.get("l", 0) + bar.get("c", 0)) / 3
                        vol = bar.get("v", 0)
                        cum_tpv_940 += tp * vol
                        cum_vol_940 += vol
                    
                    vwap_940 = cum_tpv_940 / cum_vol_940 if cum_vol_940 > 0 else vwap_to_10am
                    
                    # Opening range high/low
                    or_high = max(or_highs) if or_highs else price_10am
                    or_low = min(or_lows) if or_lows else price_10am
                    
                    price_9am_map[date_part] = {
                        "price_10am": price_10am,
                        "vwap": vwap_to_10am,
                        "vwap_940": vwap_940,
                        "or_high": or_high,
                        "or_low": or_low
                    }
                
                # #region agent log
                with open("/app/debug.log", "a") as _f:
                    _f.write(_json.dumps({"hypothesisId": "H2", "location": "server.py:802", "message": "price_10am_map built", "data": {"num_dates": len(price_9am_map), "sample_dates": list(price_9am_map.keys())[:5], "sample_prices": list(price_9am_map.values())[:5]}, "timestamp": int(__import__("time").time()*1000)}) + "\n")
                # #endregion
                
                logger.info(f"Found 10AM prices for {len(price_9am_map)} days from cache")
            else:
                logger.warning(f"No intraday data available for {symbol}")
        except Exception as e:
            logger.warning(f"Could not fetch 9AM prices: {e}")
    
    for i, bar in enumerate(bars):  # Process all days in the period
        open_price = bar.get("o", bar.get("c", 0))
        close_price = bar.get("c", 0)
        high_price = bar.get("h", close_price)
        low_price = bar.get("l", close_price)
        bar_date = bar.get("t", "")
        date_key = bar_date[:10] if bar_date else ""
        
        if open_price <= 0:
            continue
        
        # =====================================================================
        # ChatGPT 9AM Strategy (matches GPUChatGPT9AMBacktester)
        # =====================================================================
        if algo == "chatgpt":
            # Get 10 AM price from intraday cache (handles both old and new format)
            price_data = price_9am_map.get(date_key, {})
            if isinstance(price_data, dict):
                price_10am = price_data.get("price_10am", 0)
            else:
                price_10am = price_data  # Old format: single value
            
            if price_10am <= 0:
                # No 10 AM price available, skip or use open as fallback
                price_10am = open_price
            
            # Buy threshold: open * (1 + buy_trigger)
            buy_threshold = open_price * (1 + buy_trigger_pct / 100)
            
            # Buy signal: 10 AM price >= buy_threshold
            bought = price_10am >= buy_threshold
            buy_price = price_10am if bought else None
            
            # Sell threshold: open * (1 + sell_trigger) - based on OPEN, not buy price
            sell_threshold = open_price * (1 + sell_trigger_pct / 100)
            
            # Did we hit sell target?
            sold_at_target = bought and high_price >= sell_threshold
            
            # Exit price: sell_threshold if target hit, else close
            actual_sell_price = sell_threshold if sold_at_target else close_price if bought else None
            
            shares = 0
            profit = 0
            pct_change_10am = ((price_10am - open_price) / open_price * 100) if open_price > 0 else 0
            
            if bought and buy_price:
                shares = int(equity / buy_price) if buy_price > 0 else 0
                
                if shares > 0:
                    profit = shares * (actual_sell_price - buy_price)
                    if compound:
                        equity += profit
            
            trade_log.append({
                "date": bar_date,
                "open": round(open_price, 2),
                "price_10am": round(price_10am, 2),
                "pct_change_10am": round(pct_change_10am, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "bought": bought,
                "sold_at_target": sold_at_target,
                "buy_price": round(buy_price, 2) if buy_price else None,
                "sell_price": round(sell_threshold, 2),
                "actual_sell": round(actual_sell_price, 2) if actual_sell_price else None,
                "shares": shares,
                "profit": round(profit, 0),
                "equity": round(equity, 0),
                "algo": "chatgpt"
            })
        elif algo == "chatgpt_stoploss":
            # =====================================================================
            # ChatGPT 9AM with Stop Loss (OPTIMIZED parameters)
            # Entry: 10AM if +X% from open (optimized entry_trigger)
            # Take Profit: +Y% from entry (optimized sell_trigger_pct)
            # Stop Loss: -Z% from entry (optimized stop_loss_pct)
            # =====================================================================
            price_data = price_9am_map.get(date_key, {})
            if isinstance(price_data, dict):
                price_10am = price_data.get("price_10am", 0)
            else:
                price_10am = price_data  # Old format
            if price_10am <= 0:
                price_10am = open_price
            
            # Use OPTIMIZED entry trigger from buy_trigger_pct
            entry_trigger = buy_trigger_pct / 100  # Convert from % to decimal
            entry_threshold = open_price * (1 + entry_trigger)
            
            # Entry signal: 10AM price >= entry threshold
            bought = price_10am >= entry_threshold
            buy_price = price_10am if bought else None
            
            # Use OPTIMIZED take profit and stop loss from optimization result
            take_profit_decimal = sell_trigger_pct / 100  # Convert from % to decimal
            # Use passed stop_loss_pct if available, else use 1.5x entry trigger as fallback
            sl_pct = stop_loss_pct if stop_loss_pct else buy_trigger_pct * 1.5
            stop_loss_decimal = -(sl_pct / 100)  # Convert to negative decimal
            
            take_profit_price = buy_price * (1 + take_profit_decimal) if buy_price else 0
            stop_loss_price = buy_price * (1 + stop_loss_decimal) if buy_price else 0
            
            shares = 0
            profit = 0
            actual_sell_price = None
            exit_reason = None
            pct_change_10am = ((price_10am - open_price) / open_price * 100) if open_price > 0 else 0
            
            if bought and buy_price:
                shares = int(equity / buy_price) if buy_price > 0 else 0
                
                # Calculate trailing stop price if enabled
                trailing_decimal = -(trailing_stop_pct / 100) if trailing_stop_pct else 0
                trailing_stop_price = high_price * (1 + trailing_decimal) if trailing_decimal < 0 else 0
                
                if shares > 0:
                    # Determine exit (TP > Trailing (if profitable) > SL > Time)
                    if high_price >= take_profit_price:
                        # Take profit hit
                        actual_sell_price = take_profit_price
                        exit_reason = "TP"
                    elif trailing_decimal < 0 and low_price <= trailing_stop_price and trailing_stop_price > buy_price:
                        # Trailing stop hit (only if in profit)
                        actual_sell_price = trailing_stop_price
                        exit_reason = "TRAIL"
                    elif low_price <= stop_loss_price:
                        # Stop loss hit
                        actual_sell_price = stop_loss_price
                        exit_reason = "SL"
                    else:
                        # Time exit at close
                        actual_sell_price = close_price
                        exit_reason = "TIME"
                    
                    profit = shares * (actual_sell_price - buy_price)
                    if compound:
                        equity += profit
            
            trade_log.append({
                "date": bar_date,
                "open": round(open_price, 2),
                "price_10am": round(price_10am, 2),
                "pct_change_10am": round(pct_change_10am, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "bought": bought,
                "sold_at_target": exit_reason == "TP",
                "exit_reason": exit_reason,
                "buy_price": round(buy_price, 2) if buy_price else None,
                "sell_price": round(take_profit_price, 2) if take_profit_price else None,
                "stop_loss": round(stop_loss_price, 2) if stop_loss_price else None,
                "actual_sell": round(actual_sell_price, 2) if actual_sell_price else None,
                "shares": shares,
                "profit": round(profit, 0),
                "equity": round(equity, 0),
                "algo": "chatgpt_stoploss"
            })
        elif algo in ("chatgpt_vwap", "chatgpt_vwap_rust"):
            # =====================================================================
            # ChatGPT Adaptive VWAP Strategy
            # Entry: price > VWAP AND vwap_slope > β*or_vol AND vwap_stretch < α*or_vol
            # =====================================================================
            price_data = price_9am_map.get(date_key, {})
            if isinstance(price_data, dict):
                price_10am = price_data.get("price_10am", 0)
                vwap = price_data.get("vwap", bar.get("c", 0))
                vwap_940 = price_data.get("vwap_940", vwap)
                or_high = price_data.get("or_high", 0)
                or_low = price_data.get("or_low", 0)
            else:
                price_10am = price_data if price_data else bar.get("o", 0)
                vwap = bar.get("c", 0)
                vwap_940 = vwap
                or_high = bar.get("h", 0)
                or_low = bar.get("l", 0)
            if price_10am <= 0:
                price_10am = open_price
            if vwap <= 0:
                vwap = bar.get("c", 0)
            high_price = bar.get("h", 0)
            low_price = bar.get("l", 0)
            close_price = bar.get("c", 0)
            
            # Entry parameters from optimization (TP/SL only)
            take_profit_decimal = sell_trigger_pct / 100
            stop_loss_decimal = -(stop_loss_pct / 100) if stop_loss_pct else -0.007
            
            # Simplified VWAP entry conditions (NO % threshold)
            # Alpha = stretch threshold, Gamma = momentum threshold
            alpha = 0.50  # vwap_stretch < 0.50 * or_vol
            gamma = 0.25  # momentum_score > 0.25 * or_vol
            
            # Calculate adaptive metrics
            # Step 1: VWAP stretch = (price_10am - vwap) / vwap
            vwap_stretch = (price_10am - vwap) / vwap if vwap > 0 else 0
            
            # Step 2: VWAP-anchored opening range volatility
            or_vol = max(or_high - vwap, vwap - or_low) / vwap if vwap > 0 else 0.02
            
            # Step 3: Robust VWAP slope = (vwap_10:00 - vwap_9:40) / vwap_9:40
            vwap_slope = (vwap - vwap_940) / vwap_940 if vwap_940 > 0 else 0
            
            # Step 4: Momentum score = (price_10am - open) / open
            momentum_score = (price_10am - open_price) / open_price if open_price > 0 else 0
            
            # Determine skip reason if trade not taken
            skip_reason = None
            
            # Simplified VWAP conditions (NO % threshold):
            # 1. price_10am > vwap (price above VWAP)
            price_above_vwap = price_10am > vwap and vwap > 0
            
            # 2. vwap_slope > 0 (VWAP trending up)
            slope_ok = vwap_slope > 0
            
            # 3. vwap_stretch < 0.50 * or_vol (not too extended)
            stretch_ok = vwap_stretch < alpha * or_vol if or_vol > 0 else vwap_stretch < 0.01
            
            # 4. momentum_score > 0.25 * or_vol (sufficient momentum)
            momentum_ok = momentum_score > gamma * or_vol if or_vol > 0 else momentum_score > 0.005
            
            vwap_ok = price_above_vwap and slope_ok and stretch_ok and momentum_ok
            
            # Skip reason (in priority order)
            if not price_above_vwap:
                if vwap <= 0:
                    skip_reason = "No VWAP"
                else:
                    skip_reason = "$ < VWAP"
            elif not slope_ok:
                skip_reason = f"slope↓ ({vwap_slope*100:.2f}%)"
            elif not stretch_ok:
                skip_reason = f"stretch↑ ({vwap_stretch*100:.2f}%)"
            elif not momentum_ok:
                skip_reason = f"mom↓ ({momentum_score*100:.2f}%)"
            
            # Entry: ALL VWAP conditions passed (no % threshold)
            bought = vwap_ok
            
            buy_price = price_10am if bought else None
            take_profit_price = price_10am * (1 + take_profit_decimal) if bought else None
            stop_loss_price = open_price if bought else None  # SL = Open price
            actual_sell_price = None
            exit_reason = None
            pct_change_10am = ((price_10am - open_price) / open_price * 100) if open_price > 0 else 0
            shares = 0  # Initialize to prevent UnboundLocalError
            profit = 0  # Initialize to prevent UnboundLocalError
            
            if bought and buy_price:
                shares = int(equity / buy_price) if buy_price > 0 else 0
                
                if shares > 0:
                    # Mean reversion: TP or CLOSE only (no SL - let it ride)
                    if high_price >= take_profit_price:
                        actual_sell_price = take_profit_price
                        exit_reason = "TP"
                    else:
                        # Exit at close - capture mean reversion recovery profits
                        actual_sell_price = close_price
                        exit_reason = "CLOSE"
                    
                    profit = shares * (actual_sell_price - buy_price)
                    if compound:
                        equity += profit
            
            trade_log.append({
                "date": bar_date,
                "open": round(open_price, 2),
                "price_10am": round(price_10am, 2),
                "vwap": round(vwap, 2),
                "pct_change_10am": round(pct_change_10am, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "bought": bought,
                "sold_at_target": exit_reason == "TP",
                "exit_reason": exit_reason,
                "skip_reason": skip_reason,  # Why trade was not taken
                "buy_price": round(buy_price, 2) if buy_price else None,
                "sell_price": round(take_profit_price, 2) if take_profit_price else None,
                "stop_loss": round(stop_loss_price, 2) if stop_loss_price else None,
                "actual_sell": round(actual_sell_price, 2) if actual_sell_price else None,
                "shares": shares,
                "profit": round(profit, 0),
                "equity": round(equity, 0),
                "algo": "chatgpt_vwap"
            })
        else:
            # =====================================================================
            # Dipper Strategy: Breakout (matches GPUBatchBacktester)
            # =====================================================================
            # Get previous close for reference
            prev_close = bars[i-1].get("c", 0) if i > 0 else open_price
            
            if prev_close <= 0:
                continue
            
            # Buy signal: high >= prev_close * (1 + buy_trigger)
            buy_price_target = prev_close * (1 + buy_trigger_pct / 100)
            bought = high_price >= buy_price_target
            
            # Calculate pct change from prev_close to high
            pct_change = ((high_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
            
            # Sell target from buy price
            sell_price_target = buy_price_target * (1 + sell_trigger_pct / 100)
            
            # Did we hit sell target?
            sold_at_target = bought and high_price >= sell_price_target
            
            shares = 0
            profit = 0
            actual_sell_price = None
            
            if bought:
                buy_price = buy_price_target
                shares = int(equity / buy_price) if buy_price > 0 else 0
                
                if shares > 0:
                    if sold_at_target:
                        actual_sell_price = sell_price_target
                    else:
                        actual_sell_price = close_price
                    
                    profit = shares * (actual_sell_price - buy_price)
                    
                    if compound:
                        equity += profit
            
            trade_log.append({
                "date": bar_date,
                "prev_close": round(prev_close, 2),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "pct_change": round(pct_change, 2),
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
            "compound": True,  # Always compound
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
    
    # Check cache range first to determine if we need to download missing data
    from datetime import datetime, timedelta
    cached_min, cached_max = get_cached_price_range(symbol.upper(), "1Day")
    start_dt = datetime.strptime(config.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(config.end_date, "%Y-%m-%d")
    
    need_download = False
    download_msg = ""
    if cached_min is None:
        need_download = True
        download_msg = f"Downloading price data for {symbol}..."
    else:
        if start_dt.date() < cached_min:
            need_download = True
            download_msg = f"Downloading data for {symbol} ({config.start_date} to {cached_min})..."
        if end_dt.date() > cached_max:
            need_download = True
            if download_msg:
                download_msg = f"Downloading missing data for {symbol}..."
            else:
                download_msg = f"Downloading data for {symbol} ({cached_max} to {config.end_date})..."
    
    # Get data source for status messages
    data_source = getattr(config, 'data_source', 'alpaca')
    data_source_label = "Widesurf API" if data_source == "widesurf" else "Alpaca API"
    
    if need_download or data_source == "widesurf":
        await broadcast_message({
            "type": "status",
            "job_id": job_id,
            "symbol": symbol,
            "status": "downloading_data",
            "message": f"Fetching {symbol} from {data_source_label}..." if data_source == "widesurf" else download_msg
        })
    else:
        await broadcast_message({
            "type": "status",
            "job_id": job_id,
            "symbol": symbol,
            "status": "fetching_data",
            "message": f"Using cached data for {symbol} ({cached_min} to {cached_max}) [{data_source_label}]..."
        })
    
    # Fetch price data with caching (downloads missing ranges automatically)
    try:
        # Use data source from config (default: alpaca)
        data_source = getattr(config, 'data_source', 'alpaca')
        
        if data_source == "widesurf":
            # Use Widesurf API directly
            bars = fetch_prices_from_widesurf(symbol, config.start_date, config.end_date, "1Day")
            if not bars:
                logger.warning(f"Widesurf returned no data for {symbol}, trying cache")
                bars = fetch_and_cache_prices(symbol, config.start_date, config.end_date, "1Day")
        else:
            # Use the cached system for Alpaca API
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
    # Note: With Stocks Developer plan ($79/mo), we have 10 years of historical data
    if actual_start_date and actual_start_date > config.start_date:
        data_range_warning = f"⚠️ Data only available from {actual_start_date}. Requested start {config.start_date} not available (stock may not have been trading then)."
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

    # GPU / Parallel Grid Search Path
    logger.info(f"Optimization Path Decision: use_gpu={config.use_gpu}, algo={getattr(config, 'algo', 'default')}")
    
    # We enter this block if GPU is requested OR if the Rust algo is selected (which doesn't need GPU but needs Grid Search)
    if (config.use_gpu and GPU_BACKTEST_AVAILABLE) or getattr(config, 'algo', '') == 'chatgpt_vwap_rust':
        await broadcast_message({
            "type": "status",
            "job_id": job_id,
            "symbol": symbol,
            "status": "optimizing_parallel",
            "message": f"Running parallelized massive grid search...",
        })
        
        try:
            algo_type = getattr(config, 'algo', 'default') or 'default'
            
            # ChatGPT 9AM Strategy uses different backtester
            if algo_type == "chatgpt":
                from gpu_backtest import run_chatgpt_9am_optimization
                
                await broadcast_message({
                    "type": "status",
                    "job_id": job_id,
                    "symbol": symbol,
                    "status": "fetching_10am_prices",
                    "message": f"Fetching 10 AM prices for ChatGPT 9AM Strategy..."
                })
                
                # Fetch 10 AM prices from intraday cache
                intraday_bars = fetch_and_cache_prices(symbol, config.start_date, config.end_date, "1Min")
                
                # Build map of date -> 10 AM price (find exact 10AM bar)
                price_10am_map = {}
                bars_by_date = {}
                for bar in intraday_bars or []:
                    bar_time = bar.get("t", "")
                    date_part = bar_time[:10] if bar_time else ""
                    if date_part:
                        if date_part not in bars_by_date:
                            bars_by_date[date_part] = []
                        bars_by_date[date_part].append(bar)
                
                for date_part, day_bars in bars_by_date.items():
                    best_10am_bar = None
                    best_10am_diff = 9999
                    for bar in day_bars:
                        bar_time = bar.get("t", "")
                        et_minutes = get_et_minutes_from_timestamp(bar_time)
                        
                        if et_minutes is None:
                            continue
                        
                        # Find closest bar to 10:00 AM EST (600 minutes)
                        # Only consider bars within market hours (9:30 AM - 4:00 PM)
                        if 570 <= et_minutes <= 960:
                            diff = abs(et_minutes - 600)
                            if diff < best_10am_diff:
                                best_10am_bar = bar
                                best_10am_diff = diff
                    
                    # Only use the bar if it's within 5 minutes of 10:00 AM
                    if best_10am_bar and best_10am_diff <= 5:
                        price_10am_map[date_part] = best_10am_bar.get("c", best_10am_bar.get("o", 0))
                
                logger.info(f"Found 10AM prices for {len(price_10am_map)} days for ChatGPT strategy")
                
                # Add 10 AM prices to bars
                bars_with_10am = []
                for bar in bars:
                    bar_date = bar.get("t", "")[:10] if bar.get("t") else ""
                    new_bar = dict(bar)
                    new_bar["price_10am"] = price_10am_map.get(bar_date, bar.get("o", 0))
                    bars_with_10am.append(new_bar)
                
                # ChatGPT uses different parameter ranges (smaller buy triggers since buying on up move)
                buy_range = (0.001, 0.03, 0.001)  # 0.1% to 3%
                sell_range = (0.01, 0.10, 0.002)   # 1% to 10%
                
                import numpy as np
                total_combinations = len(np.arange(buy_range[0], buy_range[1], buy_range[2])) * \
                                    len(np.arange(sell_range[0], sell_range[1], sell_range[2]))
                
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
                
                loop = asyncio.get_event_loop()
                gpu_result = await loop.run_in_executor(
                    None,
                    run_chatgpt_9am_optimization,
                    bars_with_10am,
                    config.capital,
                    buy_range,
                    sell_range,
                    config.optimization_metric
                )
            elif algo_type == "chatgpt_stoploss":
                # ChatGPT 9AM with Stop Loss - Fixed parameters
                from gpu_backtest import run_chatgpt_stoploss_optimization
                
                await broadcast_message({
                    "type": "status",
                    "job_id": job_id,
                    "symbol": symbol,
                    "status": "fetching_10am_prices",
                    "message": f"Fetching 10 AM prices for ChatGPT Stop Loss Strategy..."
                })
                
                # Fetch 10 AM prices from intraday cache
                intraday_bars = fetch_and_cache_prices(symbol, config.start_date, config.end_date, "1Min")
                
                # Build map of date -> 10 AM price (find exact 10AM bar)
                price_10am_map = {}
                bars_by_date = {}
                for bar in intraday_bars or []:
                    bar_time = bar.get("t", "")
                    date_part = bar_time[:10] if bar_time else ""
                    if date_part:
                        if date_part not in bars_by_date:
                            bars_by_date[date_part] = []
                        bars_by_date[date_part].append(bar)
                
                for date_part, day_bars in bars_by_date.items():
                    best_10am_bar = None
                    best_10am_diff = 9999
                    for bar in day_bars:
                        bar_time = bar.get("t", "")
                        et_minutes = get_et_minutes_from_timestamp(bar_time)
                        
                        if et_minutes is None:
                            continue
                        
                        # Find closest bar to 10:00 AM EST (600 minutes)
                        # Only consider bars within market hours (9:30 AM - 4:00 PM)
                        if 570 <= et_minutes <= 960:
                            diff = abs(et_minutes - 600)
                            if diff < best_10am_diff:
                                best_10am_bar = bar
                                best_10am_diff = diff
                    
                    # Only use the bar if it's within 5 minutes of 10:00 AM
                    if best_10am_bar and best_10am_diff <= 5:
                        price_10am_map[date_part] = best_10am_bar.get("c", best_10am_bar.get("o", 0))
                
                logger.info(f"Found 10AM prices for {len(price_10am_map)} days for ChatGPT Stop Loss")
                
                # Add 10 AM prices to bars
                bars_with_10am = []
                for bar in bars:
                    bar_date = bar.get("t", "")[:10] if bar.get("t") else ""
                    new_bar = dict(bar)
                    new_bar["price_10am"] = price_10am_map.get(bar_date, bar.get("o", 0))
                    bars_with_10am.append(new_bar)
                
                # Grid search ranges for ChatGPT Stop Loss
                entry_range = (0.003, 0.020, 0.003)   # Entry: 0.3% to 2% above open
                tp_range = (0.008, 0.035, 0.004)      # Take Profit: 0.8% to 3.5%
                sl_range = (-0.035, -0.008, 0.004)    # Stop Loss: -3.5% to -0.8%
                trailing_range = (-0.025, -0.005, 0.004)  # Trailing Stop: -2.5% to -0.5%
                
                import numpy as np
                n_entry = len(np.arange(entry_range[0], entry_range[1], entry_range[2]))
                n_tp = len(np.arange(tp_range[0], tp_range[1], tp_range[2]))
                n_sl = len(np.arange(sl_range[0], sl_range[1], sl_range[2]))
                n_trail = len(np.arange(trailing_range[0], trailing_range[1], trailing_range[2])) + 1  # +1 for disabled
                n_filter = 2  # Trend filter on/off
                total_combinations = n_entry * n_tp * n_sl * n_trail * n_filter
                
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
                
                loop = asyncio.get_event_loop()
                gpu_result = await loop.run_in_executor(
                    None,
                    run_chatgpt_stoploss_optimization,
                    bars_with_10am,
                    config.capital,
                    entry_range,
                    tp_range,
                    sl_range,
                    trailing_range,
                    config.optimization_metric
                )
            elif algo_type == "chatgpt_vwap":
                # ChatGPT with VWAP - Uses SAME optimization as ChatGPT 9AM
                # VWAP filter is applied only during backtesting/trade log generation
                from gpu_backtest import run_chatgpt_9am_optimization
                
                await broadcast_message({
                    "type": "status",
                    "job_id": job_id,
                    "symbol": symbol,
                    "status": "fetching_vwap_data",
                    "message": f"Fetching 10 AM prices and VWAP data..."
                })
                
                # Fetch intraday data for 10AM prices and VWAP
                intraday_bars = fetch_and_cache_prices(symbol, config.start_date, config.end_date, "1Min")
                
                # Build map of date -> {price_10am, vwap} from intraday bars
                vwap_data_map = {}
                # Group bars by date first
                bars_by_date = {}
                for bar in intraday_bars or []:
                    bar_time = bar.get("t", "")
                    date_part = bar_time[:10] if bar_time else ""
                    if date_part:
                        if date_part not in bars_by_date:
                            bars_by_date[date_part] = []
                        bars_by_date[date_part].append(bar)
                
                # For each date, find the 10AM bar and calculate VWAP from 9:30 to 10AM
                # Uses DST-aware timezone conversion
                for date_part, day_bars in bars_by_date.items():
                    price_10am = None
                    best_10am_bar = None
                    best_10am_diff = 9999
                    vwap_bars = []
                    
                    for bar in day_bars:
                        bar_time = bar.get("t", "")
                        et_minutes = get_et_minutes_from_timestamp(bar_time)
                        
                        if et_minutes is None:
                            continue
                        
                        # 9:30 AM EST = 570 minutes, 10:00 AM EST = 600 minutes
                        # Collect bars from 9:30 to 10:00 for VWAP calculation
                        if 570 <= et_minutes <= 600:
                            vwap_bars.append(bar)
                        
                        # Find exact 10:00 AM EST bar (600 minutes or closest)
                        # Only consider bars within market hours (9:30 AM - 4:00 PM)
                        if 570 <= et_minutes <= 960:
                            diff = abs(et_minutes - 600)
                            if diff < best_10am_diff:
                                best_10am_bar = bar
                                best_10am_diff = diff
                    
                    # Get 10AM price from closest bar (must be within 5 minutes)
                    if best_10am_bar and best_10am_diff <= 5:
                        price_10am = best_10am_bar.get("c", best_10am_bar.get("o", 0))
                    
                    # Calculate VWAP from 9:30 to 10AM
                    cum_tpv = 0
                    cum_vol = 0
                    for bar in vwap_bars:
                        tp = (bar.get("h", 0) + bar.get("l", 0) + bar.get("c", 0)) / 3
                        vol = bar.get("v", 0)
                        cum_tpv += tp * vol
                        cum_vol += vol
                    
                    vwap_to_10am = cum_tpv / cum_vol if cum_vol > 0 else (price_10am or 0)
                    
                    vwap_data_map[date_part] = {
                        "price_10am": price_10am or 0,
                        "vwap": vwap_to_10am
                    }
                
                logger.info(f"Found VWAP data for {len(vwap_data_map)} days for ChatGPT VWAP")
                
                # Add 10AM prices (and VWAP for trade log) to bars
                bars_with_10am = []
                for bar in bars:
                    bar_date = bar.get("t", "")[:10] if bar.get("t") else ""
                    new_bar = dict(bar)
                    vwap_info = vwap_data_map.get(bar_date, {})
                    new_bar["price_10am"] = vwap_info.get("price_10am", bar.get("o", 0))
                    new_bar["vwap"] = vwap_info.get("vwap", bar.get("c", 0))  # Store for trade log
                    bars_with_10am.append(new_bar)
                
                # Use SAME parameter ranges and optimization as ChatGPT 9AM
                buy_range = (0.001, 0.03, 0.001)  # 0.1% to 3%
                sell_range = (0.01, 0.10, 0.002)   # 1% to 10%
                
                import numpy as np
                total_combinations = len(np.arange(buy_range[0], buy_range[1], buy_range[2])) * \
                                    len(np.arange(sell_range[0], sell_range[1], sell_range[2]))
                
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
                
                # Use SAME optimization as ChatGPT 9AM (no VWAP filter during optimization)
                loop = asyncio.get_event_loop()
                gpu_result = await loop.run_in_executor(
                    None,
                    run_chatgpt_9am_optimization,
                    bars_with_10am,
                    config.capital,
                    buy_range,
                    sell_range,
                    config.optimization_metric
                )
            elif algo_type == "chatgpt_vwap_rust":
                # =====================================================================
                # RUST Rayon-parallel VWAP Momentum 10am optimizer
                # Same strategy as chatgpt_vwap but uses Rust + Rayon for parallelism
                # across all CPU cores (32 threads on i9-13900K)
                # =====================================================================
                from rust_optimizer import run_rust_vwap_optimization
                
                await broadcast_message({
                    "type": "status",
                    "job_id": job_id,
                    "symbol": symbol,
                    "status": "fetching_vwap_data",
                    "message": f"Fetching 10 AM prices and VWAP data for Rust optimizer..."
                })
                
                # Fetch intraday data for 10AM prices and VWAP
                intraday_bars = fetch_and_cache_prices(symbol, config.start_date, config.end_date, "1Min")
                
                # Build map of date -> {price_10am, vwap, vwap_940, or_high, or_low}
                vwap_data_map = {}
                bars_by_date = {}
                for bar in intraday_bars or []:
                    bar_time = bar.get("t", "")
                    date_part = bar_time[:10] if bar_time else ""
                    if date_part:
                        if date_part not in bars_by_date:
                            bars_by_date[date_part] = []
                        bars_by_date[date_part].append(bar)
                
                for date_part, day_bars in bars_by_date.items():
                    best_10am_bar = None
                    best_10am_diff = 9999
                    vwap_bars_list = []
                    or_highs = []
                    or_lows = []
                    vwap_940_bars = []
                    
                    for bar in day_bars:
                        bar_time = bar.get("t", "")
                        et_minutes = get_et_minutes_from_timestamp(bar_time)
                        if et_minutes is None:
                            continue
                        if 570 <= et_minutes <= 600:
                            vwap_bars_list.append(bar)
                            or_highs.append(bar.get("h", 0))
                            or_lows.append(bar.get("l", 0))
                        if 570 <= et_minutes <= 580:
                            vwap_940_bars.append(bar)
                        if 570 <= et_minutes <= 960:
                            diff = abs(et_minutes - 600)
                            if diff < best_10am_diff:
                                best_10am_bar = bar
                                best_10am_diff = diff
                    
                    price_10am = 0
                    if best_10am_bar and best_10am_diff <= 5:
                        price_10am = best_10am_bar.get("c", best_10am_bar.get("o", 0))
                    
                    cum_tpv, cum_vol = 0, 0
                    for bar in vwap_bars_list:
                        tp = (bar.get("h", 0) + bar.get("l", 0) + bar.get("c", 0)) / 3
                        vol = bar.get("v", 0)
                        cum_tpv += tp * vol
                        cum_vol += vol
                    vwap_to_10am = cum_tpv / cum_vol if cum_vol > 0 else (price_10am or 0)
                    
                    cum_tpv_940, cum_vol_940 = 0, 0
                    for bar in vwap_940_bars:
                        tp = (bar.get("h", 0) + bar.get("l", 0) + bar.get("c", 0)) / 3
                        vol = bar.get("v", 0)
                        cum_tpv_940 += tp * vol
                        cum_vol_940 += vol
                    vwap_940 = cum_tpv_940 / cum_vol_940 if cum_vol_940 > 0 else vwap_to_10am
                    
                    or_high = max(or_highs) if or_highs else (price_10am or 0)
                    or_low = min(or_lows) if or_lows else (price_10am or 0)
                    
                    vwap_data_map[date_part] = {
                        "price_10am": price_10am,
                        "vwap": vwap_to_10am,
                        "vwap_940": vwap_940,
                        "or_high": or_high,
                        "or_low": or_low,
                    }
                
                logger.info(f"Found VWAP data for {len(vwap_data_map)} days for Rust VWAP")
                
                # Enrich daily bars with VWAP data
                bars_with_vwap = []
                for bar in bars:
                    bar_date = bar.get("t", "")[:10] if bar.get("t") else ""
                    new_bar = dict(bar)
                    vwap_info = vwap_data_map.get(bar_date, {})
                    new_bar["price_10am"] = vwap_info.get("price_10am", bar.get("o", 0))
                    new_bar["vwap"] = vwap_info.get("vwap", bar.get("c", 0))
                    new_bar["vwap_940"] = vwap_info.get("vwap_940", vwap_info.get("vwap", bar.get("c", 0)))
                    new_bar["or_high"] = vwap_info.get("or_high", bar.get("h", 0))
                    new_bar["or_low"] = vwap_info.get("or_low", bar.get("l", 0))
                    bars_with_vwap.append(new_bar)
                
                buy_range = (0.001, 0.03, 0.001)
                sell_range = (0.01, 0.10, 0.002)
                
                import numpy as np
                total_combinations = len(np.arange(buy_range[0], buy_range[1], buy_range[2])) * \
                                    len(np.arange(sell_range[0], sell_range[1], sell_range[2]))
                
                await broadcast_message({
                    "type": "progress",
                    "job_id": job_id,
                    "symbol": symbol,
                    "trials_completed": 0,
                    "total_trials": total_combinations,
                    "best_score": 0,
                    "best_params": {},
                    "phase": "rust_preparing",
                    "message": f"🦀 Rust Rayon optimizer starting ({total_combinations} combos across 32 threads)..."
                })
                
                # Run the Rust binary (parallel via Rayon)
                loop = asyncio.get_event_loop()
                gpu_result = await loop.run_in_executor(
                    None,
                    run_rust_vwap_optimization,
                    bars_with_vwap,
                    config.capital,
                    buy_range,
                    sell_range,
                    config.optimization_metric
                )
            else:
                # Default Dipper (breakout) strategy
                from gpu_backtest import run_gpu_optimization
                
                buy_range = (bounds["buy_trigger"][0]/100, bounds["buy_trigger"][1]/100, 0.001)
                sell_range = (bounds["sell_trigger"][0]/100, bounds["sell_trigger"][1]/100, 0.001)
                
                import numpy as np
                buy_count = len(np.arange(buy_range[0], buy_range[1], buy_range[2]))
                sell_count = len(np.arange(sell_range[0], sell_range[1], sell_range[2]))
                total_combinations = buy_count * sell_count
                
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
                
                loop = asyncio.get_event_loop()
                
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
            
            # Add algo and data_source to result
            algo_type = getattr(config, 'algo', 'default') or 'default'
            data_source = getattr(config, 'data_source', 'alpaca')
            logger.info(f"[GPU] Generating trade log for {symbol} with algo={algo_type}, data_source={data_source}")
            result["algo"] = algo_type
            result["data_source"] = data_source
            
            # Generate trade log for Analysis view
            # Use backtest dates if provided, otherwise use optimization dates
            bt_start = config.backtest_start_date or config.start_date
            bt_end = config.backtest_end_date or config.end_date
            
            # If backtest dates differ, fetch new price data for backtest range
            if bt_start != config.start_date or bt_end != config.end_date:
                logger.info(f"Fetching price data for backtest range: {bt_start} to {bt_end}")
                backtest_bars = fetch_and_cache_prices(symbol, bt_start, bt_end)
                trade_log_bars = backtest_bars if backtest_bars else bars
            else:
                trade_log_bars = bars
            
            # Broadcast: Generating trade log
            n_trade_log_bars = len(trade_log_bars)
            await broadcast_message({
                "type": "status",
                "job_id": job_id,
                "symbol": symbol,
                "status": "generating_trade_log",
                "message": f"Generating trade log ({n_trade_log_bars} bars)..."
            })
            
            result["trade_log"] = generate_trade_log(
                trade_log_bars,
                result["best_params"]["buy_trigger_pct"],
                result["best_params"]["sell_trigger_pct"],
                result["best_params"]["compound"],
                config.capital,
                algo=algo_type,
                symbol=symbol,
                start_date=bt_start,
                end_date=bt_end,
                stop_loss_pct=result["best_params"].get("stop_loss_pct"),
                trailing_stop_pct=result["best_params"].get("trailing_stop_pct")
            )
            result["backtest_start_date"] = bt_start
            result["backtest_end_date"] = bt_end
            
            # Broadcast: Saving to database
            await broadcast_message({
                "type": "status",
                "job_id": job_id,
                "symbol": symbol,
                "status": "saving_to_db",
                "message": f"Saving results ({len(result.get('trade_log', []))} trades)..."
            })
            
            # Save result to database
            history_id = save_result_to_db(job_id, result)
            result["history_id"] = history_id  # Add to result for frontend
            
            await broadcast_message({
                "type": "complete",
                "job_id": job_id,
                "symbol": symbol,
                "result": result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Parallel optimization failed: {e}")
            await broadcast_message({
                "type": "error",
                "job_id": job_id,
                "symbol": symbol,
                "message": f"Parallel optimization failed: {str(e)}. Falling back to sequential..."
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
            "compound": True,  # Always compound for consistent results
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

    # Add algo and data_source to result
    algo_type = getattr(config, 'algo', 'default') or 'default'
    data_source = getattr(config, 'data_source', 'alpaca')
    logger.info(f"Generating trade log for {symbol} with algo={algo_type}, data_source={data_source}")
    result["algo"] = algo_type
    result["data_source"] = data_source
    
    # Generate trade log for Analysis view
    # Use backtest dates if provided, otherwise use optimization dates
    bt_start = config.backtest_start_date or config.start_date
    bt_end = config.backtest_end_date or config.end_date
    
    # If backtest dates differ, fetch new price data for backtest range
    if bt_start != config.start_date or bt_end != config.end_date:
        logger.info(f"Fetching price data for backtest range: {bt_start} to {bt_end}")
        backtest_bars = fetch_and_cache_prices(symbol, bt_start, bt_end)
        trade_log_bars = backtest_bars if backtest_bars else bars
    else:
        trade_log_bars = bars
    
    result["trade_log"] = generate_trade_log(
        trade_log_bars,
        result["best_params"]["buy_trigger_pct"],
        result["best_params"]["sell_trigger_pct"],
        result["best_params"]["compound"],
        config.capital,
        algo=algo_type,
        symbol=symbol,
        start_date=bt_start,
        end_date=bt_end,
        stop_loss_pct=result["best_params"].get("stop_loss_pct"),
        trailing_stop_pct=result["best_params"].get("trailing_stop_pct")
    )
    result["backtest_start_date"] = bt_start
    result["backtest_end_date"] = bt_end
    
    # Save result to database
    history_id = save_result_to_db(job_id, result)
    result["history_id"] = history_id  # Add to result for frontend
    
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


# Track cancelled jobs
cancelled_jobs = set()


class CancelRequest(BaseModel):
    job_ids: List[str]


@app.post("/api/cancel")
async def cancel_optimization(request: CancelRequest):
    """Cancel running optimizations."""
    for job_id in request.job_ids:
        cancelled_jobs.add(job_id)
        logger.info(f"Cancellation requested for job: {job_id}")
    
    return {"status": "cancelled", "message": f"Cancelled {len(request.job_ids)} job(s)", "job_ids": request.job_ids}


def is_job_cancelled(job_id: str) -> bool:
    """Check if a job has been cancelled."""
    return job_id in cancelled_jobs


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
        semaphore = asyncio.Semaphore(16)  # Process 16 stocks at a time to keep it stable
        
        async def optimize_wrapper(symbol):
            async with semaphore:
                if is_job_cancelled(job_id):
                    return None
                return await optimize_stock(symbol, request, job_id)

        # Create tasks for all symbols
        tasks = [optimize_wrapper(symbol) for symbol in request.symbols]
        
        # Run tasks in parallel and collect results
        completed_count = 0
        total_count = len(request.symbols)
        
        for f in asyncio.as_completed(tasks):
            res = await f
            if res:
                symbol = res.get("symbol")
                if symbol:
                    results[symbol] = res
                    optimization_results[symbol] = res
            
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == total_count:
                logger.info(f"Job {job_id} progress: {completed_count}/{total_count} stocks completed")
        
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


# NOTE: /api/price-compare endpoint moved to routers/api_tester.py
# The endpoint is now imported and registered via app.include_router()


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



# NOTE: /api/history endpoints moved to routers/history.py


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

