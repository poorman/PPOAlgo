#!/usr/bin/env python3
"""
Database module for Stock Optimizer GUI.
Contains PostgreSQL connection handling, table creation, and CRUD operations.
"""

import logging
from typing import List, Optional

import numpy as np

from config import DB_CONFIG

logger = logging.getLogger(__name__)

# Try importing psycopg2 for PostgreSQL
try:
    import psycopg2
    from psycopg2.extras import Json, execute_values
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.warning("psycopg2 not installed, database features disabled")


# Database connection pool for performance
from psycopg2 import pool
db_pool = None

def init_db_pool():
    global db_pool
    if POSTGRES_AVAILABLE and db_pool is None:
        try:
            # Use ThreadedConnectionPool for thread safety since we use ThreadPoolExecutor
            db_pool = pool.ThreadedConnectionPool(5, 200, **DB_CONFIG)
            logger.info("Threaded database connection pool initialized (5-200 connections)")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")

def get_db_conn():
    """Get a connection from the pool."""
    global db_pool
    if not POSTGRES_AVAILABLE:
        return None
    
    if db_pool is None:
        init_db_pool()
        
    try:
        return db_pool.getconn()
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def release_db_conn(conn):
    """Release a connection back to the pool."""
    global db_pool
    if db_pool and conn:
        db_pool.putconn(conn)


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

            # Add algo and data_source columns to optimizer_jobs (migration)
            cur.execute("""
                ALTER TABLE optimizer_jobs
                ADD COLUMN IF NOT EXISTS algo VARCHAR(50) DEFAULT 'default';
            """)
            cur.execute("""
                ALTER TABLE optimizer_jobs
                ADD COLUMN IF NOT EXISTS data_source VARCHAR(50) DEFAULT 'widesurf';
            """)
            cur.execute("""
                ALTER TABLE optimizer_jobs
                ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ;
            """)
            
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
        if conn:
            release_db_conn(conn)


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


def save_job_to_db(job_id: str, config):
    """Save optimization job to database."""
    conn = get_db_conn()
    if not conn:
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO optimizer_jobs
                (job_id, symbols, start_date, end_date, capital, n_trials, optimization_metric, smart_timing, algo, data_source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (job_id) DO NOTHING
            """, (
                job_id,
                config.symbols,
                config.start_date,
                config.end_date,
                config.capital,
                config.n_trials,
                config.optimization_metric,
                config.smart_timing,
                getattr(config, 'algo', 'default'),
                getattr(config, 'data_source', 'widesurf'),
            ))
        conn.commit()
        logger.info(f"Saved job {job_id} to database")
    except Exception as e:
        logger.error(f"Failed to save job: {e}")
    finally:
        if conn:
            release_db_conn(conn)


def save_result_to_db(job_id: str, result: dict) -> Optional[int]:
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
                result.get("optimal_buy_time_cdt"),
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
        if conn:
            release_db_conn(conn)


def update_job_status(job_id: str, status: str):
    """Update job status in database."""
    conn = get_db_conn()
    if not conn:
        return
    
    try:
        with conn.cursor() as cur:
            if status == "completed":
                cur.execute("""
                    UPDATE optimizer_jobs SET status = %s, completed_at = NOW() WHERE job_id = %s
                """, (status, job_id))
            else:
                cur.execute("""
                    UPDATE optimizer_jobs SET status = %s WHERE job_id = %s
                """, (status, job_id))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")
    finally:
        if conn:
            release_db_conn(conn)


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
        if conn:
            release_db_conn(conn)


# ============================================================================
# JOB LOG QUERIES
# ============================================================================

def get_jobs_with_summary(limit: int = 100) -> list:
    """Get all optimizer jobs with aggregated result metrics."""
    conn = get_db_conn()
    if not conn:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT j.job_id, j.symbols, j.start_date, j.end_date, j.capital,
                       j.n_trials, j.optimization_metric, j.status, j.smart_timing,
                       j.created_at, j.algo, j.data_source,
                       COUNT(r.id) AS result_count,
                       AVG((r.full_result->'metrics'->>'total_return')::numeric) AS avg_return,
                       AVG((r.full_result->'metrics'->>'win_rate')::numeric) AS avg_win_rate,
                       EXTRACT(EPOCH FROM (j.completed_at - j.created_at)) AS duration_seconds
                FROM optimizer_jobs j
                LEFT JOIN optimizer_results r ON j.job_id = r.job_id
                GROUP BY j.id
                ORDER BY j.created_at DESC
                LIMIT %s
            """, (limit,))
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
            results = []
            for row in rows:
                d = dict(zip(columns, row))
                if d.get("created_at"):
                    d["created_at"] = d["created_at"].isoformat()
                if d.get("start_date"):
                    d["start_date"] = str(d["start_date"])
                if d.get("end_date"):
                    d["end_date"] = str(d["end_date"])
                # symbols is a Postgres TEXT[] — already comes back as list
                for key in ("capital", "avg_return", "avg_win_rate", "duration_seconds"):
                    if d.get(key) is not None:
                        try:
                            d[key] = float(d[key])
                        except (ValueError, TypeError):
                            pass
                results.append(d)
            return results
    except Exception as e:
        logger.error(f"Failed to get jobs summary: {e}")
        return []
    finally:
        if conn:
            release_db_conn(conn)


def get_job_details(job_id: str) -> list:
    """Get all per-symbol results for a specific job."""
    conn = get_db_conn()
    if not conn:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT r.symbol,
                       r.buy_trigger_pct, r.sell_trigger_pct, r.compound,
                       r.score, r.optimal_buy_time_cdt,
                       r.full_result->'metrics'->>'total_return' AS total_return,
                       r.full_result->'metrics'->>'win_rate' AS win_rate,
                       r.full_result->'metrics'->>'sharpe' AS sharpe,
                       r.full_result->'metrics'->>'final_equity' AS final_equity,
                       r.full_result->'metrics'->>'total_trades' AS total_trades,
                       r.full_result->>'duration_seconds' AS duration_seconds,
                       r.full_result->>'method' AS method,
                       r.full_result->>'algo' AS algo,
                       r.full_result->>'data_source' AS data_source
                FROM optimizer_results r
                WHERE r.job_id = %s
                ORDER BY (r.full_result->'metrics'->>'total_return')::numeric DESC NULLS LAST
            """, (job_id,))
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
            results = []
            for row in rows:
                d = dict(zip(columns, row))
                for key in ("buy_trigger_pct", "sell_trigger_pct", "score",
                            "total_return", "win_rate", "sharpe",
                            "final_equity", "total_trades", "duration_seconds"):
                    if d.get(key) is not None:
                        try:
                            d[key] = float(d[key])
                        except (ValueError, TypeError):
                            pass
                results.append(d)
            return results
    except Exception as e:
        logger.error(f"Failed to get job details for {job_id}: {e}")
        return []
    finally:
        if conn:
            release_db_conn(conn)


# ============================================================================
# KEYWORD CONFIGURATIONS
# ============================================================================

def ensure_keyword_configs_table():
    """Create keyword_configs table if it doesn't exist."""
    conn = get_db_conn()
    if not conn:
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS keyword_configs (
                    id SERIAL PRIMARY KEY,
                    keyword VARCHAR(20) UNIQUE NOT NULL,
                    api_url TEXT NOT NULL,
                    description TEXT,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        conn.commit()
        logger.info("keyword_configs table ensured")
    except Exception as e:
        logger.error(f"Failed to create keyword_configs table: {e}")
    finally:
        if conn:
            release_db_conn(conn)


def get_keyword_configs() -> dict:
    """Get all keyword configurations from database."""
    conn = get_db_conn()
    if not conn:
        return {}
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT keyword, api_url, description FROM keyword_configs
            """)
            rows = cur.fetchall()
            return {row[0]: {"api_url": row[1], "description": row[2]} for row in rows}
    except Exception as e:
        logger.error(f"Failed to get keyword configs: {e}")
        return {}
    finally:
        if conn:
            release_db_conn(conn)


def save_keyword_config(keyword: str, api_url: str, description: str = None):
    """Save or update a keyword configuration."""
    conn = get_db_conn()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO keyword_configs (keyword, api_url, description, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (keyword) DO UPDATE SET
                    api_url = EXCLUDED.api_url,
                    description = EXCLUDED.description,
                    updated_at = NOW()
            """, (keyword.upper(), api_url, description))
        conn.commit()
        logger.info(f"Saved keyword config: {keyword}")
        return True
    except Exception as e:
        logger.error(f"Failed to save keyword config: {e}")
        return False
    finally:
        if conn:
            release_db_conn(conn)


def bulk_save_prices(symbol: str, timeframe: str, bars: list):
    """Save multiple price bars to the cache database using optimized bulk insertion."""
    if not bars:
        return
    
    conn = get_db_conn()
    if not conn:
        return
        
    try:
        # Prepare data for execute_values
        data = []
        for bar in bars:
            bar_time_str = bar.get("t", "")
            if not bar_time_str:
                continue
            
            # Parse date from timestamp (first 10 chars of ISO string)
            bar_date = bar_time_str[:10]
            
            data.append((
                symbol.upper(),
                timeframe,
                bar_date,
                bar_time_str,
                float(bar.get("o", 0)),
                float(bar.get("h", 0)),
                float(bar.get("l", 0)),
                float(bar.get("c", 0)),
                int(bar.get("v", 0)) if bar.get("v") else 0,
                float(bar.get("vw", 0)) if bar.get("vw") else None
            ))

        with conn.cursor() as cur:
            # Use execute_values for efficient bulk insertion
            execute_values(cur, """
                INSERT INTO price_cache 
                (symbol, timeframe, bar_date, bar_timestamp, open, high, low, close, volume, vwap)
                VALUES %s
                ON CONFLICT (symbol, timeframe, bar_date, bar_timestamp) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                vwap = EXCLUDED.vwap
            """, data)
        
        conn.commit()
    except Exception as e:
        logger.error(f"Failed bulk save: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            release_db_conn(conn)


def save_all_keyword_configs(configs: dict) -> bool:
    """Save multiple keyword configurations at once."""
    conn = get_db_conn()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            for keyword, data in configs.items():
                api_url = data.get("api_url", "")
                description = data.get("description", "")
                cur.execute("""
                    INSERT INTO keyword_configs (keyword, api_url, description, updated_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (keyword) DO UPDATE SET
                        api_url = EXCLUDED.api_url,
                        description = EXCLUDED.description,
                        updated_at = NOW()
                """, (keyword.upper(), api_url, description))
        conn.commit()
        logger.info(f"Saved {len(configs)} keyword configs")
        return True
    except Exception as e:
        logger.error(f"Failed to save keyword configs: {e}")
        return False
    finally:
        if conn:
            release_db_conn(conn)

# ============================================================================
# API KEY CONFIGURATIONS
# ============================================================================

def ensure_api_keys_table():
    """Create api_keys table if it doesn't exist."""
    conn = get_db_conn()
    if not conn:
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id SERIAL PRIMARY KEY,
                    provider VARCHAR(50) UNIQUE NOT NULL,
                    key_id TEXT,
                    secret_key TEXT,
                    base_url TEXT,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        conn.commit()
        logger.info("api_keys table ensured")
    except Exception as e:
        logger.error(f"Failed to create api_keys table: {e}")
    finally:
        if conn:
            release_db_conn(conn)


def get_api_keys() -> dict:
    """Get all API keys from database."""
    conn = get_db_conn()
    if not conn:
        return {}
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT provider, key_id, secret_key, base_url FROM api_keys
            """)
            rows = cur.fetchall()
            return {row[0]: {"key_id": row[1], "secret_key": row[2], "base_url": row[3]} for row in rows}
    except Exception as e:
        logger.error(f"Failed to get api keys: {e}")
        return {}
    finally:
        if conn:
            release_db_conn(conn)


def save_api_key(provider: str, key_id: str = None, secret_key: str = None, base_url: str = None):
    """Save or update an API key configuration."""
    conn = get_db_conn()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO api_keys (provider, key_id, secret_key, base_url, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (provider) DO UPDATE SET
                    key_id = COALESCE(EXCLUDED.key_id, api_keys.key_id),
                    secret_key = COALESCE(EXCLUDED.secret_key, api_keys.secret_key),
                    base_url = COALESCE(EXCLUDED.base_url, api_keys.base_url),
                    updated_at = NOW()
            """, (provider.upper(), key_id, secret_key, base_url))
        conn.commit()
        logger.info(f"Saved API key for provider: {provider}")
        return True
    except Exception as e:
        logger.error(f"Failed to save API key: {e}")
        return False
    finally:
        if conn:
            release_db_conn(conn)
