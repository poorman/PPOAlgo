#!/usr/bin/env python3
import os
import json
import logging
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
import sys

# Add current dir to path to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DB_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"

def get_db_conn():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def convert_numpy_types(obj):
    # Already have this in database.py, but re-implementing here for simplicity
    import numpy as np
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
    return obj

def migrate():
    conn = get_db_conn()
    if not conn:
        return
    
    try:
        cur = conn.cursor()
        
        files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
        logger.info(f"Found {len(files)} JSON result files")
        
        imported_count = 0
        job_count = 0
        
        for filename in sorted(files):
            job_id = filename.replace(".json", "")
            filepath = os.path.join(RESULTS_DIR, filename)
            
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                if not data or not isinstance(data, dict):
                    continue
                
                # Try to extract job info from the first symbol
                first_symbol = next(iter(data))
                if not isinstance(data[first_symbol], dict):
                    continue
                    
                sample = data[first_symbol]
                
                # Create a job entry
                symbols = list(data.keys())
                start_date = sample.get("requested_start_date", "2024-01-01")
                end_date = sample.get("requested_end_date", "2024-12-31")
                capital = sample.get("initial_capital", 100000.0)
                n_trials = sample.get("n_trials", 100)
                metric = sample.get("optimized_for", "total_return")
                created_at = sample.get("timestamp", datetime.now().isoformat())
                
                cur.execute("""
                    INSERT INTO optimizer_jobs 
                    (job_id, symbols, start_date, end_date, capital, n_trials, optimization_metric, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'completed', %s)
                    ON CONFLICT (job_id) DO NOTHING
                """, (job_id, symbols, start_date, end_date, capital, n_trials, metric, created_at))
                
                job_count += 1
                
                # Insert results
                for symbol, res in data.items():
                    if not isinstance(res, dict) or "best_params" not in res:
                        continue
                        
                    score = 0
                    if res.get("metrics"):
                        score_values = list(res["metrics"].values())
                        if score_values:
                            score = float(score_values[0]) if score_values[0] is not None else 0
                    
                    vol_profile = res.get("volatility_profile", {})
                    avg_range = vol_profile.get("avg_daily_range", 0)
                    max_gain = vol_profile.get("max_daily_gain", 0)
                    vol_score = vol_profile.get("volatility_score", 0)
                    
                    # Ensure numeric types
                    try:
                        buy_pct = float(res["best_params"].get("buy_trigger_pct", 0))
                        sell_pct = float(res["best_params"].get("sell_trigger_pct", 1.0))
                    except:
                        buy_pct, sell_pct = 0, 1.0
                        
                    cur.execute("""
                        INSERT INTO optimizer_results 
                        (job_id, symbol, buy_trigger_pct, sell_trigger_pct, compound, 
                         optimized_for, score, volatility_avg_range, volatility_max_gain, 
                         volatility_score, optimal_buy_time_cdt, full_result, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (job_id, symbol) DO NOTHING
                    """, (
                        job_id,
                        symbol,
                        buy_pct,
                        sell_pct,
                        bool(res["best_params"].get("compound", True)),
                        res.get("optimized_for", "total_return"),
                        score,
                        float(avg_range),
                        float(max_gain),
                        float(vol_score),
                        res.get("optimal_buy_time_cdt"),
                        Json(res),
                        res.get("timestamp", datetime.now().isoformat())
                    ))
                    imported_count += 1
                
                if job_count % 10 == 0:
                    conn.commit()
                    logger.info(f"Progress: {job_count} jobs, {imported_count} results imported...")
                    
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")
                continue
        
        conn.commit()
        logger.info(f"Done! Imported {job_count} jobs and {imported_count} results.")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
