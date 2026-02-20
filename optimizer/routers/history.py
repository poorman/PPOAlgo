#!/usr/bin/env python3
"""
History Router
Provides /api/history endpoints for viewing and managing optimization history.
"""

import logging
from fastapi import APIRouter, HTTPException

from database import get_db_conn, release_db_conn, get_jobs_with_summary, get_job_details

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["History"])


@router.get("/history/count")
async def get_history_count():
    """Get total number of optimization results in database."""
    conn = get_db_conn()
    if not conn:
        return {"count": 0}
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM optimizer_results")
            count = cur.fetchone()[0]
            return {"count": count}
    except Exception as e:
        logger.error(f"Failed to get history count: {e}")
        return {"count": 0}
    finally:
        if conn:
            release_db_conn(conn)


@router.get("/history")
async def get_history(limit: int = 10000):
    """Get optimization history from database. Default limit 1000 for fast loading."""
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
                    r.full_result->'best_params'->>'stop_loss_pct' as stop_loss_pct,
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
                    r.full_result->'metrics'->>'final_equity' as final_equity,
                    r.full_result->'metrics'->>'total_trades' as total_trades,
                    r.full_result->'metrics'->>'max_drawdown' as max_drawdown,
                    r.full_result->>'duration_seconds' as duration_seconds,
                    r.full_result->>'method' as method,
                    r.full_result->>'algo' as algo,
                    r.full_result->>'data_source' as data_source,
                    r.full_result->>'backtest_start_date' as backtest_start_date,
                    r.full_result->>'backtest_end_date' as backtest_end_date,
                    j.smart_timing
                FROM optimizer_results r
                LEFT JOIN optimizer_jobs j ON r.job_id = j.job_id
                ORDER BY r.created_at DESC
                LIMIT %s
            """
            cur.execute(query, (limit,))
            
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
                for key in ["buy_trigger_pct", "sell_trigger_pct", "stop_loss_pct", "score", 
                           "volatility_avg_range", "volatility_max_gain", 
                           "volatility_score", "capital", "total_return", "win_rate", "sharpe_from_json", "final_equity",
                           "total_trades", "max_drawdown"]:
                    if result.get(key) is not None:
                        try:
                            result[key] = float(result[key])
                        except (ValueError, TypeError):
                            pass
                # Parse trade_log if it's a string (JSON from PostgreSQL)
                if result.get("trade_log"):
                    if isinstance(result["trade_log"], str):
                        try:
                            import json
                            result["trade_log"] = json.loads(result["trade_log"])
                        except (json.JSONDecodeError, TypeError):
                            result["trade_log"] = []
                    elif not isinstance(result["trade_log"], list):
                        result["trade_log"] = []
                else:
                    result["trade_log"] = []
                results.append(result)
            
            return results
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return []
    finally:
        if conn:
            release_db_conn(conn)


@router.get("/history/{item_id}")
async def get_history_detail(item_id: int):
    """Get full details for a single optimization result, including trade_log."""
    conn = get_db_conn()
    if not conn:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        with conn.cursor() as cur:
            query = """
                SELECT 
                    r.id, r.job_id, r.symbol, r.buy_trigger_pct, r.sell_trigger_pct,
                    r.full_result->'best_params'->>'stop_loss_pct' as stop_loss_pct,
                    r.compound, r.optimized_for, r.score,
                    r.volatility_avg_range, r.volatility_max_gain, r.volatility_score,
                    r.optimal_buy_time_cdt, r.created_at,
                    j.start_date, j.end_date, j.capital, j.n_trials,
                    r.full_result->'metrics'->>'total_return' as total_return,
                    r.full_result->'metrics'->>'win_rate' as win_rate,
                    r.full_result->'metrics'->>'sharpe' as sharpe_from_json,
                    r.full_result->'metrics'->>'final_equity' as final_equity,
                    r.full_result->>'duration_seconds' as duration_seconds,
                    r.full_result->>'method' as method,
                    r.full_result->>'algo' as algo,
                    r.full_result->>'data_source' as data_source,
                    r.full_result->>'backtest_start_date' as backtest_start_date,
                    r.full_result->>'backtest_end_date' as backtest_end_date,
                    r.full_result->'trade_log' as trade_log,
                    j.smart_timing
                FROM optimizer_results r
                LEFT JOIN optimizer_jobs j ON r.job_id = j.job_id
                WHERE r.id = %s
            """
            cur.execute(query, (item_id,))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Result not found")
            
            columns = [desc[0] for desc in cur.description]
            result = dict(zip(columns, row))
            
            # Formatting
            if result.get("created_at"):
                result["created_at"] = result["created_at"].isoformat()
            if result.get("start_date"):
                result["start_date"] = str(result["start_date"])
            if result.get("end_date"):
                result["end_date"] = str(result["end_date"])
                
            for key in ["buy_trigger_pct", "sell_trigger_pct", "stop_loss_pct", "score", 
                       "volatility_avg_range", "volatility_max_gain", 
                       "volatility_score", "capital", "total_return", "win_rate", "sharpe_from_json", "final_equity"]:
                if result.get(key) is not None:
                    try:
                        result[key] = float(result[key])
                    except (ValueError, TypeError):
                        pass

            # Parse trade_log
            if result.get("trade_log"):
                if isinstance(result["trade_log"], str):
                    try:
                        import json
                        result["trade_log"] = json.loads(result["trade_log"])
                    except:
                        result["trade_log"] = []
                elif not isinstance(result["trade_log"], list):
                    result["trade_log"] = []
            else:
                result["trade_log"] = []
                
            return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get detail for {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            release_db_conn(conn)


@router.delete("/history")
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
        release_db_conn(conn)


@router.delete("/history/{result_id}")
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
        release_db_conn(conn)


# ── Job log endpoints ─────────────────────────────────────────────────────────

@router.get("/jobs")
async def list_jobs(limit: int = 100):
    """Get all optimizer jobs with aggregated metrics for the Logs tab."""
    return get_jobs_with_summary(limit)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get per-symbol results for a specific job."""
    results = get_job_details(job_id)
    if not results:
        raise HTTPException(status_code=404, detail="Job not found or has no results")
    return results


@router.delete("/jobs")
async def delete_all_jobs():
    """Delete all optimizer job logs (preserves optimization results/history)."""
    conn = get_db_conn()
    if not conn:
        raise HTTPException(status_code=500, detail="Database not available")

    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM optimizer_jobs")
            jobs_deleted = cur.rowcount
        conn.commit()
        logger.info(f"Cleared job logs: {jobs_deleted} jobs deleted (results preserved)")
        return {"status": "ok", "jobs_deleted": jobs_deleted}
    except Exception as e:
        logger.error(f"Failed to delete jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        release_db_conn(conn)
