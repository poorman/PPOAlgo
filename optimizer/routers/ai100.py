#!/usr/bin/env python3
"""
AI100 Router
Provides /api/ai100 endpoints for external API access to stock optimization results.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from database import get_db_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai100", tags=["AI100"])

# Hardcoded API key for authentication
API_KEY = "pluq8P0XTgucCN6kyxey5EPTof36R54lQc3rfgQsoNQ"


@router.get("/list")
async def get_ai100_list(
    auth: str = Query(..., description="API authentication key"),
    sort: str = Query("fin", description="Sort by: fin (final return), mnt (amount), or win2 (win/loss ratio)"),
    limit: int = Query(100, description="Number of stocks to return (default: 100)")
):
    """
    Get a sorted list of optimized stocks for AI trading.
    
    Returns stocks sorted by the specified metric:
    - fin: Sort by final return (total_return) descending
    - mnt: Sort by amount/score descending  
    - win2: Sort by win rate descending
    
    Response format:
    {
        "success": true,
        "data": {
            "stocks": [
                {
                    "pair": "AAPL",
                    "score": 1234.56,
                    "details": [{
                        "fin": 1234.56,
                        "amnt": 100000,
                        "win2": "29_8"
                    }]
                }
            ],
            "count": 100
        }
    }
    """
    # Validate API key
    if auth != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Validate sort parameter
    sort = sort.lower()
    if sort not in ["fin", "mnt", "win2"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid sort parameter. Must be 'fin', 'mnt', or 'win2'"
        )
    
    # Validate limit
    if limit < 1:
        limit = 1
    if limit > 1000:
        limit = 1000
    
    conn = get_db_conn()
    if not conn:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        with conn.cursor() as cur:
            # Determine sort column based on parameter
            if sort == "fin":
                order_by = "CAST(total_return AS NUMERIC) DESC NULLS LAST"
            elif sort == "mnt":
                order_by = "score DESC NULLS LAST"
            else:  # win2
                order_by = "CAST(win_rate AS NUMERIC) DESC NULLS LAST"
            
            # Get the most recent result for each symbol, sorted by the specified metric
            query = f"""
                WITH latest_results AS (
                    SELECT DISTINCT ON (symbol) 
                        r.symbol,
                        r.score,
                        r.full_result->'metrics'->>'total_return' as total_return,
                        r.full_result->'metrics'->>'win_rate' as win_rate,
                        r.full_result->'trade_log' as trade_log,
                        j.capital,
                        r.created_at
                    FROM optimizer_results r
                    LEFT JOIN optimizer_jobs j ON r.job_id = j.job_id
                    ORDER BY symbol, r.created_at DESC
                )
                SELECT 
                    symbol,
                    score,
                    total_return,
                    win_rate,
                    trade_log,
                    capital
                FROM latest_results
                ORDER BY {order_by}
                LIMIT %s
            """
            cur.execute(query, (limit,))
            
            rows = cur.fetchall()
            
            stocks = []
            for row in rows:
                symbol, score, total_return, win_rate, trade_log, capital = row
                
                # Parse values
                try:
                    fin_value = float(total_return) if total_return else 0.0
                except (ValueError, TypeError):
                    fin_value = 0.0
                
                try:
                    score_value = float(score) if score else 0.0
                except (ValueError, TypeError):
                    score_value = 0.0
                
                try:
                    capital_value = float(capital) if capital else 100000
                except (ValueError, TypeError):
                    capital_value = 100000
                
                # Calculate win/loss counts from trade_log
                wins = 0
                losses = 0
                if trade_log and isinstance(trade_log, list):
                    for trade in trade_log:
                        if isinstance(trade, dict):
                            # Only count trades that were actually executed
                            if trade.get("bought") == True:
                                profit = trade.get("profit", 0)
                                if profit is not None and profit > 0:
                                    wins += 1
                                elif profit is not None and profit < 0:
                                    losses += 1
                
                win2_str = f"{wins}_{losses}"
                
                # Calculate score based on sort type for proper ranking
                if sort == "fin":
                    display_score = fin_value
                elif sort == "mnt":
                    display_score = score_value
                else:  # win2
                    display_score = float(win_rate) if win_rate else 0.0
                
                stocks.append({
                    "pair": symbol,
                    "score": round(display_score, 2),
                    "details": [{
                        "fin": round(fin_value, 2),
                        "amnt": round(capital_value, 2),
                        "win2": win2_str
                    }]
                })
            
            return {
                "success": True,
                "data": {
                    "stocks": stocks,
                    "count": len(stocks)
                }
            }
            
    except Exception as e:
        logger.error(f"Failed to get AI100 list: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
