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


def calculate_score(fin_value: float, position: int, total: int) -> float:
    """
    Calculate score based on fin (total return) and position in the sorted list.
    
    Score ranges:
    - 99.99 to 90.00: Stocks with profit >= 115% (high performers)
    - 89.99 to 80.00: Stocks with profit >= 100% but < 115% (profitable)  
    - 79.99 to 70.00: Stocks with profit >= 0% but < 100% (marginally profitable)
    - 69.99 to 60.00: Stocks with negative profit (losing money)
    
    Within each tier, scores decrease based on position.
    """
    # Determine tier based on fin value (total return percentage)
    # fin_value is the total return percentage (e.g., 115.5 means 115.5% return)
    if fin_value >= 115.0:
        # High performers: 99.99 to 90.00
        tier_min, tier_max = 90.00, 99.99
    elif fin_value >= 100.0:
        # Profitable but < 115%: 89.99 to 80.00
        tier_min, tier_max = 80.00, 89.99
    elif fin_value >= 0.0:
        # Marginally profitable (0-100%): 79.99 to 70.00
        tier_min, tier_max = 70.00, 79.99
    else:
        # Losing money: 69.99 to 60.00
        tier_min, tier_max = 60.00, 69.99
    
    # Calculate position within tier (normalize to 0-1)
    # position is 0-indexed, so position 0 = highest score in tier
    if total > 1:
        # Spread scores within the tier based on position
        tier_range = tier_max - tier_min
        score = tier_max - (position / (total - 1)) * tier_range if total > 1 else tier_max
    else:
        score = tier_max
    
    return round(score, 2)


@router.get("/list")
async def get_ai100_list(
    auth: str = Query(..., description="API authentication key"),
    sort: str = Query("fin", description="Sort by: fin (final return), amnt (amount), or win2 (win/loss ratio)"),
    limit: int = Query(100, description="Number of stocks to return (default: 100)")
):
    """
    Get a sorted list of optimized stocks for AI trading.
    
    Returns stocks sorted by the specified metric:
    - fin: Sort by final return (total_return) descending
    - amnt: Sort by amount/score descending  
    - win2: Sort by win ratio (wins / total trades) descending
    
    Score is calculated based on fin (total return):
    - 99.99-90.00: Profit >= 115%
    - 89.99-80.00: Profit >= 100% but < 115%
    - 79.99-70.00: Profit >= 0% but < 100%
    - 69.99-60.00: Negative profit
    
    Response format:
    {
        "success": true,
        "data": {
            "stocks": [
                {
                    "pair": "AAPL",
                    "score": 95.50,
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
    
    # Validate sort parameter - accept both 'mnt' and 'amnt' for backwards compatibility
    sort = sort.lower()
    if sort == "mnt":
        sort = "amnt"  # Map mnt to amnt for backwards compatibility
    if sort not in ["fin", "amnt", "win2"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid sort parameter. Must be 'fin', 'amnt', or 'win2'"
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
            # Get all latest results for each symbol (we'll sort in Python for win2)
            query = """
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
            """
            cur.execute(query)
            
            rows = cur.fetchall()
            
            # Process all rows first
            processed_stocks = []
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
                
                # Calculate win ratio the same way frontend does: wins / total trades
                total_trades = wins + losses
                calculated_win_ratio = (wins / total_trades * 100) if total_trades > 0 else 0.0
                
                processed_stocks.append({
                    "symbol": symbol,
                    "fin_value": fin_value,
                    "score_value": score_value,
                    "capital_value": capital_value,
                    "wins": wins,
                    "losses": losses,
                    "calculated_win_ratio": calculated_win_ratio,  # Use calculated ratio for sorting
                    "win2_str": f"{wins}_{losses}"
                })
            
            # Sort based on the requested criteria
            if sort == "fin":
                # Sort by fin (total return) descending
                processed_stocks.sort(key=lambda x: x["fin_value"], reverse=True)
            elif sort == "amnt":
                # Sort by score/amount descending
                processed_stocks.sort(key=lambda x: x["score_value"], reverse=True)
            else:  # win2
                # Sort by calculated win ratio descending (same as UI: wins/total trades)
                processed_stocks.sort(key=lambda x: x["calculated_win_ratio"], reverse=True)
            
            # Apply limit
            processed_stocks = processed_stocks[:limit]
            
            # Group stocks by tier for score calculation
            tier_groups = {
                "high": [],      # fin >= 115%
                "medium": [],    # fin >= 100% but < 115%
                "low": [],       # fin >= 0% but < 100%
                "negative": []   # fin < 0%
            }
            
            for i, stock in enumerate(processed_stocks):
                fin = stock["fin_value"]
                if fin >= 115.0:
                    tier_groups["high"].append((i, stock))
                elif fin >= 100.0:
                    tier_groups["medium"].append((i, stock))
                elif fin >= 0.0:
                    tier_groups["low"].append((i, stock))
                else:
                    tier_groups["negative"].append((i, stock))
            
            # Calculate scores within each tier
            stock_scores = {}
            
            for tier_name, tier_stocks in tier_groups.items():
                tier_count = len(tier_stocks)
                if tier_name == "high":
                    tier_min, tier_max = 90.00, 99.99
                elif tier_name == "medium":
                    tier_min, tier_max = 80.00, 89.99
                elif tier_name == "low":
                    tier_min, tier_max = 70.00, 79.99
                else:  # negative
                    tier_min, tier_max = 60.00, 69.99
                
                # Sort stocks within tier by fin to assign scores properly
                tier_stocks_sorted = sorted(tier_stocks, key=lambda x: x[1]["fin_value"], reverse=True)
                
                for pos, (orig_idx, stock) in enumerate(tier_stocks_sorted):
                    if tier_count > 1:
                        tier_range = tier_max - tier_min
                        score = tier_max - (pos / (tier_count - 1)) * tier_range
                    else:
                        score = tier_max
                    stock_scores[orig_idx] = round(score, 2)
            
            # Build final response
            stocks = []
            for i, stock in enumerate(processed_stocks):
                calculated_score = stock_scores.get(i, 70.00)
                
                stocks.append({
                    "pair": stock["symbol"],
                    "score": calculated_score,
                    "details": [{
                        "fin": round(stock["fin_value"], 2),
                        "amnt": round(stock["capital_value"], 2),
                        "win2": stock["win2_str"]
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
