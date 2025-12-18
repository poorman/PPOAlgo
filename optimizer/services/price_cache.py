#!/usr/bin/env python3
"""
Price Cache Service
Functions for caching and retrieving stock price data from database.
"""

import logging
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import requests

from database import get_db_conn
from config import PPOALGO_API

logger = logging.getLogger(__name__)


def get_cached_price_range(symbol: str, timeframe: str) -> Tuple[Optional[datetime], Optional[datetime]]:
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


def get_cached_prices(symbol: str, timeframe: str, start_date: str, end_date: str) -> List[dict]:
    """Get cached prices from database. Checks both price_cache and legacy tables."""
    conn = get_db_conn()
    if not conn:
        return []
    
    try:
        bars = []
        with conn.cursor() as cur:
            if timeframe == "1Day":
                # First try price_cache table
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
                # For intraday, check price_cache
                cur.execute("""
                    SELECT bar_timestamp, open, high, low, close, volume, vwap
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
                
                bar_date = bar_time_str[:10]
                
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
                    bar.get("vw", None)
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


def fetch_and_cache_prices(symbol: str, start_date: str, end_date: str, timeframe: str = "1Day") -> List[dict]:
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
    symbol = symbol.upper()
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Check what we have cached
    cached_min, cached_max = get_cached_price_range(symbol, timeframe)
    logger.info(f"Price cache for {symbol} ({timeframe}): {cached_min} to {cached_max}")
    
    # Determine what data we need to fetch
    fetch_ranges = []
    
    if cached_min is None:
        fetch_ranges.append((start_date, end_date))
    else:
        if start_dt.date() < cached_min:
            fetch_ranges.append((start_date, (cached_min - timedelta(days=1)).strftime("%Y-%m-%d")))
        if end_dt.date() > cached_max:
            fetch_ranges.append(((cached_max + timedelta(days=1)).strftime("%Y-%m-%d"), end_date))
    
    # Fetch missing data
    for fetch_start, fetch_end in fetch_ranges:
        logger.info(f"Fetching {symbol} ({timeframe}) from {fetch_start} to {fetch_end}")
        
        if timeframe == "1Day":
            try:
                resp = requests.get(
                    f"{PPOALGO_API}/api/prices",
                    params={"symbol": symbol, "start": fetch_start, "end": fetch_end, "timeframe": "1Day"},
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
            # Fetch intraday data in monthly chunks
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
                            sample_t = bars[0].get("t", "")
                            if "T" in sample_t and not sample_t.endswith("T05:00:00Z"):
                                save_prices_to_cache(symbol, timeframe, bars)
                except Exception as e:
                    logger.error(f"Failed to fetch intraday prices: {e}")
                
                current = chunk_end + timedelta(days=1)
    
    # Return cached data for the requested range
    return get_cached_prices(symbol, timeframe, start_date, end_date)
