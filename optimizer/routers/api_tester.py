#!/usr/bin/env python3
"""
API Tester Router
Provides /api/price-compare endpoint for comparing Alpaca vs Polygon data.
"""

import os
import logging
import requests
from datetime import datetime
from fastapi import APIRouter

from config import PPOALGO_API, POLYGON_API_KEY

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["API Tester"])


def get_et_minutes(ts: str) -> int | None:
    """Convert UTC timestamp to Eastern Time minutes since midnight."""
    if "T" in ts:
        hour_str = ts.split("T")[1][:5]
        h, m = map(int, hour_str.split(":"))
        et_min = (h - 5) * 60 + m  # UTC to EST
        if et_min < 0:
            et_min += 1440
        return et_min
    return None


def get_poly_et_minutes(ts_ms: int) -> int:
    """Convert Polygon timestamp (ms since epoch) to Eastern Time minutes."""
    dt = datetime.utcfromtimestamp(ts_ms / 1000)
    et_min = (dt.hour - 5) * 60 + dt.minute  # UTC to EST
    if et_min < 0:
        et_min += 1440
    return et_min


def calculate_vwap(bars: list) -> float | None:
    """
    Calculate VWAP from bars.
    VWAP = Σ(TP × Volume) / Σ(Volume)
    TP (Typical Price) = (High + Low + Close) / 3
    """
    cum_tpv = 0
    cum_vol = 0
    for bar in bars:
        tp = (bar.get("h", 0) + bar.get("l", 0) + bar.get("c", 0)) / 3
        vol = bar.get("v", 0)
        cum_tpv += tp * vol
        cum_vol += vol
    return cum_tpv / cum_vol if cum_vol > 0 else None


@router.get("/price-compare")
async def compare_prices(symbol: str, date: str, time: str = "10:00"):
    """
    Compare price data from Alpaca and Polygon APIs.
    Calculates VWAP from 9:30 AM to user-specified time.
    """
    symbol = symbol.upper()
    result = {"symbol": symbol, "date": date, "time": time, "alpaca": None, "polygon": None}
    
    # Parse user requested time
    target_hour, target_min = map(int, time.split(":"))
    target_minutes_et = target_hour * 60 + target_min
    
    # =========================================================================
    # ALPACA DATA (via PPOALGO_API)
    # =========================================================================
    try:
        resp = requests.get(
            f"{PPOALGO_API}/api/prices",
            params={
                "symbol": symbol,
                "start": date,
                "end": date,
                "timeframe": "1Min"
            },
            timeout=60
        )
        
        if resp.status_code == 200:
            bars = resp.json()
            
            if bars:
                # Find key bars
                open_bar = None
                user_bar = None
                close_bar = None
                vwap_bars = []
                all_bars_sorted = []
                
                for bar in bars:
                    ts = bar.get("t", "")
                    et_min = get_et_minutes(ts)
                    if et_min is None:
                        continue
                    
                    bar["_et_min"] = et_min
                    all_bars_sorted.append(bar)
                    
                    # 9:30 AM = 570 minutes
                    if open_bar is None or abs(et_min - 570) < abs(open_bar.get("_et_min", 0) - 570):
                        if 565 <= et_min <= 575:
                            open_bar = bar
                    
                    # User's time
                    if user_bar is None or abs(et_min - target_minutes_et) < abs(user_bar.get("_et_min", 0) - target_minutes_et):
                        user_bar = bar
                    
                    # 4:00 PM = 960 minutes
                    if 570 <= et_min <= 960:
                        close_bar = bar
                    
                    # Collect bars from 9:30 to user time for VWAP
                    if 570 <= et_min <= target_minutes_et:
                        vwap_bars.append(bar)
                
                vwap_to_time = calculate_vwap(vwap_bars)
                
                # Daily VWAP
                daily_bars = [b for b in all_bars_sorted if 570 <= b.get("_et_min", 0) <= 960]
                vwap_daily = calculate_vwap(daily_bars)
                
                total_volume = sum(b.get("v", 0) for b in all_bars_sorted)
                volume_to_time = sum(b.get("v", 0) for b in vwap_bars)
                
                result["alpaca"] = {
                    "open_price": open_bar.get("o") if open_bar else None,
                    "open_time": "9:30 AM EST",
                    "user_price": user_bar.get("c") if user_bar else None,
                    "user_time": f"{target_hour}:{target_min:02d} EST" if user_bar else None,
                    "user_bar_timestamp": user_bar.get("t") if user_bar else None,
                    "close_price": close_bar.get("c") if close_bar else None,
                    "close_time": "4:00 PM EST",
                    "volume_daily": total_volume,
                    "volume_to_time": volume_to_time,
                    "vwap_to_time": round(vwap_to_time, 4) if vwap_to_time else None,
                    "vwap_daily": round(vwap_daily, 4) if vwap_daily else None,
                    "bars_used_for_vwap": len(vwap_bars)
                }
        else:
            result["error"] = f"Alpaca API returned status {resp.status_code}"
            
    except Exception as e:
        logger.error(f"Alpaca price compare error: {e}")
        result["alpaca_error"] = str(e)
    
    # =========================================================================
    # POLYGON DATA (direct API call)
    # =========================================================================
    try:
        polygon_key = POLYGON_API_KEY
        
        if polygon_key:
            polygon_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{date}/{date}"
            params = {"apiKey": polygon_key, "limit": 500, "sort": "asc"}
            
            poly_resp = requests.get(polygon_url, params=params, timeout=60)
            
            if poly_resp.status_code == 200:
                poly_data = poly_resp.json()
                poly_bars = poly_data.get("results", [])
                
                if poly_bars:
                    poly_open_bar = None
                    poly_user_bar = None
                    poly_close_bar = None
                    poly_vwap_bars = []
                    poly_all_bars = []
                    
                    for bar in poly_bars:
                        ts_ms = bar.get("t", 0)
                        et_min = get_poly_et_minutes(ts_ms)
                        bar["_et_min"] = et_min
                        poly_all_bars.append(bar)
                        
                        # 9:30 AM = 570 minutes
                        if poly_open_bar is None and 565 <= et_min <= 575:
                            poly_open_bar = bar
                        
                        # User's time
                        if poly_user_bar is None or abs(et_min - target_minutes_et) < abs(poly_user_bar.get("_et_min", 0) - target_minutes_et):
                            poly_user_bar = bar
                        
                        # Latest bar before 4 PM
                        if 570 <= et_min <= 960:
                            poly_close_bar = bar
                        
                        # Collect bars from 9:30 to user time
                        if 570 <= et_min <= target_minutes_et:
                            poly_vwap_bars.append(bar)
                    
                    poly_vwap_to_time = calculate_vwap(poly_vwap_bars)
                    
                    # Daily VWAP
                    poly_daily_bars = [b for b in poly_all_bars if 570 <= b.get("_et_min", 0) <= 960]
                    poly_vwap_daily = calculate_vwap(poly_daily_bars)
                    
                    poly_total_volume = sum(b.get("v", 0) for b in poly_all_bars)
                    poly_volume_to_time = sum(b.get("v", 0) for b in poly_vwap_bars)
                    
                    result["polygon"] = {
                        "open_price": poly_open_bar.get("o") if poly_open_bar else None,
                        "open_time": "9:30 AM EST",
                        "user_price": poly_user_bar.get("c") if poly_user_bar else None,
                        "user_time": f"{target_hour}:{target_min:02d} EST" if poly_user_bar else None,
                        "close_price": poly_close_bar.get("c") if poly_close_bar else None,
                        "close_time": "4:00 PM EST",
                        "volume_daily": poly_total_volume,
                        "volume_to_time": poly_volume_to_time,
                        "vwap_to_time": round(poly_vwap_to_time, 4) if poly_vwap_to_time else None,
                        "vwap_daily": round(poly_vwap_daily, 4) if poly_vwap_daily else None,
                        "bars_used_for_vwap": len(poly_vwap_bars),
                        "source": "Polygon.io"
                    }
            else:
                result["polygon_error"] = f"Polygon API returned status {poly_resp.status_code}"
                
    except Exception as e:
        logger.error(f"Polygon price compare error: {e}")
        result["polygon_error"] = str(e)
    
    return result
