#!/usr/bin/env python3
"""
API Tester Router
Provides /api/price-compare endpoint for comparing Alpaca, Widesurf, and Massive.com data.
"""

import os
import logging
import requests
from datetime import datetime
from fastapi import APIRouter

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from config import PPOALGO_API, WIDESURF_API_KEY, WIDESURF_API_URL, MASSIVE_API_KEY, MASSIVE_API_URL

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["API Tester"])

# Timezone constants
ET_ZONE = ZoneInfo("America/New_York")
UTC_ZONE = ZoneInfo("UTC")


def get_et_minutes(ts: str) -> int | None:
    """
    Convert UTC timestamp to Eastern Time minutes since midnight.
    Properly handles DST by using America/New_York timezone.
    """
    if not ts or "T" not in ts:
        return None
    
    try:
        # Parse the timestamp - handle various formats
        ts_clean = ts.replace("Z", "+00:00")
        
        # If no timezone info, assume UTC
        if "+" not in ts_clean and ts_clean.count("-") == 2:
            ts_clean = ts_clean + "+00:00"
        
        # Parse with fromisoformat (handles timezone)
        dt_utc = datetime.fromisoformat(ts_clean)
        
        # If no timezone, assume UTC
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=UTC_ZONE)
        
        # Convert to Eastern Time (handles DST automatically)
        dt_et = dt_utc.astimezone(ET_ZONE)
        
        # Return minutes since midnight
        return dt_et.hour * 60 + dt_et.minute
    except Exception:
        # Fallback to manual parsing (legacy behavior)
        try:
            hour_str = ts.split("T")[1][:5]
            h, m = map(int, hour_str.split(":"))
            et_min = (h - 5) * 60 + m  # Assume EST (may be off by 1 hour during DST)
            if et_min < 0:
                et_min += 1440
            return et_min
        except Exception:
            return None


def get_ms_et_minutes(ts_ms: int) -> int:
    """
    Convert millisecond timestamp (used by Widesurf and Massive) to Eastern Time minutes.
    Properly handles DST by using America/New_York timezone.
    """
    try:
        # Create UTC datetime from milliseconds timestamp
        dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=UTC_ZONE)
        
        # Convert to Eastern Time (handles DST automatically)
        dt_et = dt_utc.astimezone(ET_ZONE)
        
        return dt_et.hour * 60 + dt_et.minute
    except Exception:
        # Fallback to legacy behavior
        dt = datetime.utcfromtimestamp(ts_ms / 1000)
        et_min = (dt.hour - 5) * 60 + dt.minute  # Assume EST
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


def process_bars_for_comparison(bars: list, target_minutes_et: int, target_hour: int, target_min: int, use_ms_timestamp: bool = False):
    """
    Common logic to process bars and extract key prices/VWAP.
    Returns a dict with open_price, user_price, close_price, volumes, VWAP etc.
    """
    open_bar = None
    user_bar = None
    close_bar = None
    vwap_bars = []
    all_bars = []
    
    for bar in bars:
        if use_ms_timestamp:
            ts_ms = bar.get("t", 0)
            et_min = get_ms_et_minutes(ts_ms)
        else:
            ts = bar.get("t", "")
            et_min = get_et_minutes(ts)
            if et_min is None:
                continue
        
        bar["_et_min"] = et_min
        all_bars.append(bar)
        
        # 9:30 AM = 570 minutes
        if open_bar is None and 565 <= et_min <= 575:
            open_bar = bar
        
        # User's time - find closest bar
        if user_bar is None or abs(et_min - target_minutes_et) < abs(user_bar.get("_et_min", 0) - target_minutes_et):
            user_bar = bar
        
        # 4:00 PM = 960 minutes - latest bar in trading hours
        if 570 <= et_min <= 960:
            close_bar = bar
        
        # Collect bars from 9:30 to user time for VWAP
        if 570 <= et_min <= target_minutes_et:
            vwap_bars.append(bar)
    
    vwap_to_time = calculate_vwap(vwap_bars)
    
    # Daily VWAP
    daily_bars = [b for b in all_bars if 570 <= b.get("_et_min", 0) <= 960]
    vwap_daily = calculate_vwap(daily_bars)
    
    total_volume = sum(b.get("v", 0) for b in all_bars)
    volume_to_time = sum(b.get("v", 0) for b in vwap_bars)
    
    return {
        "open_price": open_bar.get("o") if open_bar else None,
        "open_time": "9:30 AM EST",
        "user_price": user_bar.get("c") if user_bar else None,
        "user_time": f"{target_hour}:{target_min:02d} EST" if user_bar else None,
        "close_price": close_bar.get("c") if close_bar else None,
        "close_time": "4:00 PM EST",
        "volume_daily": total_volume,
        "volume_to_time": volume_to_time,
        "vwap_to_time": round(vwap_to_time, 4) if vwap_to_time else None,
        "vwap_daily": round(vwap_daily, 4) if vwap_daily else None,
        "bars_used_for_vwap": len(vwap_bars)
    }


@router.get("/price-compare")
async def compare_prices(symbol: str, date: str, time: str = "10:00"):
    """
    Compare price data from Alpaca, Widesurf, and Massive.com APIs.
    Calculates VWAP from 9:30 AM to user-specified time.
    """
    symbol = symbol.upper()
    result = {"symbol": symbol, "date": date, "time": time, "alpaca": None, "widesurf": None, "massive": None}
    
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
                result["alpaca"] = process_bars_for_comparison(
                    bars, target_minutes_et, target_hour, target_min, use_ms_timestamp=False
                )
                result["alpaca"]["source"] = "Alpaca API"
        else:
            result["alpaca_error"] = f"Alpaca API returned status {resp.status_code}"
            
    except Exception as e:
        logger.error(f"Alpaca price compare error: {e}")
        result["alpaca_error"] = str(e)
    
    # =========================================================================
    # WIDESURF DATA (direct API call - uses X-API-KEY header)
    # =========================================================================
    try:
        widesurf_url = f"{WIDESURF_API_URL}/v1/aggs/ticker/{symbol}/range/1/minute/{date}/{date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        headers = {"X-API-KEY": WIDESURF_API_KEY}
        
        ws_resp = requests.get(widesurf_url, params=params, headers=headers, timeout=60)
        
        if ws_resp.status_code == 200:
            ws_data = ws_resp.json()
            ws_bars = ws_data.get("results", [])
            
            if ws_bars:
                result["widesurf"] = process_bars_for_comparison(
                    ws_bars, target_minutes_et, target_hour, target_min, use_ms_timestamp=True
                )
                result["widesurf"]["source"] = "Widesurf API"
        else:
            result["widesurf_error"] = f"Widesurf API returned status {ws_resp.status_code}"
                
    except Exception as e:
        logger.error(f"Widesurf price compare error: {e}")
        result["widesurf_error"] = str(e)
    
    # =========================================================================
    # MASSIVE.COM DATA (direct API call)
    # =========================================================================
    try:
        # Massive.com uses same format as Polygon
        massive_url = f"{MASSIVE_API_URL}/v2/aggs/ticker/{symbol}/range/1/minute/{date}/{date}"
        params = {"apiKey": MASSIVE_API_KEY, "limit": 50000, "sort": "asc", "adjusted": "true"}
        
        mas_resp = requests.get(massive_url, params=params, timeout=60)
        
        if mas_resp.status_code == 200:
            mas_data = mas_resp.json()
            mas_bars = mas_data.get("results", [])
            
            if mas_bars:
                result["massive"] = process_bars_for_comparison(
                    mas_bars, target_minutes_et, target_hour, target_min, use_ms_timestamp=True
                )
                result["massive"]["source"] = "Massive.com API"
        else:
            result["massive_error"] = f"Massive.com API returned status {mas_resp.status_code}"
                
    except Exception as e:
        logger.error(f"Massive.com price compare error: {e}")
        result["massive_error"] = str(e)
    
    return result
