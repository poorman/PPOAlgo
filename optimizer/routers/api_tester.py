#!/usr/bin/env python3
"""
API Tester Router
Provides /api/price-compare endpoint for comparing Alpaca, Widesurf, and Massive.com data.
"""

import os
import logging
import requests
from datetime import datetime, timedelta
from fastapi import APIRouter

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from config import PPOALGO_API, WIDESURF_API_KEY, WIDESURF_API_URL, MASSIVE_API_KEY, MASSIVE_API_URL

# Fallback for API credentials (not in all config versions)
ALPACA_API_KEY_ID = os.getenv("ALPACA_API_KEY_ID", "PKYAH5AZB64NRG2UB6QT3ISHPG")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
WIDESURF_API_KEY = os.getenv("WIDESURF_API_KEY", "69xn13ehEccqzDJxw29KH0mzzbIIgI2NRttD7m6p9gA")
WIDESURF_API_URL = os.getenv("WIDESURF_API_URL", "http://10.0.0.94:1020")




# US Market Holidays (fixed and approximate)
US_MARKET_HOLIDAYS = {
    # Fixed holidays
    (1, 1): "New Year's Day",
    (7, 4): "Independence Day",
    (12, 25): "Christmas Day",
    
    # MLK Day: 3rd Monday of January
    # Presidents Day: 3rd Monday of February
    # Good Friday: varies
    # Memorial Day: last Monday of May
    # Labor Day: 1st Monday of September
    # Thanksgiving: 4th Thursday of November
}

def get_nth_weekday(year, month, weekday, n):
    """Get the nth occurrence of a weekday in a month."""
    from calendar import monthcalendar
    weeks = monthcalendar(year, month)
    days = [week[weekday] for week in weeks if week[weekday] != 0]
    return days[n-1] if len(days) >= n else None

def get_last_weekday(year, month, weekday):
    """Get the last occurrence of a weekday in a month."""
    from calendar import monthcalendar
    weeks = monthcalendar(year, month)
    days = [week[weekday] for week in weeks if week[weekday] != 0]
    return days[-1] if days else None

def is_market_holiday(date_str: str) -> tuple[bool, str]:
    """Check if date is a US market holiday. Returns (is_holiday, holiday_name)."""
    try:
        from calendar import MONDAY, THURSDAY
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        
        # Check fixed holidays
        if (dt.month, dt.day) in US_MARKET_HOLIDAYS:
            return (True, US_MARKET_HOLIDAYS[(dt.month, dt.day)])
        
        # MLK Day: 3rd Monday of January
        if dt.month == 1:
            mlk_day = get_nth_weekday(dt.year, 1, MONDAY, 3)
            if dt.day == mlk_day:
                return (True, "Martin Luther King Jr. Day")
        
        # Presidents Day: 3rd Monday of February
        if dt.month == 2:
            pres_day = get_nth_weekday(dt.year, 2, MONDAY, 3)
            if dt.day == pres_day:
                return (True, "Presidents Day")
        
        # Memorial Day: Last Monday of May
        if dt.month == 5:
            mem_day = get_last_weekday(dt.year, 5, MONDAY)
            if dt.day == mem_day:
                return (True, "Memorial Day")
        
        # Labor Day: 1st Monday of September
        if dt.month == 9:
            labor_day = get_nth_weekday(dt.year, 9, MONDAY, 1)
            if dt.day == labor_day:
                return (True, "Labor Day")
        
        # Thanksgiving: 4th Thursday of November
        if dt.month == 11:
            thanksgiving = get_nth_weekday(dt.year, 11, THURSDAY, 4)
            if dt.day == thanksgiving:
                return (True, "Thanksgiving Day")
        
        return (False, "")
    except Exception:
        return (False, "")

def get_market_closed_reason(date_str: str) -> str:
    """Get the reason why market is closed for a date."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        weekday = dt.weekday()
        
        # Check weekend
        if weekday == 5:  # Saturday
            return "Saturday - Market Closed"
        elif weekday == 6:  # Sunday
            return "Sunday - Market Closed"
        
        # Check holiday
        is_holiday, holiday_name = is_market_holiday(date_str)
        if is_holiday:
            return f"{holiday_name} - Market Closed"
        
        return "Market Closed"
    except Exception:
        return "Market Closed"

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["API Tester"])

# Timezone constants
ET_ZONE = ZoneInfo("America/New_York")
UTC_ZONE = ZoneInfo("UTC")


def is_weekend(date_str: str) -> bool:
    """Check if date is a weekend (Saturday or Sunday)."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.weekday() >= 5  # Saturday=5, Sunday=6
    except Exception:
        return False


def get_previous_trading_day(date_str: str) -> str:
    """Get the previous trading day (skip weekends)."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        # If Saturday, go back 1 day to Friday
        # If Sunday, go back 2 days to Friday
        if dt.weekday() == 5:  # Saturday
            dt = dt - timedelta(days=1)
        elif dt.weekday() == 6:  # Sunday
            dt = dt - timedelta(days=2)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return date_str


def mask_api_key(api_key: str) -> str:
    """Mask API key, showing only last 6 characters."""
    if not api_key or len(api_key) <= 6:
        return "***"
    return "***" + api_key[-6:]


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
    
    # Check if market is closed (weekend or holiday)
    is_weekend_date = is_weekend(date)
    is_holiday, holiday_name = is_market_holiday(date)
    is_market_closed = is_weekend_date or is_holiday
    
    nearest_trading_day = get_previous_trading_day(date) if is_market_closed else None
    
    result = {
        "symbol": symbol,
        "date": date,
        "time": time,
        "is_weekend": is_weekend_date,
        "is_holiday": is_holiday,
        "market_status": "closed" if is_market_closed else "likely_open",
        "nearest_trading_day": nearest_trading_day,
        "alpaca": None,
        "widesurf": None,
        "massive": None,
        "api_sources": {
            "alpaca": {
                "endpoint": ALPACA_BASE_URL,
                "key_masked": mask_api_key(ALPACA_API_KEY_ID)
            },
            "widesurf": {
                "endpoint": WIDESURF_API_URL,
                "key_masked": mask_api_key(WIDESURF_API_KEY)
            },
            "massive": {
                "endpoint": MASSIVE_API_URL,
                "key_masked": mask_api_key(MASSIVE_API_KEY)
            }
        }
    }
    
    # Add warning if closed
    if is_market_closed:
        result["warning"] = f"{get_market_closed_reason(date)}. No trading data available. Try {nearest_trading_day} instead."
        return result  # STRICT BLOCK: Return early if market is closed
    
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
    # WIDESURF DATA (direct API call - uses apiKey query param)
    # =========================================================================
    try:
        widesurf_url = f"{WIDESURF_API_URL}/v1/aggs/ticker/{symbol}/range/1/minute/{date}/{date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": WIDESURF_API_KEY}
        
        ws_resp = requests.get(widesurf_url, params=params, timeout=60)
        
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
