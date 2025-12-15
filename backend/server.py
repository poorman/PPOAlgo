import os
from datetime import datetime
import psycopg2
from psycopg2.extras import Json
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx


DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "ppoalgo"),
    "user": os.getenv("POSTGRES_USER", "ppoalgo"),
    "password": os.getenv("POSTGRES_PASSWORD", "ppoalgo"),
    "host": os.getenv("POSTGRES_HOST", "db"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}

ALPACA_KEY = os.getenv("ALPACA_API_KEY_ID", "")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET_KEY", "")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_DATA_URL = os.getenv("POLYGON_DATA_URL", "https://api.polygon.io")


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def ensure_table():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS gui_events (
                    id SERIAL PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    payload JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )
        conn.commit()


class Event(BaseModel):
    event_type: str
    payload: dict


app = FastAPI(title="PPOAlgo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    ensure_table()


@app.get("/health")
def health():
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        return {"status": "ok"}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/event")
def ingest_event(event: Event):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO gui_events (event_type, payload) VALUES (%s, %s) RETURNING id;",
                    (event.event_type, Json(event.payload)),
                )
                new_id = cur.fetchone()[0]
            conn.commit()
        return {"status": "ok", "id": new_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/prices")
def get_prices(
    symbol: str = Query(..., min_length=1),
    start: str = Query(..., description="ISO date or datetime"),
    end: str = Query(..., description="ISO date or datetime"),
    timeframe: str = Query("1Day"),
    limit: int = Query(1000, ge=1, le=10000),
):
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid start/end datetime format")

    # Auto-correct if start is after end by swapping to keep ascending range.
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    # Prefer Polygon 1-minute aggregates when key is present.
    if POLYGON_KEY:
        try:
            days = (end_dt.date() - start_dt.date()).days + 1
            approx_minutes = days * 390  # rough trading minutes/day
            use_day = timeframe.lower().startswith("1day") or approx_minutes > 45000
            timespan = "day" if use_day else "minute"

            start_date = start_dt.date().isoformat()
            end_date = end_dt.date().isoformat()
            url = f"{POLYGON_DATA_URL}/v2/aggs/ticker/{symbol.upper()}/range/1/{timespan}/{start_date}/{end_date}"
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apiKey": POLYGON_KEY,
            }
            with httpx.Client(timeout=20.0) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        results = data.get("results", [])
        bars = []
        for r in results:
            t_ms = r.get("t")
            if t_ms is None:
                continue
            t_iso = datetime.utcfromtimestamp(t_ms / 1000).isoformat() + "Z"
            bars.append(
                {
                    "t": t_iso,
                    "o": r.get("o"),
                    "h": r.get("h"),
                    "l": r.get("l"),
                    "c": r.get("c"),
                    "v": r.get("v"),
                }
            )
        return bars

    if ALPACA_KEY and ALPACA_SECRET:
        params = {
            "start": start_dt.isoformat() + "Z",
            "end": end_dt.isoformat() + "Z",
            "timeframe": timeframe,
            "limit": limit,
            "adjustment": "all",
        }

        url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
        headers = {
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET,
        }
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        bars = data.get("bars", [])
        return [
            {
              "t": bar.get("t"),
              "o": bar.get("o"),
              "h": bar.get("h"),
              "l": bar.get("l"),
              "c": bar.get("c"),
              "v": bar.get("v"),
            }
            for bar in bars
        ]

    raise HTTPException(status_code=400, detail="No market data provider configured (Polygon or Alpaca)")


# Backtest request model
class BacktestRequest(BaseModel):
    symbol: str
    start: str  # ISO date format
    end: str    # ISO date format
    capital: float = 100000
    buy_trigger_pct: float = 5.0
    buy_time: str = "09:00"
    sell_trigger_pct: float = 9.0
    buy_amount: float = 1000
    compound: bool = False
    fees_bps: float = 5.0
    slippage_pct: float = 0.05


def run_momentum_backtest(bars: list, config: BacktestRequest) -> dict:
    """Run momentum backtest on price bars and return trades + metrics."""
    if not bars:
        return {"trades": [], "metrics": {}}
    
    buy_pct = config.buy_trigger_pct / 100
    sell_pct = config.sell_trigger_pct / 100
    buy_amount = config.buy_amount
    compound = config.compound
    capital = config.capital
    
    trades = []
    in_pos = False
    entry_price = None
    qty = 0
    cash = capital
    compounded_amount = buy_amount
    
    portfolio = []
    peak = capital
    max_dd = 0
    
    for i, bar in enumerate(bars):
        open_price = bar.get("o") or bar.get("c")
        high = bar.get("h") or bar.get("c")
        close = bar.get("c")
        timestamp = bar.get("t", "")
        
        # Get previous close
        prev_close = bars[i - 1].get("c") if i > 0 else open_price
        
        # Calculate % change from previous close to today's high
        pct_change_from_prev = (high - prev_close) / prev_close if prev_close else 0
        
        # Buy logic
        if not in_pos and pct_change_from_prev >= buy_pct:
            entry_price = prev_close * (1 + buy_pct)
            
            # Determine invest amount
            if buy_amount == 0:
                invest_amount = cash
            elif compound:
                invest_amount = min(compounded_amount, cash)
            else:
                invest_amount = buy_amount
            
            qty = invest_amount / entry_price
            notional = qty * entry_price
            cash -= notional
            in_pos = True
            
            trades.append({
                "timestamp": timestamp,
                "action": "buy (C)" if compound else "buy",
                "price": round(entry_price, 4),
                "pct_change": round(buy_pct * 100, 2),
                "notional": round(notional, 2),
                "wallet": round(cash + notional, 2),
                "position": 100,
                "realized": 0
            })
        
        # Sell logic (same day)
        if in_pos:
            target_price = entry_price * (1 + sell_pct)
            hit_target = high >= target_price
            
            if hit_target:
                exit_price = target_price
                sell_time = "10:30"
            else:
                exit_price = close
                sell_time = "15:59"
            
            notional = qty * exit_price
            realized = (exit_price - entry_price) * qty
            cash += notional
            
            # Update compounded amount
            if compound:
                compounded_amount = notional
            
            trades.append({
                "timestamp": timestamp,
                "action": "sell" if hit_target else "sell (EOD)",
                "price": round(exit_price, 4),
                "pct_change": round(((exit_price - entry_price) / entry_price) * 100, 2),
                "notional": round(notional, 2),
                "wallet": round(cash, 2),
                "position": 0,
                "realized": round(realized, 2)
            })
            
            in_pos = False
            entry_price = None
            qty = 0
        
        # Track portfolio and drawdown
        nav = cash + (qty * close if in_pos else 0)
        portfolio.append(nav)
        peak = max(peak, nav)
        dd = ((nav - peak) / peak) * 100 if peak else 0
        max_dd = min(max_dd, dd)
    
    # Calculate metrics
    sells = [t for t in trades if t["action"].startswith("sell")]
    round_trip_pnls = [t["realized"] for t in sells]
    
    wins = [p for p in round_trip_pnls if p > 0]
    losses = [p for p in round_trip_pnls if p < 0]
    
    total_pnl = sum(round_trip_pnls)
    total_return = (total_pnl / capital) * 100 if capital else 0
    win_rate = (len(wins) / len(round_trip_pnls)) * 100 if round_trip_pnls else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0
    
    # Estimate commission and slippage
    total_notional = sum(t.get("notional", 0) for t in trades)
    commission = (total_notional * config.fees_bps) / 10000
    slippage = (total_notional * config.slippage_pct) / 100
    
    # Calculate Sharpe (simplified)
    daily_returns = []
    for i in range(1, len(portfolio)):
        if portfolio[i - 1] != 0:
            daily_returns.append((portfolio[i] - portfolio[i - 1]) / portfolio[i - 1])
    
    if daily_returns:
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
        std_dev = variance ** 0.5
        sharpe = (mean_return / std_dev) * (252 ** 0.5) if std_dev else 0
    else:
        sharpe = 0
    
    metrics = {
        "total_return": round(total_return, 2),
        "sharpe": round(max(0, sharpe), 2),
        "max_drawdown": round(max_dd, 2),
        "win_rate": round(win_rate, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "commission": round(commission, 2),
        "slippage": round(slippage, 2),
        "total_trades": len(sells),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "final_wallet": round(cash, 2)
    }
    
    return {"trades": trades, "metrics": metrics}


@app.post("/api/backtest")
def run_backtest(request: BacktestRequest):
    """
    Run a momentum backtest with the given parameters.
    
    Example request:
    {
        "symbol": "TSLA",
        "start": "2025-01-11",
        "end": "2025-12-11",
        "capital": 100000,
        "buy_trigger_pct": 1,
        "buy_time": "09:00",
        "sell_trigger_pct": 4,
        "buy_amount": 100000,
        "compound": false,
        "fees_bps": 5,
        "slippage_pct": 0.05
    }
    """
    # Fetch price data
    try:
        bars = get_prices(
            symbol=request.symbol,
            start=request.start,
            end=request.end,
            timeframe="1Day"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch prices: {str(e)}")
    
    # Run backtest
    result = run_momentum_backtest(bars, request)
    
    # Log event
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO gui_events (event_type, payload) VALUES (%s, %s);",
                    ("api_backtest", Json({
                        "symbol": request.symbol,
                        "config": request.dict(),
                        "metrics": result["metrics"]
                    })),
                )
            conn.commit()
    except Exception:
        pass  # Don't fail if logging fails
    
    return result


# ============================================================================
# PAPER TRADING API ENDPOINTS
# ============================================================================

ALPACA_PAPER_URL = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")

# In-memory session storage (for production, use Redis or DB)
paper_sessions = {}


def ensure_paper_settings_table():
    """Create paper trading settings table if it doesn't exist."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS paper_trading_settings (
                        id SERIAL PRIMARY KEY,
                        key_id TEXT,
                        secret_key TEXT,
                        algorithm TEXT DEFAULT 'momentum',
                        symbol TEXT DEFAULT 'TSLA',
                        buy_trigger_pct REAL DEFAULT 2.5,
                        sell_trigger_pct REAL DEFAULT 5.0,
                        buy_amount REAL DEFAULT 1000,
                        compound BOOLEAN DEFAULT FALSE,
                        entry_time TEXT DEFAULT '09:35',
                        exit_time TEXT DEFAULT '15:55',
                        duration_days INTEGER DEFAULT 5,
                        max_daily_loss REAL DEFAULT 500,
                        max_position_size REAL DEFAULT 5000,
                        stop_on_loss BOOLEAN DEFAULT TRUE,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
                # Also create paper_trading_sessions table for persistence
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS paper_trading_sessions (
                        id TEXT PRIMARY KEY,
                        status TEXT DEFAULT 'running',
                        algorithm TEXT,
                        symbol TEXT,
                        config JSONB,
                        account_id TEXT,
                        buying_power REAL,
                        key_id TEXT,
                        secret_key TEXT,
                        started_at TIMESTAMPTZ DEFAULT NOW(),
                        trades INTEGER DEFAULT 0,
                        pnl REAL DEFAULT 0,
                        position TEXT
                    );
                    """
                )
                # Create trades history table
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS paper_trades (
                        id SERIAL PRIMARY KEY,
                        session_id TEXT REFERENCES paper_trading_sessions(id) ON DELETE CASCADE,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        price REAL NOT NULL,
                        notional REAL,
                        pnl REAL DEFAULT 0,
                        executed_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
            conn.commit()
    except Exception as e:
        print(f"Error creating paper trading tables: {e}")


def load_sessions_from_db():
    """Load paper trading sessions from database on startup."""
    global paper_sessions
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, status, algorithm, symbol, config, account_id, 
                           buying_power, key_id, secret_key, started_at, trades, pnl, position
                    FROM paper_trading_sessions
                    WHERE status = 'running'
                    """
                )
                rows = cur.fetchall()
                for row in rows:
                    session = {
                        "id": row[0],
                        "status": row[1],
                        "algorithm": row[2],
                        "symbol": row[3],
                        "config": row[4] or {},
                        "account_id": row[5],
                        "buying_power": float(row[6] or 0),
                        "key_id": row[7],
                        "secret_key": row[8],
                        "started_at": row[9].isoformat() if row[9] else None,
                        "trades": row[10] or 0,
                        "pnl": float(row[11] or 0),
                        "position": row[12],
                    }
                    paper_sessions[row[0]] = session
                print(f"Loaded {len(rows)} paper trading sessions from database")
    except Exception as e:
        print(f"Error loading sessions from DB: {e}")


def save_session_to_db(session: dict):
    """Save a paper trading session to database."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO paper_trading_sessions 
                    (id, status, algorithm, symbol, config, account_id, buying_power, 
                     key_id, secret_key, started_at, trades, pnl, position)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        status = EXCLUDED.status,
                        trades = EXCLUDED.trades,
                        pnl = EXCLUDED.pnl,
                        position = EXCLUDED.position
                    """,
                    (
                        session["id"],
                        session.get("status", "running"),
                        session.get("algorithm"),
                        session.get("symbol"),
                        Json(session.get("config", {})),
                        session.get("account_id"),
                        session.get("buying_power"),
                        session.get("key_id"),
                        session.get("secret_key"),
                        session.get("started_at"),
                        session.get("trades", 0),
                        session.get("pnl", 0),
                        session.get("position"),
                    )
                )
            conn.commit()
    except Exception as e:
        print(f"Error saving session to DB: {e}")


def delete_session_from_db(session_id: str):
    """Delete a paper trading session from database."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM paper_trading_sessions WHERE id = %s",
                    (session_id,)
                )
            conn.commit()
    except Exception as e:
        print(f"Error deleting session from DB: {e}")


# Create the tables on module load
ensure_paper_settings_table()
# Load existing sessions
load_sessions_from_db()


class AlpacaTestRequest(BaseModel):
    key_id: str
    secret_key: str


class PaperStartRequest(BaseModel):
    key_id: str
    secret_key: str
    algorithm: str = "momentum"
    symbol: str = "TSLA"
    buy_trigger_pct: float = 2.5
    sell_trigger_pct: float = 5.0
    buy_amount: float = 1000
    compound: bool = False
    entry_time: str = "09:35"
    exit_time: str = "15:55"
    duration_days: int = 5
    max_daily_loss: float = 500
    max_position_size: float = 5000
    stop_on_loss: bool = True


class PaperStopRequest(BaseModel):
    session_id: str = None


class PaperSettingsRequest(BaseModel):
    key_id: str = ""
    secret_key: str = ""
    algorithm: str = "momentum"
    symbol: str = "TSLA"
    buy_trigger_pct: float = 2.5
    sell_trigger_pct: float = 5.0
    buy_amount: float = 1000
    compound: bool = False
    entry_time: str = "09:35"
    exit_time: str = "15:55"
    duration_days: int = 5
    max_daily_loss: float = 500
    max_position_size: float = 5000
    stop_on_loss: bool = True


@app.post("/api/alpaca/test")
def test_alpaca_connection(request: AlpacaTestRequest):
    """Test connection to Alpaca paper trading API."""
    if not request.key_id or not request.secret_key:
        raise HTTPException(status_code=400, detail="Missing API credentials")
    
    try:
        headers = {
            "APCA-API-KEY-ID": request.key_id,
            "APCA-API-SECRET-KEY": request.secret_key,
        }
        
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                f"{ALPACA_PAPER_URL}/v2/account",
                headers=headers
            )
            
            if resp.status_code == 200:
                account = resp.json()
                return {
                    "status": "connected",
                    "account_number": account.get("account_number"),
                    "id": account.get("id"),
                    "buying_power": account.get("buying_power"),
                    "cash": account.get("cash"),
                    "equity": account.get("equity"),
                    "portfolio_value": account.get("portfolio_value"),
                }
            elif resp.status_code == 401:
                raise HTTPException(status_code=401, detail="Invalid API credentials")
            elif resp.status_code == 403:
                raise HTTPException(status_code=403, detail="API access forbidden - check API permissions")
            else:
                raise HTTPException(status_code=resp.status_code, detail=f"Alpaca API error: {resp.text}")
                
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection error: {str(e)}")


@app.post("/api/paper/start")
def start_paper_trading(request: PaperStartRequest):
    """Start a paper trading session."""
    import uuid
    
    # Validate credentials first
    try:
        headers = {
            "APCA-API-KEY-ID": request.key_id,
            "APCA-API-SECRET-KEY": request.secret_key,
        }
        
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                f"{ALPACA_PAPER_URL}/v2/account",
                headers=headers
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid Alpaca credentials")
            account = resp.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify credentials: {str(e)}")
    
    # Save settings to database
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Delete old settings and insert new
                cur.execute("DELETE FROM paper_trading_settings;")
                cur.execute(
                    """
                    INSERT INTO paper_trading_settings 
                    (key_id, secret_key, algorithm, symbol, buy_trigger_pct, sell_trigger_pct,
                     buy_amount, compound, entry_time, exit_time, duration_days,
                     max_daily_loss, max_position_size, stop_on_loss)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (request.key_id, request.secret_key, request.algorithm, request.symbol,
                     request.buy_trigger_pct, request.sell_trigger_pct, request.buy_amount,
                     request.compound, request.entry_time, request.exit_time, request.duration_days,
                     request.max_daily_loss, request.max_position_size, request.stop_on_loss)
                )
            conn.commit()
    except Exception as e:
        print(f"Error saving settings: {e}")
    
    # Create session
    session_id = str(uuid.uuid4())[:8]
    
    session = {
        "id": session_id,
        "status": "running",
        "algorithm": request.algorithm,
        "symbol": request.symbol,
        "config": {
            "buy_trigger_pct": request.buy_trigger_pct,
            "sell_trigger_pct": request.sell_trigger_pct,
            "buy_amount": request.buy_amount,
            "compound": request.compound,
            "entry_time": request.entry_time,
            "exit_time": request.exit_time,
            "duration_days": request.duration_days,
            "max_daily_loss": request.max_daily_loss,
            "max_position_size": request.max_position_size,
            "stop_on_loss": request.stop_on_loss,
        },
        "account_id": account.get("id"),
        "buying_power": float(account.get("buying_power", 0)),
        "started_at": datetime.now().isoformat(),
        "trades": 0,
        "pnl": 0.0,
        "position": None,
    }
    
    paper_sessions[session_id] = session
    
    # Persist to database
    session["key_id"] = request.key_id
    session["secret_key"] = request.secret_key
    save_session_to_db(session)
    
    # Log to database
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO gui_events (event_type, payload) VALUES (%s, %s);",
                    ("paper_trading_started", Json({
                        "session_id": session_id,
                        "algorithm": request.algorithm,
                        "symbol": request.symbol,
                        "config": session["config"],
                    })),
                )
            conn.commit()
    except Exception:
        pass  # Don't fail if logging fails
    
    return {
        "status": "started",
        "session_id": session_id,
        "buying_power": session["buying_power"],
        "message": f"Paper trading started for {request.symbol} using {request.algorithm} strategy",
    }


@app.post("/api/paper/stop")
def stop_paper_trading(request: PaperStopRequest):
    """Stop a paper trading session."""
    session_id = request.session_id
    
    if session_id and session_id in paper_sessions:
        session = paper_sessions[session_id]
        session["status"] = "stopped"
        final_pnl = session.get("pnl", 0)
        
        # Log to database
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO gui_events (event_type, payload) VALUES (%s, %s);",
                        ("paper_trading_stopped", Json({
                            "session_id": session_id,
                            "final_pnl": final_pnl,
                            "total_trades": session.get("trades", 0),
                        })),
                    )
                conn.commit()
        except Exception:
            pass
        
        del paper_sessions[session_id]
        delete_session_from_db(session_id)
        
        return {
            "status": "stopped",
            "session_id": session_id,
            "final_pnl": final_pnl,
            "message": "Paper trading session stopped",
        }
    else:
        # Stop any active session
        for sid, session in list(paper_sessions.items()):
            session["status"] = "stopped"
            delete_session_from_db(sid)
            del paper_sessions[sid]
        
        return {
            "status": "stopped",
            "message": "All paper trading sessions stopped",
        }


@app.get("/api/paper/status")
def get_paper_status():
    """Get status of active paper trading sessions."""
    if not paper_sessions:
        return {
            "status": "idle",
            "pnl": 0,
            "position": "None",
            "trades": 0,
        }
    
    # Return status of most recent session
    session = list(paper_sessions.values())[-1]
    
    return {
        "status": session.get("status", "running"),
        "session_id": session.get("id"),
        "symbol": session.get("symbol"),
        "algorithm": session.get("algorithm"),
        "pnl": session.get("pnl", 0),
        "position": f"{session.get('symbol')} (simulated)" if session.get("position") else "None",
        "trades": session.get("trades", 0),
        "started_at": session.get("started_at"),
        "buying_power": session.get("buying_power"),
    }


@app.get("/api/paper/sessions")
def list_paper_sessions():
    """List all active paper trading sessions."""
    return {
        "sessions": list(paper_sessions.values()),
        "count": len(paper_sessions),
    }


@app.get("/api/paper/settings")
def get_paper_settings():
    """Get saved paper trading settings from database."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT key_id, secret_key, algorithm, symbol, buy_trigger_pct, 
                           sell_trigger_pct, buy_amount, compound, entry_time, 
                           exit_time, duration_days, max_daily_loss, max_position_size, 
                           stop_on_loss
                    FROM paper_trading_settings
                    ORDER BY updated_at DESC
                    LIMIT 1;
                    """
                )
                row = cur.fetchone()
                
                if row:
                    return {
                        "key_id": row[0] or "",
                        "secret_key": row[1] or "",
                        "algorithm": row[2] or "momentum",
                        "symbol": row[3] or "TSLA",
                        "buy_trigger_pct": row[4] or 2.5,
                        "sell_trigger_pct": row[5] or 5.0,
                        "buy_amount": row[6] or 1000,
                        "compound": row[7] or False,
                        "entry_time": row[8] or "09:35",
                        "exit_time": row[9] or "15:55",
                        "duration_days": row[10] or 5,
                        "max_daily_loss": row[11] or 500,
                        "max_position_size": row[12] or 5000,
                        "stop_on_loss": row[13] if row[13] is not None else True,
                    }
                else:
                    return {
                        "key_id": "",
                        "secret_key": "",
                        "algorithm": "momentum",
                        "symbol": "TSLA",
                        "buy_trigger_pct": 2.5,
                        "sell_trigger_pct": 5.0,
                        "buy_amount": 1000,
                        "compound": False,
                        "entry_time": "09:35",
                        "exit_time": "15:55",
                        "duration_days": 5,
                        "max_daily_loss": 500,
                        "max_position_size": 5000,
                        "stop_on_loss": True,
                    }
    except Exception as e:
        print(f"Error loading settings: {e}")
        return {"error": str(e)}


@app.post("/api/paper/settings")
def save_paper_settings(request: PaperSettingsRequest):
    """Save paper trading settings to database."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Delete old settings and insert new
                cur.execute("DELETE FROM paper_trading_settings;")
                cur.execute(
                    """
                    INSERT INTO paper_trading_settings 
                    (key_id, secret_key, algorithm, symbol, buy_trigger_pct, sell_trigger_pct,
                     buy_amount, compound, entry_time, exit_time, duration_days,
                     max_daily_loss, max_position_size, stop_on_loss)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (request.key_id, request.secret_key, request.algorithm, request.symbol,
                     request.buy_trigger_pct, request.sell_trigger_pct, request.buy_amount,
                     request.compound, request.entry_time, request.exit_time, request.duration_days,
                     request.max_daily_loss, request.max_position_size, request.stop_on_loss)
                )
            conn.commit()
        return {"status": "saved", "message": "Settings saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")


@app.get("/api/paper/portfolio")
def get_portfolio_summary():
    """Get aggregated portfolio summary across all sessions."""
    sessions = list(paper_sessions.values())
    
    if not sessions:
        return {
            "total_pnl": 0,
            "total_capital": 0,
            "active_positions": 0,
            "best_performer": None,
            "worst_performer": None,
            "session_count": 0,
            "market_status": get_market_status(),
        }
    
    total_pnl = sum(s.get("pnl", 0) for s in sessions)
    total_capital = sum(s.get("config", {}).get("buy_amount", 0) for s in sessions)
    active_positions = sum(1 for s in sessions if s.get("position"))
    
    # Find best/worst performers
    sorted_by_pnl = sorted(sessions, key=lambda s: s.get("pnl", 0), reverse=True)
    best = sorted_by_pnl[0] if sorted_by_pnl else None
    worst = sorted_by_pnl[-1] if sorted_by_pnl else None
    
    return {
        "total_pnl": total_pnl,
        "total_capital": total_capital,
        "active_positions": active_positions,
        "best_performer": {
            "symbol": best.get("symbol"),
            "pnl": best.get("pnl", 0),
        } if best else None,
        "worst_performer": {
            "symbol": worst.get("symbol"),
            "pnl": worst.get("pnl", 0),
        } if worst else None,
        "session_count": len(sessions),
        "market_status": get_market_status(),
    }


def get_market_status():
    """Get current market status based on US Eastern time."""
    from datetime import datetime, timezone, timedelta
    
    try:
        # Use simple UTC offset for Eastern time (-5 hours, or -4 during DST)
        # This is a simplification - proper DST handling would need zoneinfo
        utc_now = datetime.now(timezone.utc)
        # Eastern is UTC-5 (or UTC-4 during DST), we'll approximate with -5
        eastern_offset = timedelta(hours=-5)
        now = utc_now + eastern_offset
        
        # Market hours: 9:30 AM - 4:00 PM ET, Mon-Fri
        weekday = now.weekday()
        hour = now.hour
        minute = now.minute
        
        if weekday >= 5:  # Saturday or Sunday
            return {"status": "closed", "label": "Weekend"}
        
        market_open = (hour == 9 and minute >= 30) or (hour > 9 and hour < 16)
        pre_market = hour >= 4 and (hour < 9 or (hour == 9 and minute < 30))
        after_hours = hour >= 16 and hour < 20
        
        if market_open:
            close_minutes = (16 * 60) - (hour * 60 + minute)
            hours_left = close_minutes // 60
            mins_left = close_minutes % 60
            return {"status": "open", "label": f"Open ({hours_left}h {mins_left}m left)"}
        elif pre_market:
            return {"status": "pre-market", "label": "Pre-Market"}
        elif after_hours:
            return {"status": "after-hours", "label": "After Hours"}
        else:
            return {"status": "closed", "label": "Closed"}
    except Exception:
        return {"status": "unknown", "label": "Unknown"}


@app.get("/api/paper/trades")
def get_trades_history(session_id: str = None, limit: int = 100):
    """Get trade history from database."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                if session_id:
                    cur.execute(
                        """
                        SELECT id, session_id, symbol, side, quantity, price, notional, pnl, executed_at
                        FROM paper_trades
                        WHERE session_id = %s
                        ORDER BY executed_at DESC
                        LIMIT %s
                        """,
                        (session_id, limit)
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, session_id, symbol, side, quantity, price, notional, pnl, executed_at
                        FROM paper_trades
                        ORDER BY executed_at DESC
                        LIMIT %s
                        """,
                        (limit,)
                    )
                rows = cur.fetchall()
                trades = []
                for row in rows:
                    trades.append({
                        "id": row[0],
                        "session_id": row[1],
                        "symbol": row[2],
                        "side": row[3],
                        "quantity": row[4],
                        "price": row[5],
                        "notional": row[6],
                        "pnl": row[7],
                        "executed_at": row[8].isoformat() if row[8] else None,
                    })
                return {"trades": trades, "count": len(trades)}
    except Exception as e:
        return {"trades": [], "count": 0, "error": str(e)}


class PauseResumeRequest(BaseModel):
    session_id: str


@app.post("/api/paper/pause")
def pause_session(request: PauseResumeRequest):
    """Pause a paper trading session."""
    session_id = request.session_id
    if session_id not in paper_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    paper_sessions[session_id]["status"] = "paused"
    save_session_to_db(paper_sessions[session_id])
    return {"status": "paused", "session_id": session_id}


@app.post("/api/paper/resume")
def resume_session(request: PauseResumeRequest):
    """Resume a paused paper trading session."""
    session_id = request.session_id
    if session_id not in paper_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    paper_sessions[session_id]["status"] = "running"
    save_session_to_db(paper_sessions[session_id])
    return {"status": "running", "session_id": session_id}


class CloneRequest(BaseModel):
    session_id: str
    new_symbol: str = None


@app.post("/api/paper/clone")
def clone_session(request: CloneRequest):
    """Clone an existing session with optional new symbol."""
    import uuid
    
    source_id = request.session_id
    if source_id not in paper_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    source = paper_sessions[source_id]
    new_id = str(uuid.uuid4())[:8]
    
    new_session = {
        "id": new_id,
        "status": "running",
        "algorithm": source.get("algorithm"),
        "symbol": request.new_symbol or source.get("symbol"),
        "config": source.get("config", {}).copy(),
        "account_id": source.get("account_id"),
        "buying_power": source.get("buying_power"),
        "key_id": source.get("key_id"),
        "secret_key": source.get("secret_key"),
        "started_at": datetime.now().isoformat(),
        "trades": 0,
        "pnl": 0,
        "position": None,
    }
    
    paper_sessions[new_id] = new_session
    save_session_to_db(new_session)
    
    return {
        "status": "cloned",
        "new_session_id": new_id,
        "symbol": new_session["symbol"],
    }


@app.get("/api/paper/analytics")
def get_analytics():
    """Get analytics data for charts and rankings."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # 1. Get daily P&L history (aggregated)
                # This is a simplification. Ideally we'd use a proper time-series query.
                cur.execute(
                    """
                    SELECT date_trunc('hour', executed_at) as hour, SUM(pnl)
                    FROM paper_trades
                    GROUP BY 1
                    ORDER BY 1
                    """
                )
                history = [{"time": row[0].isoformat(), "pnl": row[1]} for row in cur.fetchall()]
                
                # Calculate cumulative P&L
                cumulative = []
                running_sum = 0
                for item in history:
                    running_sum += item["pnl"]
                    cumulative.append({"time": item["time"], "value": running_sum})

                # 2. Get detailed rankings
                sessions = list(paper_sessions.values())
                rankings = []
                for s in sessions:
                    # Calculate basic stats if not present
                    pnl = s.get("pnl", 0)
                    initial = s.get("config", {}).get("buy_amount", 10000)
                    roi = (pnl / initial * 100) if initial > 0 else 0
                    
                    rankings.append({
                        "id": s.get("id"),
                        "symbol": s.get("symbol"),
                        "algorithm": s.get("algorithm"),
                        "pnl": pnl,
                        "roi": roi,
                        "trades": s.get("trades", 0),
                        "status": s.get("status")
                    })
                
                # Sort by P&L desc
                rankings.sort(key=lambda x: x["pnl"], reverse=True)
                
                return {
                    "chart_data": cumulative,
                    "rankings": rankings
                }
    except Exception as e:
        print(f"Analytics error: {e}")
        return {"chart_data": [], "rankings": [], "error": str(e)}
