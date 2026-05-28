"""
main.py — PortfolioAI Unified Entry Point
==========================================
Single FastAPI application that serves everything from one process:

  /                → portfolioai_dashboard.html  (market dashboard + native chat)
  /api/*           → REST + SSE bridge endpoints  (DJIA, analysis, file upload)
  /chat            → Chainlit AI agent UI         (mounted via mount_chainlit)
  /health          → Railway healthcheck

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8080 --reload

Deploy:
    Docker / Railway — see Dockerfile and railway.toml
"""

import asyncio
import json
import logging
import os
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    StreamingResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Mount Chainlit ─────────────────────────────────────────────────────────────
# This must happen before any other route definition so Chainlit's
# WebSocket and static asset routes don't conflict.
from chainlit.utils import mount_chainlit

# ── Import shared functions from app.py ───────────────────────────────────────
from app import (
    get_djia_data,
    get_quant_metrics,
    resolve_ticker,
    PortfolioBot,
    _validate_env,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("PortfolioAI-Main")

# ── Validate env on startup ────────────────────────────────────────────────────
_validate_env()

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PortfolioAI",
    description="AI-powered equity research platform with live market dashboard.",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount Chainlit at /chat ────────────────────────────────────────────────────
# app.py must be in the same directory as main.py.
# Chainlit's full UI (WebSocket, static files, React frontend) is served here.
mount_chainlit(app=app, target="app.py", path="/chat")

# ── Static files (outputs directory — Excel downloads) ────────────────────────
OUTPUTS_DIR = Path("./outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# ── Bot singleton ──────────────────────────────────────────────────────────────
_bot: Optional[PortfolioBot] = None

def get_bot() -> PortfolioBot:
    global _bot
    if _bot is None:
        log.info("Initialising PortfolioBot…")
        _bot = PortfolioBot()
        log.info("PortfolioBot ready.")
    return _bot


# ── Symbols ────────────────────────────────────────────────────────────────────
DOW_30 = [
    "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
    "GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM",
    "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT",
]

TAPE_SYMBOLS = [
    "^DJI","^GSPC","^IXIC","^RUT","GLD","USO","^TNX",
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","V","UNH",
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def _safe(v, fallback=None):
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return fallback

def _fmt_cap(n) -> str:
    try:
        n = float(n)
        if np.isnan(n): return "N/A"
        if n >= 1e12: return f"${n/1e12:.2f}T"
        if n >= 1e9:  return f"${n/1e9:.2f}B"
        if n >= 1e6:  return f"${n/1e6:.2f}M"
        return f"${n:,.0f}"
    except Exception:
        return "N/A"

def _is_market_open() -> bool:
    try:
        from zoneinfo import ZoneInfo
        et = datetime.now(ZoneInfo("America/New_York"))
    except ImportError:
        import pytz
        et = datetime.now(pytz.timezone("America/New_York"))
    if et.weekday() >= 5:
        return False
    open_t  = et.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = et.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_t <= et <= close_t

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ═══════════════════════════════════════════════════════════════════════════════
#  ROOT — serve the dashboard HTML
# ═══════════════════════════════════════════════════════════════════════════════
DASHBOARD_PATH = Path("./portfolioai_dashboard.html")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    """
    Serves the market dashboard HTML.
    The dashboard's BRIDGE constant must point to /api (same origin).
    """
    if not DASHBOARD_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Dashboard file not found. Ensure portfolioai_dashboard.html is in the app directory."
        )
    return HTMLResponse(content=DASHBOARD_PATH.read_text(encoding="utf-8"))


# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTHCHECK — Railway requires this to pass before marking deploy healthy
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "service":      "PortfolioAI v2.0",
        "timestamp":    datetime.now().isoformat(),
        "market_open":  _is_market_open(),
        "dashboard":    DASHBOARD_PATH.exists(),
        "chainlit":     "mounted at /chat",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS  (prefixed /api/*)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/djia")
def djia_snapshot():
    """Full DJIA snapshot for the hero section."""
    data = get_djia_data()
    if "error" in data:
        raise HTTPException(503, detail=data["error"])
    return data


@app.get("/api/chart/{period}")
def djia_chart(period: str = "1y"):
    """OHLCV series + moving averages for LightweightCharts."""
    valid = {"1mo", "3mo", "1y", "2y", "5y", "10y"}
    if period not in valid:
        raise HTTPException(400, f"period must be one of {valid}")
    interval = "1wk" if period in {"2y", "5y", "10y"} else "1d"
    try:
        hist = yf.Ticker("^DJI").history(period=period, interval=interval, auto_adjust=True)
        if hist.empty:
            raise HTTPException(503, "No chart data returned")
        hist = hist.reset_index()
        dc = "Date" if "Date" in hist.columns else "Datetime"
        hist[dc] = pd.to_datetime(hist[dc]).dt.tz_localize(None)
        series, closes_list, ma50, ma200 = [], [], [], []
        for _, row in hist.iterrows():
            ts = int(row[dc].timestamp())
            c  = _safe(row.get("Close"))
            closes_list.append(c)
            series.append({
                "time":  ts, "value": c,
                "open":  _safe(row.get("Open")),
                "high":  _safe(row.get("High")),
                "low":   _safe(row.get("Low")),
                "close": c,
            })
        for i, s in enumerate(series):
            if i >= 49 and None not in closes_list[i-49:i+1]:
                ma50.append({"time": s["time"], "value": round(sum(closes_list[i-49:i+1])/50, 2)})
            if i >= 199 and None not in closes_list[i-199:i+1]:
                ma200.append({"time": s["time"], "value": round(sum(closes_list[i-199:i+1])/200, 2)})
        return {"period": period, "interval": interval, "series": series, "ma50": ma50, "ma200": ma200}
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Chart error: %s", exc)
        raise HTTPException(503, str(exc))


@app.get("/api/tape")
def ticker_tape():
    """Current prices for the animated ticker tape."""
    results = []
    for sym in TAPE_SYMBOLS:
        try:
            info  = yf.Ticker(sym).info
            price = info.get("regularMarketPrice") or info.get("previousClose", 0) or 0
            prev  = info.get("regularMarketPreviousClose") or info.get("previousClose", price) or price
            chg   = price - prev
            pct   = (chg / prev * 100) if prev else 0
            results.append({
                "symbol": sym, "price": _safe(price, 0),
                "change": _safe(chg, 0), "pct": _safe(pct, 0),
                "is_up":  chg >= 0,
            })
        except Exception as e:
            log.warning("Tape %s: %s", sym, e)
    return {"tickers": results, "updated": datetime.now().isoformat()}


@app.get("/api/dow30")
def dow30_components():
    """All 30 DJIA components — price, change, market cap."""
    results = []
    for sym in DOW_30:
        try:
            info   = yf.Ticker(sym).info
            price  = info.get("regularMarketPrice") or info.get("previousClose", 0) or 0
            prev   = info.get("regularMarketPreviousClose") or info.get("previousClose", price) or price
            chg    = price - prev
            pct    = (chg / prev * 100) if prev else 0
            results.append({
                "symbol":     sym,
                "name":       info.get("shortName", sym),
                "price":      _safe(price, 0),
                "change":     _safe(chg, 0),
                "pct":        _safe(pct, 0),
                "is_up":      chg >= 0,
                "market_cap": _fmt_cap(info.get("marketCap")),
                "pe_ratio":   _safe(info.get("trailingPE")),
                "volume":     info.get("regularMarketVolume"),
            })
        except Exception as e:
            log.warning("DOW30 %s: %s", sym, e)
    return {
        "components": sorted(results, key=lambda x: x["pct"] or 0, reverse=True),
        "updated": datetime.now().isoformat(),
    }


@app.get("/api/quant/{ticker}")
def quant_metrics(ticker: str):
    """Fast quantitative metrics — Sharpe, return, volatility. No LLM."""
    resolved, _ = resolve_ticker(ticker)
    data = get_quant_metrics(resolved)
    if "error" in data:
        raise HTTPException(404, data["error"])
    return data


@app.get("/api/analyse/stream")
async def analyse_stream(
    ticker:  str = Query(..., description="Ticker symbol, e.g. AAPL"),
    company: str = Query("",  description="Company display name (optional)"),
):
    """
    Server-Sent Events stream.
    Dashboard opens EventSource → receives progress events → receives result.

    Events:
        progress  { step: 'quant_start'|'quant_done'|'fund_done'|'news_done'|'consensus_done' }
        result    { full structured analysis dict }
        error     { message: string }
    """
    resolved_ticker, resolved_company = resolve_ticker(ticker)
    if not company:
        company = resolved_company

    async def event_gen():
        bot = get_bot()
        log.info("SSE start: %s (%s)", resolved_ticker, company)
        try:
            yield _sse("progress", {"step": "quant_start", "ticker": resolved_ticker})
            await asyncio.sleep(0)

            # Run quant synchronously in executor (it's CPU-light)
            quant_data = await asyncio.get_event_loop().run_in_executor(
                None, get_quant_metrics, resolved_ticker
            )
            yield _sse("progress", {"step": "quant_done", "ticker": resolved_ticker,
                                    "data": quant_data})
            await asyncio.sleep(0)

            # Signal fundamental start before launching full analysis
            yield _sse("progress", {"step": "fund_start", "ticker": resolved_ticker})
            await asyncio.sleep(0)

            # Full 3-agent analysis (concurrent internally inside PortfolioBot)
            result = await bot.analyse_ticker(resolved_ticker, company)

            yield _sse("progress", {"step": "fund_done",      "ticker": resolved_ticker})
            yield _sse("progress", {"step": "news_done",      "ticker": resolved_ticker})
            yield _sse("progress", {"step": "consensus_done", "ticker": resolved_ticker})
            await asyncio.sleep(0)

            yield _sse("result", result)
            log.info("SSE complete: %s", resolved_ticker)

        except Exception as exc:
            log.error("SSE error for %s: %s", resolved_ticker, exc)
            yield _sse("error", {"message": str(exc), "ticker": resolved_ticker})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/api/analyse/file")
async def analyse_file(file: UploadFile = File(...)):
    """
    Extract ticker symbols from an uploaded CSV or Excel file.
    Returns list of resolved tickers; dashboard streams each one via SSE.
    """
    filename = file.filename or ""
    content  = await file.read()
    tickers  = []
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(400, "Only CSV and Excel files are supported.")

        # Find ticker column
        cols_lower = {c.lower(): c for c in df.columns}
        target_col = next(
            (cols_lower[h] for h in ("ticker","symbol","stock","company","name") if h in cols_lower),
            df.columns[0]
        )
        raw_values = df[target_col].dropna().astype(str).str.strip().tolist()
        seen = set()
        for val in raw_values:
            if val and val.lower() not in ("ticker","symbol","company","name","stock"):
                if val not in seen:
                    seen.add(val)
                    resolved, _ = resolve_ticker(val)
                    if resolved:
                        tickers.append(resolved)
                if len(tickers) >= 20:
                    break
        return {"filename": filename, "tickers": tickers, "count": len(tickers)}
    except HTTPException:
        raise
    except Exception as exc:
        log.error("File parse error: %s", exc)
        raise HTTPException(400, f"Could not parse file: {exc}")
