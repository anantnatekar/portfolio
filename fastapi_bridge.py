"""
fastapi_bridge.py
─────────────────────────────────────────────────────────────────────────────
REST bridge between portfolioai_dashboard.html and app.py

SETUP:
    pip install fastapi uvicorn python-multipart

RUN (alongside your Chainlit app):
    # Terminal 1 — Chainlit app
    chainlit run app.py --port 8000

    # Terminal 2 — FastAPI bridge (used by the HTML dashboard)
    uvicorn fastapi_bridge:app --host 0.0.0.0 --port 8001 --reload

ENDPOINTS the dashboard calls:
    GET  /health            → liveness check
    GET  /djia              → DJIA snapshot (price, change, stats, sparkline)
    GET  /chart/{period}    → DJIA OHLCV series for a given range (1mo/3mo/1y/2y/5y/10y)
    GET  /tape              → Prices for ticker-tape symbols
    GET  /dow30             → All 30 DJIA components with price + daily change
    POST /quant             → Quantitative metrics for any ticker
    POST /analyse           → Full 3-agent analysis (BUY/HOLD/SELL) — slow, ~30–60s
─────────────────────────────────────────────────────────────────────────────
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Optional

import yfinance as yf
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import directly from app.py (must be in the same directory)
from app import (
    get_djia_data,
    get_quant_metrics,
    resolve_ticker,
    PortfolioBot,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PortfolioAI-Bridge")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="PortfolioAI Bridge API",
    description="REST bridge between the market dashboard and PortfolioAI agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy bot singleton ────────────────────────────────────────────────────────
_bot: Optional[PortfolioBot] = None

def get_bot() -> PortfolioBot:
    global _bot
    if _bot is None:
        logger.info("Initialising PortfolioBot…")
        _bot = PortfolioBot()
        logger.info("PortfolioBot ready.")
    return _bot


# ── Request / response models ─────────────────────────────────────────────────
class TickerRequest(BaseModel):
    ticker: str
    company_name: str = ""


# ── TAPE SYMBOLS ──────────────────────────────────────────────────────────────
TAPE_SYMBOLS = [
    "^DJI", "^GSPC", "^IXIC", "^RUT", "GLD", "USO", "^TNX",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "JPM", "V", "UNH", "GS", "BAC", "XOM",
]

DOW_30 = [
    "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
    "GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM",
    "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT",
]


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _fmt_large(n) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    if n >= 1e12: return f"${n/1e12:.2f}T"
    if n >= 1e9:  return f"${n/1e9:.2f}B"
    if n >= 1e6:  return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"


def _safe_float(v, fallback=None):
    try:
        f = float(v)
        return None if np.isnan(f) or np.isinf(f) else round(f, 4)
    except Exception:
        return fallback


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — dashboard polls this to confirm bridge is running."""
    return {
        "status": "ok",
        "service": "PortfolioAI Bridge",
        "timestamp": datetime.now().isoformat(),
        "market_hours": _is_market_open(),
    }


def _is_market_open() -> bool:
    from zoneinfo import ZoneInfo
    et = datetime.now(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    open_time  = et.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_time = et.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_time <= et <= close_time


@app.get("/djia")
def djia_snapshot():
    """
    Full DJIA snapshot used by the hero section.
    Calls get_djia_data() from app.py directly.
    """
    data = get_djia_data()
    if "error" in data:
        raise HTTPException(status_code=503, detail=data["error"])
    return data


@app.get("/chart/{period}")
def djia_chart_series(period: str = "1y"):
    """
    Returns OHLCV series for LightweightCharts.
    period: 1mo | 3mo | 1y | 2y | 5y | 10y
    """
    interval_map = {
        "1mo": "1d", "3mo": "1d", "1y": "1d",
        "2y": "1wk", "5y": "1wk", "10y": "1wk",
    }
    if period not in interval_map:
        raise HTTPException(status_code=400, detail=f"Invalid period: {period}")

    interval = interval_map[period]
    try:
        dji  = yf.Ticker("^DJI")
        hist = dji.history(period=period, interval=interval, auto_adjust=True)
        if hist.empty:
            raise HTTPException(status_code=503, detail="No chart data available")

        hist = hist.reset_index()
        # Normalise date column
        date_col = "Date" if "Date" in hist.columns else "Datetime"
        hist[date_col] = hist[date_col].dt.tz_localize(None)

        series = []
        for _, row in hist.iterrows():
            ts = int(row[date_col].timestamp())
            series.append({
                "time":  ts,
                "open":  _safe_float(row.get("Open")),
                "high":  _safe_float(row.get("High")),
                "low":   _safe_float(row.get("Low")),
                "close": _safe_float(row.get("Close")),
                "value": _safe_float(row.get("Close")),   # for area series
            })

        # Moving averages
        closes = [s["close"] for s in series if s["close"] is not None]
        ma50, ma200 = [], []
        for i, s in enumerate(series):
            if i >= 49:
                ma50.append({"time": s["time"], "value": round(sum(closes[i-49:i+1])/50,2)})
            if i >= 199:
                ma200.append({"time": s["time"], "value": round(sum(closes[i-199:i+1])/200,2)})

        return {
            "period":   period,
            "interval": interval,
            "series":   series,
            "ma50":     ma50,
            "ma200":    ma200,
            "count":    len(series),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Chart fetch failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/tape")
def ticker_tape():
    """Returns current prices for the ticker tape."""
    results = []
    for sym in TAPE_SYMBOLS:
        try:
            t    = yf.Ticker(sym)
            info = t.info
            price = info.get("regularMarketPrice") or info.get("previousClose", 0)
            prev  = info.get("regularMarketPreviousClose") or info.get("previousClose", price)
            change = price - prev
            pct    = (change / prev * 100) if prev else 0
            results.append({
                "symbol":  sym,
                "price":   _safe_float(price, 0),
                "change":  _safe_float(change, 0),
                "pct":     _safe_float(pct, 0),
                "is_up":   change >= 0,
            })
        except Exception as e:
            logger.warning("Tape fetch failed for %s: %s", sym, e)
    return {"tickers": results, "updated": datetime.now().isoformat()}


@app.get("/dow30")
def dow30_components():
    """Returns all 30 DJIA components with price and daily change."""
    results = []
    for sym in DOW_30:
        try:
            t    = yf.Ticker(sym)
            info = t.info
            price  = info.get("regularMarketPrice") or info.get("previousClose", 0)
            prev   = info.get("regularMarketPreviousClose") or info.get("previousClose", price)
            change = price - prev
            pct    = (change / prev * 100) if prev else 0
            results.append({
                "symbol":       sym,
                "name":         info.get("shortName", sym),
                "price":        _safe_float(price, 0),
                "change":       _safe_float(change, 0),
                "pct":          _safe_float(pct, 0),
                "is_up":        change >= 0,
                "market_cap":   _fmt_large(info.get("marketCap")),
                "pe_ratio":     _safe_float(info.get("trailingPE")),
                "volume":       info.get("regularMarketVolume"),
            })
        except Exception as e:
            logger.warning("DOW30 fetch failed for %s: %s", sym, e)
    return {
        "components": sorted(results, key=lambda x: x["pct"], reverse=True),
        "updated": datetime.now().isoformat(),
    }


@app.post("/quant")
def quant_metrics(req: TickerRequest):
    """
    Returns structured quantitative metrics for a ticker.
    Fast — direct yfinance call, no LLM.
    """
    ticker, _ = resolve_ticker(req.ticker)
    data = get_quant_metrics(ticker)
    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])
    return data


@app.post("/analyse")
async def full_analysis(req: TickerRequest):
    """
    Full 3-agent analysis (Quant + Fundamental + News Sentiment).
    Returns BUY / HOLD / SELL with full reasoning.
    ⚠ This endpoint takes 30–90 seconds per ticker.
    For the dashboard, use the PortfolioAI iframe for interactive analysis.
    """
    ticker, name = resolve_ticker(req.ticker)
    company_name = req.company_name or name

    logger.info("Bridge: starting full analysis for %s (%s)", ticker, company_name)
    try:
        result = await get_bot().analyse_ticker(ticker, company_name)
        return result
    except Exception as exc:
        logger.error("Bridge analysis failed for %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_bridge:app", host="0.0.0.0", port=8001, reload=True)
