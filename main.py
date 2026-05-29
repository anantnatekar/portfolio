"""
main.py — PortfolioAI Unified Entry Point
==========================================
Single process serving:
  /          → portfolioai_dashboard.html  (market dashboard + native chat)
  /api/*     → REST + SSE analysis endpoints
  /health    → Railway healthcheck
  /chat      → Chainlit AI agent UI (mounted LAST — required by Chainlit)

CRITICAL ordering rule (Chainlit issue #1166):
  mount_chainlit() MUST be called AFTER all FastAPI routes are registered.
  Any route defined after mount_chainlit() will return 404.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8080 --reload

Deploy:
    Railway reads Dockerfile → runs the CMD above automatically.
"""

import asyncio
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Import shared helpers from app.py (no Chainlit decorators imported here) ──
from app import (
    _validate_env,
    PortfolioBot,
    get_djia_data,
    get_quant_metrics,
    resolve_ticker,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("PortfolioAI")

# Validate API keys on startup — fail fast before Railway marks deploy healthy
_validate_env()

# ── FastAPI app ───────────────────────────────────────────────────────────────
# Do NOT set docs_url / openapi_url on a subpath — conflicts with Chainlit mounts
app = FastAPI(
    title="PortfolioAI",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving for Excel report downloads ───────────────────────────
OUTPUTS_DIR = Path("./outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# ── Bot singleton (lazy init — avoids slow startup blocking healthcheck) ──────
_bot: Optional[PortfolioBot] = None

def get_bot() -> PortfolioBot:
    global _bot
    if _bot is None:
        log.info("Initialising PortfolioBot…")
        _bot = PortfolioBot()
    return _bot


# ── Constants ─────────────────────────────────────────────────────────────────
DOW_30 = [
    "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
    "GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM",
    "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT",
]
TAPE_SYMBOLS = [
    "^DJI","^GSPC","^IXIC","^RUT","GLD","USO","^TNX",
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","V","UNH",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
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
        if n >= 1e12:   return f"${n/1e12:.2f}T"
        if n >= 1e9:    return f"${n/1e9:.2f}B"
        if n >= 1e6:    return f"${n/1e6:.2f}M"
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
    return (
        et.replace(hour=9, minute=30, second=0, microsecond=0)
        <= et
        <= et.replace(hour=16, minute=0, second=0, microsecond=0)
    )

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ════════════════════════════════════════════════════════════════════════════
#  ALL ROUTES — must be defined BEFORE mount_chainlit() is called
# ════════════════════════════════════════════════════════════════════════════

# ── Root — serve the dashboard HTML ─────────────────────────────────────────
DASHBOARD = Path("./portfolioai_dashboard.html")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    if not DASHBOARD.exists():
        raise HTTPException(
            500,
            "portfolioai_dashboard.html not found. "
            "Ensure it is in the same directory as main.py."
        )
    return HTMLResponse(content=DASHBOARD.read_text(encoding="utf-8"))

# ── Favicon — prevents log noise from browser requests ──────────────────────
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Return a minimal 1×1 transparent PNG as the favicon
    # Prevents continuous 404 errors in Railway logs
    import base64
    TRANSPARENT_PNG = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    from fastapi.responses import Response
    return Response(content=TRANSPARENT_PNG, media_type="image/png")

# ── Healthcheck ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":      "ok",
        "service":     "PortfolioAI v2.0",
        "timestamp":   datetime.now().isoformat(),
        "market_open": _is_market_open(),
        "dashboard":   DASHBOARD.exists(),
    }


# ── DJIA snapshot ────────────────────────────────────────────────────────────
@app.get("/api/djia")
def djia_snapshot():
    data = get_djia_data()
    if "error" in data:
        raise HTTPException(503, data["error"])
    return data


# ── DJIA chart series ────────────────────────────────────────────────────────
@app.get("/api/chart/{period}")
def djia_chart(period: str):
    valid = {"1mo", "3mo", "1y", "2y", "5y", "10y"}
    if period not in valid:
        raise HTTPException(400, f"period must be one of {sorted(valid)}")
    interval = "1wk" if period in {"2y", "5y", "10y"} else "1d"
    try:
        hist = yf.Ticker("^DJI").history(
            period=period, interval=interval, auto_adjust=True
        )
        if hist.empty:
            raise HTTPException(503, "No chart data returned from yfinance")
        hist = hist.reset_index()
        dc = "Date" if "Date" in hist.columns else "Datetime"
        hist[dc] = pd.to_datetime(hist[dc]).dt.tz_localize(None)

        series, closes_list, ma50, ma200 = [], [], [], []
        for _, row in hist.iterrows():
            ts = int(row[dc].timestamp())
            c  = _safe(row.get("Close"))
            closes_list.append(c)
            series.append({
                "time": ts, "value": c,
                "open": _safe(row.get("Open")),
                "high": _safe(row.get("High")),
                "low":  _safe(row.get("Low")),
                "close": c,
            })
        for i, s in enumerate(series):
            sl = closes_list
            if i >= 49 and None not in sl[i-49:i+1]:
                ma50.append({"time": s["time"],
                             "value": round(sum(sl[i-49:i+1]) / 50, 2)})
            if i >= 199 and None not in sl[i-199:i+1]:
                ma200.append({"time": s["time"],
                              "value": round(sum(sl[i-199:i+1]) / 200, 2)})
        return {
            "period": period, "interval": interval,
            "series": series, "ma50": ma50, "ma200": ma200,
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Chart error: %s", exc)
        raise HTTPException(503, str(exc))


# ── Ticker tape ──────────────────────────────────────────────────────────────
@app.get("/api/tape")
def ticker_tape():
    results = []
    for sym in TAPE_SYMBOLS:
        try:
            info  = yf.Ticker(sym).info
            price = info.get("regularMarketPrice") or info.get("previousClose", 0) or 0
            prev  = (
                info.get("regularMarketPreviousClose")
                or info.get("previousClose", price)
                or price
            )
            chg = price - prev
            pct = (chg / prev * 100) if prev else 0
            results.append({
                "symbol": sym,
                "price":  _safe(price, 0),
                "change": _safe(chg, 0),
                "pct":    _safe(pct, 0),
                "is_up":  chg >= 0,
            })
        except Exception as e:
            log.warning("Tape %s: %s", sym, e)
    return {"tickers": results, "updated": datetime.now().isoformat()}


# ── DOW 30 components ────────────────────────────────────────────────────────
@app.get("/api/dow30")
def dow30_components():
    results = []
    for sym in DOW_30:
        try:
            info  = yf.Ticker(sym).info
            price = info.get("regularMarketPrice") or info.get("previousClose", 0) or 0
            prev  = (
                info.get("regularMarketPreviousClose")
                or info.get("previousClose", price)
                or price
            )
            chg = price - prev
            pct = (chg / prev * 100) if prev else 0
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
        "updated":    datetime.now().isoformat(),
    }


# ── Quantitative metrics (fast — no LLM) ────────────────────────────────────
@app.get("/api/quant/{ticker}")
def quant_metrics(ticker: str):
    resolved, _ = resolve_ticker(ticker)
    data = get_quant_metrics(resolved)
    if "error" in data:
        raise HTTPException(404, data["error"])
    return data


# ── SSE streaming analysis ───────────────────────────────────────────────────
@app.get("/api/analyse/stream")
async def analyse_stream(
    ticker:  str = Query(..., description="Ticker or company name, e.g. AAPL"),
    company: str = Query("",  description="Display name override (optional)"),
):
    """
    Server-Sent Events.  Dashboard opens EventSource → receives:
      event: progress  data: {"step": "quant_start"|"quant_done"|"fund_start"|
                                      "fund_done"|"news_done"|"consensus_done"}
      event: result    data: {full analysis dict}
      event: error     data: {"message": "..."}
    """
    resolved_ticker, resolved_company = resolve_ticker(ticker)
    if not company:
        company = resolved_company

    async def event_gen():
        bot = get_bot()
        log.info("SSE start: %s (%s)", resolved_ticker, company)
        try:
            yield _sse("progress", {"step": "quant_start",
                                    "ticker": resolved_ticker})
            await asyncio.sleep(0)

            quant_data = await asyncio.get_event_loop().run_in_executor(
                None, get_quant_metrics, resolved_ticker
            )
            yield _sse("progress", {"step": "quant_done",
                                    "ticker": resolved_ticker,
                                    "data": quant_data})
            await asyncio.sleep(0)

            yield _sse("progress", {"step": "fund_start",
                                    "ticker": resolved_ticker})
            await asyncio.sleep(0)

            # Full 3-agent analysis (Quant + Fundamental + News internally)
            result = await bot.analyse_ticker(resolved_ticker, company)

            yield _sse("progress", {"step": "fund_done",
                                    "ticker": resolved_ticker})
            yield _sse("progress", {"step": "news_done",
                                    "ticker": resolved_ticker})
            yield _sse("progress", {"step": "consensus_done",
                                    "ticker": resolved_ticker})
            await asyncio.sleep(0)

            yield _sse("result", result)
            log.info("SSE complete: %s", resolved_ticker)

        except Exception as exc:
            log.error("SSE error %s: %s", resolved_ticker, exc)
            yield _sse("error", {
                "message": str(exc),
                "ticker":  resolved_ticker,
            })

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",   # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── File upload → extract tickers ────────────────────────────────────────────
@app.post("/api/analyse/file")
async def analyse_file(file: UploadFile = File(...)):
    """
    Upload a CSV or Excel portfolio file.
    Returns list of resolved ticker symbols for the dashboard to stream.
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

        cols_lower  = {c.lower(): c for c in df.columns}
        target_col  = next(
            (cols_lower[h]
             for h in ("ticker", "symbol", "stock", "company", "name")
             if h in cols_lower),
            df.columns[0],
        )
        raw = df[target_col].dropna().astype(str).str.strip().tolist()
        seen = set()
        for val in raw:
            v = val.strip()
            if v and v.lower() not in ("ticker","symbol","company","name","stock"):
                if v not in seen:
                    seen.add(v)
                    resolved, _ = resolve_ticker(v)
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


# ════════════════════════════════════════════════════════════════════════════
#  mount_chainlit MUST be LAST — after all routes above are registered.
#  (Chainlit issue #1166: routes defined after this call return 404)
# ════════════════════════════════════════════════════════════════════════════
from chainlit.utils import mount_chainlit          # noqa: E402
mount_chainlit(app=app, target="app.py", path="/chat")
