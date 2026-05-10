import os
import io
import asyncio
import logging
import json
import re
import pandas as pd
import yfinance as yf
import numpy as np
import chainlit as cl
from datetime import datetime
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.schema import Document
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.readers.sec_filings import SECFilingsReader
from llama_index.tools.tavily_research import TavilyToolSpec

# ---------------------------------------------------------------------------
# 1. LOGGING & STARTUP VALIDATION
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("PortfolioAI")


def _validate_env():
    missing = [
        var for var in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "TAVILY_API_KEY"]
        if not os.environ.get(var)
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Ensure they are set in your .env file or passed via docker-compose."
        )


# ---------------------------------------------------------------------------
# 2. TICKER RESOLUTION  (company name → ticker symbol)
# ---------------------------------------------------------------------------
def resolve_ticker(input_str: str) -> tuple[str, str]:
    """
    Accept a ticker or company name and return (ticker, company_name).
    Uses yfinance search; falls back to treating the input as a raw ticker.
    """
    input_str = input_str.strip().upper()
    try:
        search = yf.Search(input_str, max_results=1)
        quotes = search.quotes
        if quotes:
            ticker = quotes[0].get("symbol", input_str)
            name = quotes[0].get("longname") or quotes[0].get("shortname") or ticker
            return ticker, name
    except Exception:
        pass
    # Fallback: treat as raw ticker and look up name
    try:
        info = yf.Ticker(input_str).info
        name = info.get("longName") or info.get("shortName") or input_str
        return input_str, name
    except Exception:
        return input_str, input_str


def parse_ticker_input(raw: str) -> list[str]:
    """
    Parse a free-text string into a list of ticker/company tokens.
    Handles comma, space, semicolon, and newline separators.
    """
    tokens = re.split(r"[,;\n]+", raw)
    return [t.strip() for t in tokens if t.strip()]


def parse_file_upload(file_bytes: bytes, filename: str) -> list[str]:
    """
    Read a CSV or Excel file and extract all non-empty values from
    the first column (or a column named 'ticker'/'symbol'/'company').
    """
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))

        # Try to find a named column first
        col = None
        for c in df.columns:
            if c.strip().lower() in ("ticker", "symbol", "company", "name", "equity"):
                col = c
                break
        if col is None:
            col = df.columns[0]

        values = df[col].dropna().astype(str).str.strip().tolist()
        return [v for v in values if v]
    except Exception as exc:
        logger.error("File parse error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# 3. QUANT METRICS  (direct — not via agent, for structured data capture)
# ---------------------------------------------------------------------------
def get_quant_metrics(ticker: str, risk_free_rate: float = 0.04) -> dict:
    """Return structured quant metrics for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        info = stock.info

        if hist.empty:
            return {"error": f"No price data for {ticker}", "signal": "negative"}

        returns = hist["Close"].pct_change().dropna()
        ann_return = (hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else 0

        return {
            "ticker": ticker,
            "current_price": f"${info.get('currentPrice', hist['Close'].iloc[-1]):.2f}",
            "market_cap": _fmt_large(info.get("marketCap")),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "52w_high": f"${info.get('fiftyTwoWeekHigh', 'N/A')}",
            "52w_low": f"${info.get('fiftyTwoWeekLow', 'N/A')}",
            "ann_return": f"{ann_return:.2%}",
            "ann_volatility": f"{ann_vol:.2%}",
            "sharpe_ratio": round(sharpe, 2),
            "signal": "positive" if sharpe > 1.0 else "negative",
        }
    except Exception as exc:
        logger.error("Quant metrics failed for %s: %s", ticker, exc)
        return {"error": str(exc), "signal": "negative"}


def _fmt_large(n) -> str:
    if n is None:
        return "N/A"
    if n >= 1e12:
        return f"${n/1e12:.2f}T"
    if n >= 1e9:
        return f"${n/1e9:.2f}B"
    if n >= 1e6:
        return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"


# ---------------------------------------------------------------------------
# 4. AGENTS
# ---------------------------------------------------------------------------
def get_analyst_agent(client_files_path: str = "./client_input") -> ReActAgent:
    os.makedirs(client_files_path, exist_ok=True)
    client_docs = SimpleDirectoryReader(client_files_path).load_data()
    if not client_docs:
        client_docs = [Document(text="No private client files loaded.")]

    client_index = VectorStoreIndex.from_documents(client_docs)
    client_tool = QueryEngineTool(
        query_engine=client_index.as_query_engine(),
        metadata=ToolMetadata(
            name="client_file_search",
            description="Searches private client holdings and uploaded files.",
        ),
    )

    def search_web_10k(ticker: str, form_type: str = "10-K") -> str:
        """Fetch fundamental risks from SEC EDGAR. Returns text + source URL."""
        try:
            reader = SECFilingsReader()
            docs = reader.load_data(tickers=[ticker], amount=1, filing_type=form_type)
            if not docs:
                return json.dumps({"summary": f"No {form_type} found for {ticker}.", "source_url": ""})
            idx = VectorStoreIndex.from_documents(docs)
            result = str(idx.as_query_engine().query(
                f"Identify the key fundamental risks for {ticker}"
            ))
            sec_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&type={form_type}&dateb=&owner=include&count=5"
            return json.dumps({"summary": result, "source_url": sec_url})
        except Exception as exc:
            logger.error("SEC filing fetch failed for %s: %s", ticker, exc)
            return json.dumps({"summary": f"Error: {exc}", "source_url": ""})

    return ReActAgent(
        name="analyst_agent",
        description="Fundamental analyst: queries SEC EDGAR for 10-K filings.",
        system_prompt=(
            "You are a Fundamental Analyst. Use search_web_10k to get risks from SEC filings. "
            "Return a JSON object with keys: 'fundamental_summary' (string), "
            "'fundamental_signal' ('positive' or 'negative'), 'sec_url' (string)."
        ),
        tools=[client_tool, FunctionTool.from_defaults(fn=search_web_10k)],
        llm=Settings.llm,
    )


def get_pulse_agent() -> ReActAgent:
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        raise EnvironmentError("TAVILY_API_KEY is not set.")
    tavily_tool = TavilyToolSpec(api_key=tavily_key)
    return ReActAgent(
        name="pulse_agent",
        description="Pulls real-time news and sentiment via Tavily.",
        system_prompt=(
            "You are a Market Strategist. Search for recent news, analyst ratings, and "
            "insider trades for the given ticker. "
            "Return a JSON object with keys: 'news_summary' (string), "
            "'news_signal' ('positive' or 'negative'), 'news_urls' (list of up to 3 source URLs)."
        ),
        tools=tavily_tool.to_tool_list(),
        llm=Settings.llm,
    )


def get_quant_agent() -> ReActAgent:
    def get_metrics(ticker: str) -> str:
        return json.dumps(get_quant_metrics(ticker))

    return ReActAgent(
        name="quant_agent",
        description="Calculates Sharpe ratio and historical performance metrics.",
        tools=[FunctionTool.from_defaults(fn=get_metrics)],
        llm=Settings.llm,
    )


# ---------------------------------------------------------------------------
# 5. PORTFOLIO BOT
# ---------------------------------------------------------------------------
class PortfolioBot:

    def __init__(self, region: str = "us-east-1"):
        _validate_env()
        Settings.llm = Bedrock(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            region_name=region,
        )
        Settings.embed_model = BedrockEmbedding(
            model_name="amazon.titan-embed-text-v2:0",
            region_name=region,
        )
        self.output_dir = "./outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        self._build_workflow()

    def _build_workflow(self):
        optimizer = ReActAgent(
            name="optimizer_agent",
            description="Lead orchestrator: coordinates all agents and synthesises results.",
            system_prompt=(
                "You are a Quant Lead analysing a single equity. You MUST:\n"
                "1. Call quant_agent with the ticker to get performance metrics and signal.\n"
                "2. Call analyst_agent with the ticker to get SEC filing risks and signal.\n"
                "3. Call pulse_agent with the ticker to get news sentiment and signal.\n"
                "4. Call calculate_consensus with the three signals.\n"
                "5. Return ONLY a valid JSON object (no markdown, no extra text) with these exact keys:\n"
                "   ticker, company_name, recommendation (BUY/HOLD/SELL), confidence (66% or 100%),\n"
                "   quant_signal, fundamental_signal, news_signal,\n"
                "   ann_return, ann_volatility, sharpe_ratio, current_price, market_cap, pe_ratio,\n"
                "   52w_high, 52w_low,\n"
                "   quant_reasoning (2-3 sentences), fundamental_reasoning (2-3 sentences),\n"
                "   news_reasoning (2-3 sentences), overall_reasoning (3-4 sentences),\n"
                "   sec_url, news_urls (list of strings), analysis_date"
            ),
            tools=[
                FunctionTool.from_defaults(fn=self.calculate_consensus),
            ],
            llm=Settings.llm,
        )
        self.workflow = AgentWorkflow(
            agents=[
                get_analyst_agent(),
                get_pulse_agent(),
                get_quant_agent(),
                optimizer,
            ],
            root_agent="optimizer_agent",
        )

    def calculate_consensus(
        self, quant_sig: str, analyst_sig: str, pulse_sig: str
    ) -> dict:
        signals = [s.strip().lower() for s in [quant_sig, analyst_sig, pulse_sig]]
        pos = signals.count("positive")
        neg = signals.count("negative")
        vote = "BUY" if pos >= 2 else "SELL"
        confidence = "100%" if (pos == 3 or neg == 3) else "66%"
        return {"recommendation": vote, "confidence": confidence}

    async def analyse_ticker(self, ticker: str, company_name: str) -> dict:
        """Run the full agent pipeline for one ticker and return structured result."""
        prompt = (
            f"Analyse the equity with ticker '{ticker}' (company: '{company_name}'). "
            f"Run the full analysis pipeline and return the JSON result. "
            f"Set analysis_date to '{datetime.now().strftime('%Y-%m-%d')}'."
        )
        try:
            response = await self.workflow.run(user_msg=prompt)
            raw = str(response).strip()
            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()
            result = json.loads(raw)
            result.setdefault("ticker", ticker)
            result.setdefault("company_name", company_name)
            result.setdefault("analysis_date", datetime.now().strftime("%Y-%m-%d"))
            return result
        except json.JSONDecodeError:
            logger.error("Failed to parse agent JSON for %s", ticker)
            # Return a minimal error result so the pipeline continues
            quant = get_quant_metrics(ticker)
            return {
                "ticker": ticker,
                "company_name": company_name,
                "recommendation": "ERROR",
                "confidence": "N/A",
                "quant_signal": quant.get("signal", "N/A"),
                "fundamental_signal": "N/A",
                "news_signal": "N/A",
                "ann_return": quant.get("ann_return", "N/A"),
                "ann_volatility": quant.get("ann_volatility", "N/A"),
                "sharpe_ratio": quant.get("sharpe_ratio", "N/A"),
                "current_price": quant.get("current_price", "N/A"),
                "market_cap": quant.get("market_cap", "N/A"),
                "pe_ratio": quant.get("pe_ratio", "N/A"),
                "52w_high": quant.get("52w_high", "N/A"),
                "52w_low": quant.get("52w_low", "N/A"),
                "quant_reasoning": quant.get("error", "Analysis failed"),
                "fundamental_reasoning": "Agent pipeline error",
                "news_reasoning": "Agent pipeline error",
                "overall_reasoning": "Could not complete full analysis. Quant data shown where available.",
                "sec_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&type=10-K",
                "news_urls": [],
                "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            }
        except Exception as exc:
            logger.error("Workflow error for %s: %s", ticker, exc)
            return {
                "ticker": ticker, "company_name": company_name,
                "recommendation": "ERROR", "confidence": "N/A",
                "overall_reasoning": str(exc),
                "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            }

    def generate_excel(self, results: list[dict]) -> str:
        """Write a richly formatted Excel report and return the file path."""
        path = os.path.join(self.output_dir, f"PortfolioAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

        # ── Build flat rows ──────────────────────────────────────────────────
        rows = []
        for r in results:
            news_urls = r.get("news_urls", [])
            if isinstance(news_urls, str):
                try:
                    news_urls = json.loads(news_urls)
                except Exception:
                    news_urls = [news_urls]
            rows.append({
                "Ticker":               r.get("ticker", ""),
                "Company Name":         r.get("company_name", ""),
                "Analysis Date":        r.get("analysis_date", ""),
                "Recommendation":       r.get("recommendation", ""),
                "Confidence":           r.get("confidence", ""),
                "Quant Signal":         r.get("quant_signal", ""),
                "Fundamental Signal":   r.get("fundamental_signal", ""),
                "News Signal":          r.get("news_signal", ""),
                "Current Price":        r.get("current_price", ""),
                "Market Cap":           r.get("market_cap", ""),
                "P/E Ratio":            r.get("pe_ratio", ""),
                "52W High":             r.get("52w_high", ""),
                "52W Low":              r.get("52w_low", ""),
                "Annual Return":        r.get("ann_return", ""),
                "Annual Volatility":    r.get("ann_volatility", ""),
                "Sharpe Ratio":         r.get("sharpe_ratio", ""),
                "Overall Reasoning":    r.get("overall_reasoning", ""),
                "Quant Reasoning":      r.get("quant_reasoning", ""),
                "Fundamental Reasoning":r.get("fundamental_reasoning", ""),
                "News Reasoning":       r.get("news_reasoning", ""),
                "SEC Filing URL":       r.get("sec_url", ""),
                "News Source 1":        news_urls[0] if len(news_urls) > 0 else "",
                "News Source 2":        news_urls[1] if len(news_urls) > 1 else "",
                "News Source 3":        news_urls[2] if len(news_urls) > 2 else "",
            })

        df = pd.DataFrame(rows)
        df.to_excel(path, index=False, sheet_name="Analysis", engine="openpyxl")

        # ── Apply formatting ─────────────────────────────────────────────────
        wb = load_workbook(path)
        ws = wb["Analysis"]

        # Header style
        header_fill = PatternFill("solid", fgColor="1F3864")
        header_font = Font(color="FFFFFF", bold=True, size=11)
        thin = Side(style="thin", color="CCCCCC")
        border = Border(left=thin, right=thin, top=thin, bottom=thin)

        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border

        ws.row_dimensions[1].height = 30

        # Colour-code recommendation column (col D = index 4)
        rec_col = 4
        for row_idx, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row), start=2):
            rec = ws.cell(row=row_idx, column=rec_col).value or ""
            if rec == "BUY":
                fill = PatternFill("solid", fgColor="C6EFCE")
                font_color = "276221"
            elif rec == "SELL":
                fill = PatternFill("solid", fgColor="FFC7CE")
                font_color = "9C0006"
            elif rec == "HOLD":
                fill = PatternFill("solid", fgColor="FFEB9C")
                font_color = "9C5700"
            else:
                fill = PatternFill("solid", fgColor="F2F2F2")
                font_color = "000000"

            for cell in row:
                cell.border = border
                cell.alignment = Alignment(vertical="top", wrap_text=True)
            ws.cell(row=row_idx, column=rec_col).fill = fill
            ws.cell(row=row_idx, column=rec_col).font = Font(
                bold=True, color=font_color
            )
            # Alternate row shading
            if row_idx % 2 == 0:
                for cell in row:
                    if not cell.fill or cell.fill.fgColor.rgb in ("00000000", "FFFFFFFF"):
                        cell.fill = PatternFill("solid", fgColor="F7F9FC")

        # Make URLs clickable
        url_cols = [21, 22, 23, 24]  # SEC URL + 3 news sources
        for col_idx in url_cols:
            for row_idx in range(2, ws.max_row + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if cell.value and str(cell.value).startswith("http"):
                    url = cell.value
                    cell.hyperlink = url
                    cell.font = Font(color="0563C1", underline="single")

        # Column widths
        col_widths = {
            1: 10,  # Ticker
            2: 25,  # Company
            3: 14,  # Date
            4: 16,  # Recommendation
            5: 12,  # Confidence
            6: 14,  # Quant Signal
            7: 18,  # Fundamental Signal
            8: 14,  # News Signal
            9: 14,  # Price
            10: 14, # Mkt Cap
            11: 10, # PE
            12: 12, # 52W High
            13: 12, # 52W Low
            14: 14, # Ann Return
            15: 16, # Ann Vol
            16: 14, # Sharpe
            17: 50, # Overall Reasoning
            18: 40, # Quant Reasoning
            19: 40, # Fundamental Reasoning
            20: 40, # News Reasoning
            21: 35, # SEC URL
            22: 35, # News 1
            23: 35, # News 2
            24: 35, # News 3
        }
        for col_idx, width in col_widths.items():
            ws.column_dimensions[get_column_letter(col_idx)].width = width

        # Freeze header row
        ws.freeze_panes = "A2"

        # Add a summary sheet
        ws_summary = wb.create_sheet("Summary")
        ws_summary["A1"] = "PortfolioAI — Analysis Summary"
        ws_summary["A1"].font = Font(bold=True, size=14, color="1F3864")
        ws_summary["A3"] = "Generated:"
        ws_summary["B3"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        ws_summary["A4"] = "Total Equities:"
        ws_summary["B4"] = len(results)
        buys = sum(1 for r in results if r.get("recommendation") == "BUY")
        sells = sum(1 for r in results if r.get("recommendation") == "SELL")
        holds = sum(1 for r in results if r.get("recommendation") == "HOLD")
        ws_summary["A5"] = "BUY:"
        ws_summary["B5"] = buys
        ws_summary["B5"].font = Font(color="276221", bold=True)
        ws_summary["A6"] = "HOLD:"
        ws_summary["B6"] = holds
        ws_summary["B6"].font = Font(color="9C5700", bold=True)
        ws_summary["A7"] = "SELL:"
        ws_summary["B7"] = sells
        ws_summary["B7"].font = Font(color="9C0006", bold=True)

        row = 9
        ws_summary.cell(row=row, column=1, value="Ticker").font = Font(bold=True)
        ws_summary.cell(row=row, column=2, value="Company").font = Font(bold=True)
        ws_summary.cell(row=row, column=3, value="Recommendation").font = Font(bold=True)
        ws_summary.cell(row=row, column=4, value="Confidence").font = Font(bold=True)
        ws_summary.cell(row=row, column=5, value="Sharpe Ratio").font = Font(bold=True)

        for i, r in enumerate(results, start=row + 1):
            ws_summary.cell(row=i, column=1, value=r.get("ticker", ""))
            ws_summary.cell(row=i, column=2, value=r.get("company_name", ""))
            rec = r.get("recommendation", "")
            ws_summary.cell(row=i, column=3, value=rec)
            color = "276221" if rec == "BUY" else "9C0006" if rec == "SELL" else "9C5700"
            ws_summary.cell(row=i, column=3).font = Font(bold=True, color=color)
            ws_summary.cell(row=i, column=4, value=r.get("confidence", ""))
            ws_summary.cell(row=i, column=5, value=r.get("sharpe_ratio", ""))

        for col in range(1, 6):
            ws_summary.column_dimensions[get_column_letter(col)].width = 20

        wb.save(path)
        logger.info("Excel report saved: %s", path)
        return path


# ---------------------------------------------------------------------------
# 6. CHAINLIT UI
# ---------------------------------------------------------------------------
_bot: PortfolioBot | None = None


def get_bot() -> PortfolioBot:
    global _bot
    if _bot is None:
        _bot = PortfolioBot()
    return _bot


def _recommendation_emoji(rec: str) -> str:
    return {"BUY": "✅", "SELL": "🔴", "HOLD": "🟡"}.get(rec, "❓")


def _signal_emoji(sig: str) -> str:
    return "🟢" if str(sig).lower() == "positive" else "🔴"


def _format_result_card(r: dict) -> str:
    """Format a single equity result as a Markdown card for Chainlit."""
    rec = r.get("recommendation", "N/A")
    emoji = _recommendation_emoji(rec)
    sig_q = _signal_emoji(r.get("quant_signal", ""))
    sig_f = _signal_emoji(r.get("fundamental_signal", ""))
    sig_n = _signal_emoji(r.get("news_signal", ""))

    news_urls = r.get("news_urls", [])
    if isinstance(news_urls, str):
        try:
            news_urls = json.loads(news_urls)
        except Exception:
            news_urls = [news_urls]
    news_links = " | ".join(f"[Source {i+1}]({u})" for i, u in enumerate(news_urls) if u)
    sec_url = r.get("sec_url", "")
    sec_link = f"[SEC Filing]({sec_url})" if sec_url else ""

    return f"""
---
## {emoji} {r.get('company_name', r.get('ticker', ''))} `{r.get('ticker', '')}`

**Recommendation:** `{rec}` &nbsp;|&nbsp; **Confidence:** {r.get('confidence', 'N/A')} &nbsp;|&nbsp; **Date:** {r.get('analysis_date', '')}

### 📊 Signal Summary
| Agent | Signal | Reasoning |
|---|---|---|
| Quant {sig_q} | {r.get('quant_signal', 'N/A').title()} | {r.get('quant_reasoning', '')} |
| Fundamental {sig_f} | {r.get('fundamental_signal', 'N/A').title()} | {r.get('fundamental_reasoning', '')} |
| News {sig_n} | {r.get('news_signal', 'N/A').title()} | {r.get('news_reasoning', '')} |

### 📈 Key Metrics
| Metric | Value |
|---|---|
| Current Price | {r.get('current_price', 'N/A')} |
| Market Cap | {r.get('market_cap', 'N/A')} |
| P/E Ratio | {r.get('pe_ratio', 'N/A')} |
| Annual Return | {r.get('ann_return', 'N/A')} |
| Annual Volatility | {r.get('ann_volatility', 'N/A')} |
| Sharpe Ratio | {r.get('sharpe_ratio', 'N/A')} |
| 52W High | {r.get('52w_high', 'N/A')} |
| 52W Low | {r.get('52w_low', 'N/A')} |

### 💡 Overall Assessment
{r.get('overall_reasoning', 'N/A')}

### 🔗 Sources
{sec_link}{'  |  ' if sec_link and news_links else ''}{news_links}
"""


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "👋 **Welcome to PortfolioAI**\n\n"
            "I analyse equities using quantitative metrics, SEC filings, and real-time news "
            "to give you a **BUY / HOLD / SELL** recommendation with full reasoning.\n\n"
            "**How to use:**\n"
            "- Type one ticker: `AAPL`\n"
            "- Type a company name: `Apple`\n"
            "- Type multiple tickers/names: `AAPL, Microsoft, NVDA`\n"
            "- 📎 **Upload a CSV or Excel file** with a column of tickers or company names\n\n"
            "_Each equity is analysed independently across Quant, Fundamental, and News agents. "
            "A downloadable Excel report with full reasoning and source links is generated automatically._"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    bot = get_bot()
    tickers_to_analyse: list[tuple[str, str]] = []  # (ticker, company_name)

    # ── Handle file uploads ──────────────────────────────────────────────────
    if message.elements:
        for element in message.elements:
            if hasattr(element, "path") and element.path:
                filename = element.name or ""
                if filename.lower().endswith((".csv", ".xlsx", ".xls")):
                    with open(element.path, "rb") as f:
                        raw_tokens = parse_file_upload(f.read(), filename)
                    await cl.Message(
                        content=f"📂 Found **{len(raw_tokens)}** equities in `{filename}`. Resolving tickers..."
                    ).send()
                    for token in raw_tokens:
                        ticker, name = resolve_ticker(token)
                        tickers_to_analyse.append((ticker, name))

    # ── Handle text input ────────────────────────────────────────────────────
    if message.content.strip():
        tokens = parse_ticker_input(message.content)
        if tokens:
            resolving_msg = await cl.Message(
                content=f"🔍 Resolving **{len(tokens)}** equit{'y' if len(tokens)==1 else 'ies'}..."
            ).send()
            for token in tokens:
                ticker, name = resolve_ticker(token)
                tickers_to_analyse.append((ticker, name))

    if not tickers_to_analyse:
        await cl.Message(
            content="⚠️ I couldn't find any equities in your input. Please enter ticker symbols, "
                    "company names (comma-separated), or upload a CSV/Excel file."
        ).send()
        return

    # Deduplicate by ticker
    seen = set()
    unique = []
    for t, n in tickers_to_analyse:
        if t not in seen:
            seen.add(t)
            unique.append((t, n))
    tickers_to_analyse = unique

    await cl.Message(
        content=f"🚀 Starting analysis of **{len(tickers_to_analyse)}** "
                f"equit{'y' if len(tickers_to_analyse)==1 else 'ies'}: "
                f"{', '.join(f'`{t}`' for t, _ in tickers_to_analyse)}\n\n"
                "_This may take a minute per equity — each one runs through three AI agents._"
    ).send()

    # ── Analyse each equity ──────────────────────────────────────────────────
    all_results = []
    for ticker, company_name in tickers_to_analyse:
        async with cl.Step(name=f"Analysing {ticker} — {company_name}") as step:
            step.output = "Running Quant, Fundamental, and News agents..."
            result = await bot.analyse_ticker(ticker, company_name)
            all_results.append(result)
            step.output = f"✅ {ticker} complete — {result.get('recommendation', 'ERROR')}"

        # Show result card immediately so user sees progress
        await cl.Message(content=_format_result_card(result)).send()

    # ── Generate Excel ───────────────────────────────────────────────────────
    async with cl.Step(name="Generating Excel report...") as step:
        excel_path = bot.generate_excel(all_results)
        step.output = "Report ready."

    elements = [
        cl.File(
            name=os.path.basename(excel_path),
            path=excel_path,
            display="inline",
        )
    ]

    # Final summary
    buys  = sum(1 for r in all_results if r.get("recommendation") == "BUY")
    holds = sum(1 for r in all_results if r.get("recommendation") == "HOLD")
    sells = sum(1 for r in all_results if r.get("recommendation") == "SELL")

    await cl.Message(
        content=(
            f"## 📋 Analysis Complete\n\n"
            f"**{len(all_results)} equities analysed** — "
            f"✅ {buys} BUY &nbsp;|&nbsp; 🟡 {holds} HOLD &nbsp;|&nbsp; 🔴 {sells} SELL\n\n"
            f"The Excel report below includes full reasoning, all metrics, "
            f"and clickable links to SEC filings and news sources for every equity."
        ),
        elements=elements,
    ).send()
