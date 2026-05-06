import os
import asyncio
import logging
import json
import pandas as pd
import yfinance as yf
import numpy as np
import chainlit as cl
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
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
    """Fail fast with a clear message if required env vars are missing."""
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
# 2. ANALYST AGENT  (SEC Filings + Private Client Files)
# ---------------------------------------------------------------------------
def get_analyst_agent(client_files_path: str = "./client_input") -> ReActAgent:
    os.makedirs(client_files_path, exist_ok=True)

    client_docs = SimpleDirectoryReader(client_files_path).load_data()
    if not client_docs:
        logger.warning(
            "No files found in '%s'. Analyst agent will have no private client context.",
            client_files_path,
        )
        from llama_index.core.schema import Document
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
        """Fetch and summarise fundamental risks from SEC EDGAR filings."""
        try:
            reader = SECFilingsReader()
            docs = reader.load_data(tickers=[ticker], amount=1, filing_type=form_type)
            if not docs:
                return f"No {form_type} filing found for {ticker}."
            idx = VectorStoreIndex.from_documents(docs)
            return str(idx.as_query_engine().query(
                f"Identify the key fundamental risks for {ticker}"
            ))
        except Exception as exc:
            logger.error("SEC filing fetch failed for %s: %s", ticker, exc)
            return f"Error fetching SEC filing for {ticker}: {exc}"

    return ReActAgent(
        name="analyst_agent",
        description="Fundamental analyst: queries SEC EDGAR and private client files.",
        system_prompt=(
            "You are a Fundamental Analyst. "
            "Use search_web_10k for public SEC filings and client_file_search for private data. "
            "Return a concise summary of risks and a signal: 'positive' or 'negative'."
        ),
        tools=[client_tool, FunctionTool.from_defaults(fn=search_web_10k)],
        llm=Settings.llm,
    )


# ---------------------------------------------------------------------------
# 3. PULSE AGENT  (Real-time News & Sentiment)
# ---------------------------------------------------------------------------
def get_pulse_agent() -> ReActAgent:
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        raise EnvironmentError("TAVILY_API_KEY is not set.")

    tavily_tool = TavilyToolSpec(api_key=tavily_key)
    return ReActAgent(
        name="pulse_agent",
        description="Pulls real-time news headlines and market sentiment via Tavily.",
        system_prompt=(
            "You are a Market Strategist. "
            "Search for recent news, analyst ratings, and insider trades for the given ticker. "
            "Return a concise summary and a signal: 'positive' or 'negative'."
        ),
        tools=tavily_tool.to_tool_list(),
        llm=Settings.llm,
    )


# ---------------------------------------------------------------------------
# 4. QUANT AGENT  (Sharpe Ratio & Historical Performance)
# ---------------------------------------------------------------------------
def get_quant_agent() -> ReActAgent:
    def get_metrics(ticker: str, risk_free_rate: float = 0.04) -> dict | str:
        """Calculate annualised return, volatility, and Sharpe ratio."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            if hist.empty:
                return f"Error: No price data found for '{ticker}'."

            returns = hist["Close"].pct_change().dropna()
            ann_return = (hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1
            ann_vol = returns.std() * np.sqrt(252)

            if ann_vol == 0:
                return f"Error: Zero volatility for '{ticker}' — cannot compute Sharpe ratio."

            sharpe = (ann_return - risk_free_rate) / ann_vol
            return {
                "ticker": ticker,
                "ann_return": f"{ann_return:.2%}",
                "ann_volatility": f"{ann_vol:.2%}",
                "sharpe_ratio": round(sharpe, 2),
                "signal": "positive" if sharpe > 1.0 else "negative",
            }
        except Exception as exc:
            logger.error("Quant metrics failed for %s: %s", ticker, exc)
            return f"Error computing metrics for {ticker}: {exc}"

    return ReActAgent(
        name="quant_agent",
        description="Calculates historical performance metrics and Sharpe ratios.",
        tools=[FunctionTool.from_defaults(fn=get_metrics)],
        llm=Settings.llm,
    )


# ---------------------------------------------------------------------------
# 5. PORTFOLIO BOT CORE
# ---------------------------------------------------------------------------
class PortfolioBot:
    VALID_KEYWORDS = [
        "stock", "equity", "ticker", "share", "portfolio", "optimize",
        "invest", "10-k", "10-q", "sec", "analyze", "analyse", "buy",
        "sell", "market", "earnings", "fund", "etf",
    ]

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

        optimizer = ReActAgent(
            name="optimizer_agent",
            description="Lead orchestrator: coordinates all agents and generates the final report.",
            system_prompt=(
                "You are a Quant Lead. For any equity or portfolio query you MUST:\n"
                "1. Call quant_agent to get the Sharpe ratio and volatility signal.\n"
                "2. Call analyst_agent for fundamental risks and signal.\n"
                "3. Call pulse_agent for news sentiment and signal.\n"
                "4. Use calculate_consensus with the three signals to determine the majority vote.\n"
                "5. Call generate_excel_report with a JSON array of results including a 'Reasoning' column.\n"
                "6. Present a clear, structured summary to the user.\n"
                "If the question is off-topic, reply: 'I am not able to help with this question.'"
            ),
            tools=[
                FunctionTool.from_defaults(fn=self.generate_excel_report),
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
        """Determine majority vote from three agent signals."""
        signals = [s.strip().lower() for s in [quant_sig, analyst_sig, pulse_sig]]
        pos = signals.count("positive")
        neg = signals.count("negative")
        return {
            "confidence": "100%" if (pos == 3 or neg == 3) else "66%",
            "vote": "Bullish" if pos >= 2 else "Bearish",
        }

    def generate_excel_report(self, data_json: str) -> str:
        """Write consensus report to Excel."""
        cleaned = (
            data_json.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse report JSON: %s\nRaw: %s", exc, data_json)
            return f"ERROR: Could not parse report data — {exc}"

        try:
            path = os.path.join(self.output_dir, "Consensus_Report.xlsx")
            df = pd.DataFrame(data if isinstance(data, list) else [data])
            df.to_excel(path, index=False, engine="openpyxl")
            logger.info("Report saved: %s", path)
            return f"SUCCESS: Report saved at {path}"
        except Exception as exc:
            logger.error("Excel write failed: %s", exc)
            return f"ERROR: Could not write Excel report — {exc}"

    async def chat(self, user_input: str) -> tuple[str, str | None]:
        if not any(kw in user_input.lower() for kw in self.VALID_KEYWORDS):
            return "I am not able to help with this question.", None

        try:
            response = await self.workflow.run(user_msg=user_input)
        except Exception as exc:
            logger.error("Workflow error: %s", exc)
            return f"An error occurred while processing your request: {exc}", None

        report_path = os.path.join(self.output_dir, "Consensus_Report.xlsx")
        return str(response), (report_path if os.path.exists(report_path) else None)


# ---------------------------------------------------------------------------
# 6. CHAINLIT UI
# ---------------------------------------------------------------------------

# Initialise the bot once at startup — not per session (it's expensive)
_bot: PortfolioBot | None = None


def get_bot() -> PortfolioBot:
    global _bot
    if _bot is None:
        _bot = PortfolioBot()
    return _bot


@cl.on_chat_start
async def on_chat_start():
    """Greet the user when a new chat session opens."""
    await cl.Message(
        content=(
            "👋 **Welcome to PortfolioAI**\n\n"
            "I can help you analyse equities using real-time news, SEC filings, "
            "and quantitative metrics.\n\n"
            "Try asking:\n"
            "- *Analyse AAPL stock*\n"
            "- *Should I invest in MSFT?*\n"
            "- *Get the 10-K risks for NVDA*"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle each incoming user message."""
    bot = get_bot()

    # Show a thinking indicator while the agents run
    async with cl.Step(name="PortfolioAI is thinking...") as step:
        response, report_path = await bot.chat(message.content)
        step.output = "Analysis complete."

    # Send the main text response
    await cl.Message(content=response).send()

    # Attach the Excel report as a downloadable file if one was generated
    if report_path and os.path.exists(report_path):
        elements = [
            cl.File(
                name="Consensus_Report.xlsx",
                path=report_path,
                display="inline",
            )
        ]
        await cl.Message(
            content="📊 Your consensus report is ready to download:",
            elements=elements,
        ).send()
