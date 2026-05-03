import os
import logging
import json
import pandas as pd
import yfinance as yf
import numpy as np
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.readers.sec_filings import SECFilingsReader
from llama_index.tools.tavily_research import TavilyToolSpec

# 1. LOGGING & AUDIT CONFIG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PortfolioAI")

# 2. ANALYST AGENT (Web SEC Filings + Private Client PDFs)
def get_analyst_agent(client_files_path="./client_input"):
    if not os.path.exists(client_files_path):
        os.makedirs(client_files_path)
    
    # Private local data (Excel/PDF)
    client_docs = SimpleDirectoryReader(client_files_path).load_data()
    client_index = VectorStoreIndex.from_documents(client_docs)
    client_tool = QueryEngineTool(
        query_engine=client_index.as_query_engine(),
        metadata=ToolMetadata(name="client_file_search", description="Searches private client holdings/files.")
    )

    # Public SEC web search
    def search_web_10k(ticker: str, form_type: str = "10-K"):
        """Fetches public 10-Ks directly from SEC EDGAR."""
        reader = SECFilingsReader()
        docs = reader.load_data(tickers=[ticker], amount=1, filing_type=form_type)
        idx = VectorStoreIndex.from_documents(docs)
        return idx.as_query_engine().query(f"Identify fundamental risks for {ticker}")

    return ReActAgent(
        name="analyst_agent",
        description="Fundamental analyst using web SEC data and local client files.",
        system_prompt="You are a Fundamental Analyst. Use web tools for 10-Ks and local tools for private data.",
        tools=[client_tool, FunctionTool.from_defaults(fn=search_web_10k)],
        llm=Settings.llm
    )

# 3. PULSE AGENT (Real-time News & Sentiment)
def get_pulse_agent():
    tavily_tool = TavilyToolSpec(api_key=os.environ.get("TAVILY_API_KEY"))
    return ReActAgent(
        name="pulse_agent",
        description="Pulls real-time news headlines and sentiment via Google/Tavily.",
        system_prompt="You are a Market Strategist. Search for recent news, analyst ratings, and insider trades.",
        tools=tavily_tool.to_tool_list(),
        llm=Settings.llm
    )

# 4. QUANT AGENT (Sharpe Ratio & Historical Math)
def get_quant_agent():
    def get_metrics(ticker: str, risk_free_rate: float = 0.04):
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty: return f"Error: No price data for {ticker}"
        
        returns = hist['Close'].pct_change().dropna()
        ann_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1)
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = (ann_return - risk_free_rate) / ann_vol
        
        return {
            "ticker": ticker,
            "ann_return": f"{ann_return:.2%}",
            "ann_volatility": f"{ann_vol:.2%}",
            "sharpe_ratio": round(sharpe, 2)
        }

    return ReActAgent(
        name="quant_agent",
        description="Calculates historical performance and Sharpe ratios.",
        tools=[FunctionTool.from_defaults(fn=get_metrics)],
        llm=Settings.llm
    )

# 5. MASTER ORCHESTRATOR
class PortfolioBot:
    def __init__(self, region="us-east-1"):
        Settings.llm = Bedrock(model="anthropic.claude-3-5-sonnet-20240620-v1:0", region_name=region)
        Settings.embed_model = BedrockEmbedding(model_name="amazon.titan-embed-text-v2:0", region_name=region)
        self.output_dir = "./outputs"
        
        self.workflow = AgentWorkflow(
            agents=[get_analyst_agent(), get_pulse_agent(), get_quant_agent(),
                    ReActAgent(
                        name="optimizer_agent",
                        description="Lead orchestrator that determines consensus and generates reports.",
                        system_prompt=(
                            "You are a Quant Lead. For any equity, you MUST:\n"
                            "1. Query quant_agent for Sharpe Ratio and volatility.\n"
                            "2. Query analyst_agent for fundamental risks.\n"
                            "3. Query pulse_agent for news sentiment.\n"
                            "4. Use calculate_consensus tool to determine majority vote (2/3).\n"
                            "5. Generate an Excel report with a 'Reasoning' column explaining all inputs.\n"
                            "If off-topic, say: 'I am not able to help with this question.'"
                        ),
                        tools=[
                            FunctionTool.from_defaults(fn=self.generate_excel_report),
                            FunctionTool.from_defaults(fn=self.calculate_consensus)
                        ],
                        llm=Settings.llm
                    )],
            root_agent="optimizer_agent"
        )

    def calculate_consensus(self, quant_sig: str, analyst_sig: str, pulse_sig: str):
        signals = [quant_sig.lower(), analyst_sig.lower(), pulse_sig.lower()]
        pos, neg = signals.count("positive"), signals.count("negative")
        score = "100%" if (pos == 3 or neg == 3) else "66%"
        return {"confidence": score, "vote": "Bullish" if pos >= 2 else "Bearish"}

    def generate_excel_report(self, data_json: str):
        os.makedirs(self.output_dir, exist_ok=True)
        try: os.chmod(self.output_dir, 0o777)
        except Exception: pass
        
        path = f"{self.output_dir}/Consensus_Report.xlsx"
        df = pd.DataFrame(json.loads(data_json))
        df.to_excel(path, index=False)
        return f"SUCCESS: Report saved at {path}"

    async def chat(self, user_input: str):
        valid = ["stock", "portfolio", "optimize", "invest", "10-k", "10-q"]
        if not any(word in user_input.lower() for word in valid):
            return "I am not able to help with this question.", None
            
        response = await self.workflow.run(user_msg=user_input)
        report_path = f"{self.output_dir}/Consensus_Report.xlsx"
        return str(response), (report_path if os.path.exists(report_path) else None)