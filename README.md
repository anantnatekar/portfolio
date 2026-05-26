# PortfolioAI

An AI-powered equity research platform built with Chainlit, LlamaIndex, and the Anthropic Claude API. Analyses stocks using three independent ReAct agents — Quant, Fundamental (SEC filings), and News Sentiment — and delivers a **BUY / HOLD / SELL** recommendation with full reasoning and a downloadable Excel report.

**Live app:** [portfolio-production-a9a0.up.railway.app](https://portfolio-production-a9a0.up.railway.app/)

---

## Features

- **Multi-equity analysis** — enter one ticker, multiple tickers, company names, or upload a CSV/Excel file
- **Three-agent pipeline** — Quant (Sharpe ratio, annual return, volatility), Fundamental (live SEC EDGAR 10-K filings via RAG), News (real-time sentiment via Tavily Search)
- **On-screen result cards** — per-equity recommendation with metrics, reasoning, and source links as each agent completes
- **Downloadable Excel report** — colour-coded BUY/HOLD/SELL, full reasoning, clickable SEC filing and news URLs
- **DJIA market overview** — interactive price chart and market stats on startup
- **Portfolio upload** — drop a CSV or Excel with a column of tickers or company names for bulk analysis

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Chainlit ≥ 1.3.0 |
| Agents | LlamaIndex ReActAgent + AgentWorkflow |
| LLM | Anthropic Claude (via `llama-index-llms-anthropic`) |
| Embeddings | HuggingFace (`BAAI/bge-small-en-v1.5` via `llama-index-embeddings-huggingface`) |
| News | Tavily Search API (`llama-index-tools-tavily-research`) |
| SEC Data | `llama-index-readers-sec-filings` + SEC EDGAR API |
| Market Data | yfinance |
| Charts | Plotly |
| Reports | pandas + openpyxl |
| Hosting | Railway (Dockerfile deploy) |
| CI/CD | GitHub Actions → Railway |
| Code Quality | SonarQube |

---

## Project Structure

```
portfolioai/
├── app.py                          # Main application — agents + Chainlit UI
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container definition (port 8080)
├── docker-compose.yml              # Local development only
├── railway.toml                    # Railway deployment config
├── .env.example                    # Environment variable template
├── .dockerignore                   # Files excluded from Docker image
├── .gitignore                      # Files excluded from git
├── sonar-project.properties        # SonarQube analysis config
├── setup_ec2.sh                    # Optional: EC2 self-hosted setup
├── .github/
│   └── workflows/
│       └── deploy.yml              # GitHub Actions: SonarQube → Railway CI/CD
├── .chainlit/
│   └── config.toml                 # Chainlit UI configuration
├── outputs/                        # Generated Excel reports (gitignored)
└── client_input/                   # Private client portfolio files (gitignored)
```

---

## Deployment (Railway — Production)

### Required GitHub Secrets

Before deploying, add these secrets to your GitHub repository:
**Settings → Secrets and variables → Actions → New repository secret**

| Secret | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) → API Keys |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) → API Keys |
| `RAILWAY_TOKEN` | Railway dashboard → Project → Settings → Tokens |
| `SONAR_TOKEN` | SonarQube → My Account → Security → Generate Token |

### First-Time Railway Setup

**1. Create a Railway account**
Go to [railway.app](https://railway.app) and sign up with your GitHub account.

**2. Create a new project**
- Click **New Project** → **Deploy from GitHub repo**
- Select your `portfolioai` repository
- Railway detects the Dockerfile automatically and starts building

**3. Set environment variables in Railway**
Railway dashboard → your service → **Variables** → add:

| Variable | Value |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key |
| `TAVILY_API_KEY` | Your Tavily API key |

Railway injects `PORT` automatically — do **not** set it manually.

**4. Get your public URL**
Railway dashboard → your service → **Settings** → **Domains** → **Generate Domain**

**5. CI/CD is now active**
Every push to `main` automatically runs SonarQube analysis and, if it passes, deploys to Railway. No manual steps needed.

---

## Local Development

```bash
# 1. Clone the repo
git clone https://github.com/your-username/portfolioai.git
cd portfolioai

# 2. Create your .env file
cp .env.example .env
# Open .env and fill in your ANTHROPIC_API_KEY and TAVILY_API_KEY

# 3. Run with Docker Compose (recommended — mirrors production exactly)
docker compose up --build
# App available at http://localhost:8080

# 4. Or run directly with Python (faster for dev iteration)
pip install -r requirements.txt
chainlit run app.py --host 0.0.0.0 --port 8080
# App available at http://localhost:8080
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Anthropic API key for Claude LLM |
| `TAVILY_API_KEY` | ✅ | Tavily Search API key for real-time news |
| `PORT` | Auto | Injected by Railway automatically — do not set |

**No AWS credentials are required.** The app uses Anthropic directly and HuggingFace embeddings (downloaded at runtime, no API key needed).

---

## How It Works

1. User enters ticker(s), company names, or uploads a CSV/Excel portfolio file
2. Each equity is resolved to a validated ticker symbol via yfinance search
3. Three ReAct agents run in parallel per equity via LlamaIndex `AgentWorkflow`:
   - **Quant Agent** — calculates Sharpe ratio, annualised return, and volatility from historical price data
   - **Analyst Agent** — fetches the latest SEC 10-K filing via EDGAR and analyses risk factors, financials, and disclosures using RAG with HuggingFace embeddings
   - **Pulse Agent** — searches real-time news and scores market sentiment via Tavily
4. Orchestrator synthesises all three signals into a majority-vote recommendation (2/3 = 66% confidence, 3/3 = 100%)
5. Results appear as cards on screen as each equity completes — no waiting for the full batch
6. Excel report generated with colour-coded recommendations, full per-agent reasoning, and clickable source links

---

## Adding Private Client Files

Drop PDF, Excel, or CSV files into the `client_input/` directory before starting the app. The Analyst agent indexes and searches them alongside SEC filings.

- **Local:** `./client_input/` is volume-mounted into the container via `docker-compose.yml`
- **Railway:** Use Railway's persistent volume feature (attach to `/app/client_input`) or upload files to S3 and reference them via a custom env var

---

## EC2 Self-Hosted Deployment (Optional)

If you prefer to host on your own EC2 instance rather than Railway, run `setup_ec2.sh` on a fresh Amazon Linux 2023 instance. See the script for full instructions.
