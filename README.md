# PortfolioAI

An AI-powered equity analysis chatbot built with Chainlit, LlamaIndex, and Amazon Bedrock. Analyses stocks using three specialised agents — Quant, Fundamental (SEC filings), and News Sentiment — and delivers a **BUY / HOLD / SELL** recommendation with full reasoning and a downloadable Excel report.

---

## Features

- **Multi-equity analysis** — enter one ticker, multiple tickers, company names, or upload a CSV/Excel file
- **Three-agent pipeline** — Quant (Sharpe ratio, returns, volatility), Fundamental (SEC 10-K filings), News (real-time sentiment via Tavily)
- **On-screen result cards** — per-equity recommendation with metrics, reasoning, and source links
- **Downloadable Excel report** — colour-coded recommendations, full reasoning, clickable SEC and news URLs
- **Powered by Claude on Amazon Bedrock**

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Chainlit |
| Agents | LlamaIndex ReActAgent + AgentWorkflow |
| LLM | Anthropic Claude 3.5 Sonnet via Amazon Bedrock |
| Embeddings | Amazon Titan Embed v2 via Bedrock |
| News | Tavily Search API |
| SEC Data | llama-index-readers-sec-filings |
| Market Data | yfinance |
| Hosting | Railway |

---

## Project Structure

```
portfolioai/
├── app.py                          # Main application (agents + Chainlit UI)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container definition
├── docker-compose.yml              # Local development only
├── railway.toml                    # Railway deployment config
├── .env.example                    # Environment variable template
├── .dockerignore                   # Files excluded from Docker image
├── .gitignore                      # Files excluded from git
├── .github/
│   └── workflows/
│       └── deploy.yml              # GitHub Actions → Railway CI/CD
├── outputs/                        # Generated Excel reports (gitignored)
└── client_input/                   # Private client files (gitignored)
```

---

## Deployment (Railway — Production)

### First-Time Setup

**1. Create a Railway account**
Go to [railway.app](https://railway.app) and sign up with your GitHub account.

**2. Create a new project**
- Click **New Project** → **Deploy from GitHub repo**
- Select your `portfolioai` repository
- Railway will auto-detect the Dockerfile and start building

**3. Set environment variables**
In Railway dashboard → your service → **Variables**, add:

| Variable | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | Your AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key |
| `AWS_DEFAULT_REGION` | `us-east-1` |
| `TAVILY_API_KEY` | Your Tavily API key |

Railway automatically injects `PORT` — no need to set it manually.

**4. Get your public URL**
Railway dashboard → your service → **Settings** → **Domains** → **Generate Domain**

You'll get a URL like `https://portfolioai-production.up.railway.app`

**5. Set up CI/CD (auto-deploy on push)**
- In Railway dashboard → your service → **Settings** → **Tokens** → **Create Token**
- Copy the token
- In GitHub → your repo → **Settings** → **Secrets and variables** → **Actions** → **New secret**
  - Name: `RAILWAY_TOKEN`
  - Value: paste the Railway token

From now on, every push to `main` automatically redeploys.

---

## Local Development

```bash
# 1. Clone the repo
git clone https://github.com/your-username/portfolioai.git
cd portfolioai

# 2. Create your .env file
cp .env.example .env
# Edit .env with your real credentials

# 3. Run with Docker Compose
docker compose up --build

# 4. Open the app
open http://localhost:8000
```

---

## Adding Private Client Files

Drop PDF, Excel, or CSV files into the `client_input/` directory before starting the app. The Analyst agent will index and search them when answering queries.

- **Local:** files in `./client_input/` are mounted into the container
- **Railway:** use Railway's volume mount or upload files to S3 and reference them via env var

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | ✅ | AWS credentials for Bedrock |
| `AWS_SECRET_ACCESS_KEY` | ✅ | AWS credentials for Bedrock |
| `AWS_DEFAULT_REGION` | ✅ | AWS region (e.g. `us-east-1`) |
| `TAVILY_API_KEY` | ✅ | Tavily Search API key for news |
| `PORT` | Auto (Railway) | Injected automatically by Railway |

---

## How It Works

1. User enters ticker(s), company name(s), or uploads a CSV/Excel file
2. Each equity is resolved to a ticker symbol via yfinance search
3. Three agents run in parallel per equity:
   - **Quant Agent** — calculates Sharpe ratio, annual return, volatility
   - **Analyst Agent** — fetches and analyses SEC 10-K filing risks
   - **Pulse Agent** — searches real-time news and sentiment via Tavily
4. Orchestrator agent calculates majority vote (2/3 = 66%, 3/3 = 100% confidence)
5. Results shown as cards on screen as each equity completes
6. Excel report generated with colour-coded recommendations and clickable source links
