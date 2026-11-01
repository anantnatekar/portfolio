# PortfolioAI

AI-powered equity research platform — live market dashboard + three-agent AI analysis (Quantitative, Fundamental/SEC, News Sentiment) — packaged as a **single deployable app** on Railway.

**Live:** [portfolio-production-a9a0.up.railway.app](https://portfolio-production-a9a0.up.railway.app/)

---

## What's in the box

| URL | What you get |
|---|---|
| `/` | Live DJIA market dashboard + native PortfolioAI chat panel |
| `/chat` | Chainlit agent UI (original chat interface) |
| `/api/*` | REST + SSE endpoints (DJIA, analysis stream, file upload) |
| `/health` | Railway healthcheck |
| `/api/docs` | Swagger API documentation |

Everything runs from **one process, one port, one Railway service** — no separate servers.

---

## Tech stack

| Layer | Technology |
|---|---|
| Unified server | FastAPI + `mount_chainlit()` + uvicorn |
| Agent UI | Chainlit (mounted at `/chat`) |
| Market dashboard | Vanilla HTML/JS — LightweightCharts + Yahoo Finance |
| Agents | LlamaIndex ReActAgent + AgentWorkflow |
| LLM | Anthropic Claude (direct API) |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` |
| News | Tavily Search API |
| SEC data | SEC EDGAR API (live) |
| Market data | yfinance |
| Hosting | Railway (Dockerfile) |
| CI/CD | GitHub Actions → Railway |

---

## Project structure

```
portfolioai/
├── main.py                     ← NEW: unified FastAPI entry point
├── app.py                      ← Chainlit agents + business logic (unchanged)
├── portfolioai_dashboard.html  ← Market dashboard (served at /)
├── requirements.txt            ← Updated: adds fastapi, uvicorn, multipart
├── Dockerfile                  ← Updated: runs uvicorn main:app
├── docker-compose.yml          ← Local dev
├── railway.toml                ← Railway config
├── .env.example                ← Environment variable template
├── .github/workflows/
│   └── deploy.yml              ← CI/CD: SonarQube → Railway
├── outputs/                    ← Generated Excel reports (gitignored)
└── client_input/               ← Private client files (gitignored)
```

---

## Deploy to Railway

### First time

1. **Fork / clone** the repo and push to GitHub
2. **Create Railway project** → New Project → Deploy from GitHub repo
3. **Set environment variables** in Railway dashboard → Variables:

   | Variable | Value |
   |---|---|
   | `ANTHROPIC_API_KEY` | From [console.anthropic.com](https://console.anthropic.com) |
   | `TAVILY_API_KEY` | From [app.tavily.com](https://app.tavily.com) |

4. Railway detects the Dockerfile, builds, and deploys automatically
5. Get your URL: Railway dashboard → Settings → Domains → Generate Domain

### CI/CD (auto-deploy on push)

Add these GitHub Secrets (Settings → Secrets → Actions):

| Secret | Source |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic console |
| `TAVILY_API_KEY` | Tavily dashboard |
| `RAILWAY_TOKEN` | Railway → Project → Settings → Tokens |
| `SONAR_TOKEN` | SonarQube → My Account → Security |

Every push to `main` → SonarQube scan → Railway deploy.

---

## Local development

```bash
# 1. Clone and set up environment
git clone https://github.com/your-username/portfolioai.git
cd portfolioai
cp .env.example .env
# Edit .env with your API keys

# 2a. Docker (mirrors production exactly — recommended)
docker compose up --build
# Open http://localhost:8080

# 2b. Or run directly (faster iteration)
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
# Open http://localhost:8080
```

> **Note:** `chainlit run app.py` no longer works as the primary entry point.
> Use `uvicorn main:app` instead. The Chainlit UI is still available at `/chat`.

---

## How it works

### Architecture

```
Browser
  │
  ├── GET /          → FastAPI serves portfolioai_dashboard.html
  │                     Dashboard JS calls /api/* endpoints (same origin)
  │
  ├── GET /api/analyse/stream?ticker=AAPL   (SSE)
  │     → FastAPI streams progress events as each agent completes
  │     → Final event: structured BUY/HOLD/SELL result JSON
  │
  ├── GET /chat      → Chainlit React UI (full original chat experience)
  │
  └── GET /health    → Railway healthcheck
```

### Analysis flow

1. User enters ticker(s) or uploads a CSV/Excel portfolio file
2. Dashboard opens an SSE connection to `/api/analyse/stream`
3. Three ReAct agents run concurrently via LlamaIndex `AgentWorkflow`:
   - **Quant agent** — Sharpe ratio, annualised return, volatility (yfinance)
   - **Analyst agent** — SEC EDGAR 10-K filing analysis via RAG
   - **Pulse agent** — live news sentiment via Tavily Search
4. Progress events stream to the dashboard, updating the card in real time
5. Orchestrator synthesises signals → BUY / HOLD / SELL with confidence score
6. Result card appears; Excel report available for download

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Anthropic Claude API key |
| `TAVILY_API_KEY` | ✅ | Tavily news search API key |
| `PORT` | Auto | Injected by Railway — do not set |

No AWS credentials required.

---

## Migrating from the previous two-server setup

Previously you needed:
- `chainlit run app.py` on port 8080
- `uvicorn fastapi_bridge:app` on port 8001
- Open `portfolioai_dashboard.html` as a local file

Now you need:
- `uvicorn main:app --port 8080` (or `docker compose up`)
- Open `http://localhost:8080`

`fastapi_bridge.py` is no longer needed and can be deleted.
