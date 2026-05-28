FROM python:3.11-slim

WORKDIR /app

# System deps:
#   curl     — healthcheck
#   gcc/g++  — HuggingFace tokenizer compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runtime directories — mount Railway volume to /app/outputs for persistence
RUN mkdir -p /app/outputs /app/client_input \
    && chmod -R 777 /app/outputs /app/client_input

EXPOSE 8080

# Real HTTP healthcheck — Chainlit + FastAPI serve HTTP on /health
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Single uvicorn process serves everything:
#   /          → market dashboard HTML
#   /api/*     → REST + SSE bridge
#   /chat      → Chainlit AI agent UI
#   /health    → Railway healthcheck
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "120"]
