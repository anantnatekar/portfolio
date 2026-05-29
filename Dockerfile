FROM python:3.11-slim

WORKDIR /app

# System deps:
#   curl     — HEALTHCHECK CMD
#   gcc/g++  — HuggingFace tokenizer native compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Create runtime dirs — mount Railway volume to /app/outputs for persistence
RUN mkdir -p /app/outputs /app/client_input \
    && chmod -R 777 /app/outputs /app/client_input

EXPOSE 8080

# Healthcheck hits FastAPI /health (not Chainlit) — always fast
# start-period=90s gives HuggingFace model download time on cold start
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Single uvicorn process — serves dashboard + API + Chainlit from one port
# --timeout-keep-alive 120: Railway's proxy keeps SSE connections open
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--timeout-keep-alive", "120", \
     "--log-level", "info"]
