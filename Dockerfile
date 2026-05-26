FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by HuggingFace tokenizers and yfinance
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create runtime directories — Railway filesystem is ephemeral
# For persistence, mount a Railway volume to /app/outputs
RUN mkdir -p /app/outputs /app/client_input \
    && chmod -R 777 /app/outputs /app/client_input

# Port 8080 — matches railway.toml internalPort and Railway's expected default
EXPOSE 8080

# Lightweight HTTP healthcheck using curl (installed above)
# Chainlit serves its own HTTP endpoints — we check the root path
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8080"]
