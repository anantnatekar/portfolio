FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (maximises layer cache reuse)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required runtime directories
# Railway runs as root — no useradd needed, no permission conflicts
RUN mkdir -p /app/outputs /app/client_input \
    && chmod -R 777 /app/outputs /app/client_input

# Railway dynamically assigns $PORT at runtime
# The start command in railway.toml injects $PORT — this is the fallback
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import app" || exit 1

# Default CMD (railway.toml overrides this with $PORT injected)
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
