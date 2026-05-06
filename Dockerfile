FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (maximises layer cache reuse)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required runtime directories with open permissions
RUN mkdir -p /app/outputs /app/client_input \
    && chmod -R 777 /app/outputs /app/client_input

# Create a non-root user for security
RUN useradd -m botuser
USER botuser

# Chainlit listens on port 8000 by default
EXPOSE 8000

# Health check — verifies the app module imports cleanly
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import app" || exit 1

# Launch Chainlit (--host 0.0.0.0 makes it reachable outside the container)
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
