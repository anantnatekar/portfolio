FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/outputs /app/client_input \
    && chmod -R 777 /app/outputs /app/client_input

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import app" || exit 1

# Hardcode port 8000 — avoids Railway $PORT expansion issues
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8080"]
