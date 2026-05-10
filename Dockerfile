FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (maximises layer cache reuse)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required runtime directories
RUN mkdir -p /app/outputs /app/client_input \
    && chmod -R 777 /app/outputs /app/client_input

# Use shell form of CMD so $PORT is expanded by the shell at runtime
# Railway injects PORT as an env var — shell form picks it up correctly
CMD chainlit run app.py --host 0.0.0.0 --port ${PORT:-8000}
