FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/outputs && chmod -R 777 /app/outputs

RUN useradd -m botuser

USER botuser

CMD ["python", "app.py"]
