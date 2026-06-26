FROM python:3.9-slim

WORKDIR /app

COPY goemotions_service/ingestion_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

COPY shared /app/shared
COPY goemotions_service/ingestion_service /app/ingestion_service

EXPOSE 8000

ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "ingestion_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
