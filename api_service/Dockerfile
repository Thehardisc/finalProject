FROM python:3.9-slim

WORKDIR /app

COPY api_service/requirements.txt .
RUN apt-get update && apt-get install -y gcc python3-dev curl libpq-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# Copy shared module
COPY shared /app/shared
COPY api_service /app/api_service

# Expose port
EXPOSE 8001

ENV PYTHONPATH=/app

# Health check verifies the WebSocket API loop is active
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/ || exit 1

CMD ["uvicorn", "api_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
