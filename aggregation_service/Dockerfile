FROM python:3.9-slim

WORKDIR /app

COPY aggregation_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module
COPY shared /app/shared

COPY aggregation_service /app/aggregation_service

ENV PYTHONPATH=/app

# Health check verifies the aggregation process is active
HEALTHCHECK --interval=30s --timeout=15s --start-period=5s --retries=3 \
  CMD kill -0 1 || exit 1

CMD ["python", "aggregation_service/main.py"]
