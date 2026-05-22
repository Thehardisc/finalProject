FROM python:3.9-slim

WORKDIR /app

COPY persistence_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module
COPY shared /app/shared

COPY persistence_service /app/persistence_service

ENV PYTHONPATH=/app

# Health check verifies the process is active
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD kill -0 1 || exit 1

CMD ["python", "persistence_service/main.py"]
