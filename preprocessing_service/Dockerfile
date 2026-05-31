FROM python:3.9-slim

WORKDIR /app

COPY preprocessing_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module
COPY shared /app/shared

COPY preprocessing_service /app/preprocessing_service

ENV PYTHONUNBUFFERED=1

ENV PYTHONPATH=/app

# Health check verifies the preprocessing process is active
HEALTHCHECK --interval=30s --timeout=15s --start-period=5s --retries=3 \
  CMD kill -0 1 || exit 1

CMD ["python", "preprocessing_service/main.py"]
