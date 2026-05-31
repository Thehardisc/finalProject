FROM python:3.9-slim

WORKDIR /app

COPY bert_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download transformers model cache if possible, but for MVP we load on start

# Copy shared code
COPY shared/ /app/shared/

# Pre-download models
ENV TRANSFORMERS_CACHE=/app/model_cache
RUN python shared/download_models.py

# Copy service code
COPY bert_service/ /app/bert_service/

# Set Python path
ENV PYTHONPATH=/app

# Health check verifies the transformer analysis process is active
HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=3 \
  CMD kill -0 1 || exit 1

CMD ["python", "bert_service/main.py"]
