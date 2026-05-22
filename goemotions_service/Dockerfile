FROM python:3.9-slim

WORKDIR /app

COPY goemotions_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared code
COPY shared/ /app/shared/

# Pre-download models
ENV TRANSFORMERS_CACHE=/app/model_cache
RUN python shared/download_models.py

# Copy service code
COPY goemotions_service/ /app/goemotions_service/

# Set Python path
ENV PYTHONPATH=/app

# Health check verifies the GoEmotions analysis process is active
HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=3 \
  CMD kill -0 1 || exit 1

CMD ["python", "goemotions_service/main.py"]
