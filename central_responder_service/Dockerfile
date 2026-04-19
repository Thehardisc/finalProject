FROM python:3.9-slim

WORKDIR /app

COPY central_responder_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared code
COPY shared/ /app/shared/

# Copy service code (includes main.py, meta_learner.py, trainer.py, weights.json)
COPY central_responder_service/ /app/central_responder_service/

# Ensure models dir exists (will be overridden by Docker volume at runtime)
RUN mkdir -p /app/models


# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check verifies the intelligence layer (.ready marker) is ready
HEALTHCHECK --interval=5s --timeout=5s --start-period=5s --retries=100 \
  CMD test -f /app/models/.ready || exit 1

CMD ["python", "central_responder_service/main.py"]
