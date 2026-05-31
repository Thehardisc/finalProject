# Stage 1: Build dependencies
FROM python:3.9-slim AS builder

WORKDIR /app
COPY llm_reasoning_service/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM python:3.9-slim

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy shared and service code
COPY shared/ /app/shared/
COPY llm_reasoning_service/ /app/llm_reasoning_service/

ENV PYTHONPATH=/app

# Health check verifies the reasoning process is active
HEALTHCHECK --interval=30s --timeout=15s --start-period=5s --retries=3 \
  CMD kill -0 1 || exit 1

CMD ["python", "llm_reasoning_service/main.py"]
