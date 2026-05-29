FROM python:3.10-slim

WORKDIR /app

COPY context_engine_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the sentence-transformer at build time so cold-start is fast.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy shared module so the service can use the central logger / utils.
COPY shared /app/shared

COPY context_engine_service /app/context_engine_service

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["uvicorn", "context_engine_service.main:app", "--host", "0.0.0.0", "--port", "8008"]
