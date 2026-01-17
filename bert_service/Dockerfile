FROM python:3.9-slim

WORKDIR /app

COPY bert_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download transformers model cache if possible, but for MVP we load on start

# Copy shared code
COPY shared/ /app/shared/

# Copy service code
COPY bert_service/ /app/bert_service/

# Set Python path
ENV PYTHONPATH=/app

CMD ["python", "bert_service/main.py"]
