FROM python:3.9-slim

WORKDIR /app

COPY aggregation_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module
COPY shared /app/shared

COPY aggregation_service /app/aggregation_service

ENV PYTHONPATH=/app
CMD ["python", "aggregation_service/main.py"]
