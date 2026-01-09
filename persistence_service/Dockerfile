FROM python:3.9-slim

WORKDIR /app

COPY persistence_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module
COPY shared /app/shared

COPY persistence_service /app/persistence_service

ENV PYTHONPATH=/app
CMD ["python", "persistence_service/main.py"]
