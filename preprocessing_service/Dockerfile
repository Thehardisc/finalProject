FROM python:3.9-slim

WORKDIR /app

COPY preprocessing_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module
COPY shared /app/shared

COPY preprocessing_service /app/preprocessing_service

ENV PYTHONPATH=/app
CMD ["python", "preprocessing_service/main.py"]
