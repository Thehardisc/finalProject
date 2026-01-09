FROM python:3.9-slim

WORKDIR /app

COPY ingestion_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module
# We assume build context is root project dir
COPY shared /app/shared

COPY ingestion_service /app/ingestion_service

# Expose port
EXPOSE 8000

# Run the app
# We need to set pythonpath or run as module to direct imports reliably
ENV PYTHONPATH=/app
CMD ["uvicorn", "ingestion_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
