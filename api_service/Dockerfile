FROM python:3.9-slim

WORKDIR /app

COPY api_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module
COPY shared /app/shared

COPY api_service /app/api_service

# Expose port
EXPOSE 8001

ENV PYTHONPATH=/app
CMD ["uvicorn", "api_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
