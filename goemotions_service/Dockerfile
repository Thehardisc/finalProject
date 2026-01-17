FROM python:3.9-slim

WORKDIR /app

COPY goemotions_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared code
COPY shared/ /app/shared/

# Copy service code
COPY goemotions_service/ /app/goemotions_service/

# Set Python path
ENV PYTHONPATH=/app

CMD ["python", "goemotions_service/main.py"]
