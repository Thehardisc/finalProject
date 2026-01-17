FROM python:3.9-slim

WORKDIR /app

COPY vader_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared code
COPY shared/ /app/shared/

# Copy service code
COPY vader_service/ /app/vader_service/

# Set Python path
ENV PYTHONPATH=/app

CMD ["python", "vader_service/main.py"]
