FROM python:3.9-slim

WORKDIR /app

COPY central_responder_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared code
COPY shared/ /app/shared/

# Copy service code
COPY central_responder_service/ /app/central_responder_service/

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "central_responder_service/main.py"]
