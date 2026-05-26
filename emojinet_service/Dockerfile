FROM python:3.9-slim

WORKDIR /app

COPY emojinet_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module
COPY shared/ /app/shared/

# Copy service code
COPY emojinet_service/ /app/emojinet_service/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
  CMD kill -0 1 || exit 1

CMD ["python", "emojinet_service/main.py"]
