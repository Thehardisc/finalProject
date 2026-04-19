FROM python:3.9-slim

WORKDIR /app

# Context is project root, so we copy from subfolder
COPY emojinet_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module
COPY shared /app/shared

# Copy service code
# Copy service code
COPY emojinet_service /app

ENV PYTHONUNBUFFERED=1

# Health check verifies the EmojiNet analysis process is active
HEALTHCHECK --interval=30s --timeout=15s --start-period=5s --retries=3 \
  CMD ps aux | grep "python main.py" || exit 1

CMD ["python", "main.py"]
