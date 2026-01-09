FROM python:3.9-slim

WORKDIR /app

COPY emotion_analysis_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module
COPY shared /app/shared

COPY emotion_analysis_service /app/emotion_analysis_service

ENV PYTHONPATH=/app
CMD ["python", "emotion_analysis_service/main.py"]
