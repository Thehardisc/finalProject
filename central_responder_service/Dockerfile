FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

COPY central_responder_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module
COPY shared/ /app/shared/

# Copy service code
COPY central_responder_service/ /app/central_responder_service/

# Copy LSTM training pipeline (used by trainer_service to retrain ConversationLSTM)
COPY conversation_state_learner/ /app/conversation_state_learner/

ENV PYTHONPATH=/app:/app/central_responder_service
ENV PYTHONUNBUFFERED=1

CMD ["python", "central_responder_service/main.py"]
