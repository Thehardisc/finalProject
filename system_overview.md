# System Overview & Architecture

## 1. Overview
The Emotion Analysis System is a microservices-based application that analyzes emotions in text conversations using a **Meta-Learner ensemble pipeline**. It uses an event-driven architecture powered by Redis Streams to process messages asynchronously through 11 containerized services: ingestion, preprocessing, parallel NLP analysis, ML-based aggregation, state tracking, and persistence.

## 2. Microservices

### **Ingestion Service** (FastAPI :8000)
- Entry point for raw messages via REST API.
- Publishes to Redis `message_stream`.

### **Preprocessing Service**
- Normalizes text (lowercasing, whitespace cleanup).
- Generates dual payloads: raw text (for VADER) and demojized text (for BERT models).
- Publishes to `preprocessed_stream`.

### **VADER Service**
- Lexicon-based sentiment analysis.
- Outputs: `vader_neg`, `vader_neu`, `vader_pos`, `vader_compound`.

### **Basic BERT Service**
- Transformer model (`j-hartmann/emotion-english-distilroberta-base`).
- Outputs 7 Ekman emotion scores: anger, disgust, fear, joy, neutral, sadness, surprise.

### **GoEmotions Service**
- Transformer model (`bhadresh-savani/bert-base-go-emotion`).
- Outputs 28-class emotion probability distribution.

### **EmojiNet Service**
- Embedded emoji knowledge base with emotion/sentiment/sarcasm mappings.
- Handles slang interpretations (e.g., 💀 = "dying of laughter").

### **Central Responder Service**
- Collects partial results from all 4 analyzers via Redis hash aggregation.
- Builds a **67-dimension feature vector** from combined model outputs.
- Runs **Logistic Regression inference** (Meta-Learner) to produce the final dominant emotion and full probability distribution.
- Includes a **background retraining daemon** that periodically retrains on GoEmotions data with 4-layer data filtering and atomic hot-reload.

### **Aggregation Service**
- Tracks conversation state over time (valence, mood trajectory).
- Supports dynamic user-defined rules (e.g., "when I say X it means Y").
- Publishes enriched events to `conversation_update_stream`.

### **Persistence Service**
- Writes messages, emotion analyses, and conversation states to PostgreSQL.

### **API Service** (FastAPI :8001)
- WebSocket server for real-time frontend updates.
- REST endpoints for conversation history (CQRS read path from PostgreSQL).
- Background Redis listener broadcasts analysis results to connected clients.

### **Frontend Service** (React + Nginx :5173)
- Real-time chat interface with live emotion analysis display.

## 3. Data Flow
1. **User Input** → Frontend → WebSocket → API Service → Redis `message_stream`
2. **Preprocessing** normalizes text → Redis `preprocessed_stream`
3. **4 Analyzers run in parallel** (VADER, BERT, GoEmotions, EmojiNet) → Redis `partial_analysis_stream`
4. **Central Responder** waits for all 4, builds feature vector, runs Meta-Learner → Redis `emotion_stream`
5. **Aggregation** updates conversation state → Redis `conversation_update_stream`
6. **Persistence** writes to PostgreSQL; **API** broadcasts to WebSocket clients
7. **Frontend** renders real-time emotion analysis

## 4. Meta-Learner Feature Vector (67 dimensions)

| Block | Source | Dimensions |
|-------|--------|-----------|
| VADER | neg, neu, pos, compound | 4 |
| Basic BERT | 7 Ekman emotions | 7 |
| GoEmotions | 28 emotion scores | 28 |
| EmojiNet | 28 emoji-derived scores | 28 |
| **Total** | | **67** |

## 5. Key Technologies
- **Docker & Docker Compose**: Container orchestration with health checks and dependency ordering.
- **Redis 7**: Message broker (Streams), temporary state (Hashes), consumer groups.
- **PostgreSQL 15**: Long-term persistence.
- **Python 3.9 (FastAPI/Asyncio)**: All backend services.
- **PyTorch + HuggingFace Transformers**: NLP model inference.
- **scikit-learn**: Meta-Learner (Logistic Regression pipeline with StandardScaler).
- **React + Vite**: Frontend SPA.

## 6. Cross-Platform Deployment
- **Mac/CPU**: `bash start.sh` — auto-detects no GPU, runs in CPU mode.
- **Linux/Windows + NVIDIA**: `bash start.sh` or `start.bat` — auto-detects `nvidia-smi`, injects `docker-compose.gpu.yml` for GPU acceleration.
