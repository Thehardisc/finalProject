# System Overview & Architecture

## 1. Overview
The Emotion Analysis System is a microservices-based application designed to analyze emotions in text conversations. It uses an event-driven architecture powered by Redis Streams to process messages asynchronously through various stages: ingestion, preprocessing, analysis, aggregation, and persistence.

## 2. Current Microservices
The system consists of the following dockerized services:

### **Ingestion Service**
- **Role**: Entry point for raw messages.
- **Function**: Receives messages (e.g., via API), pushes them to a Redis Stream for processing.

### **Preprocessing Service**
- **Role**: Cleans and normalizes text.
- **Function**: Performs "demojization" (converting emojis to text), lowercasing, and other text normalization steps. Publishes to `preprocessed_stream`.

### **Emotion Analysis Service (To Be Refactored)**
- **Role**: The core intelligence engine.
- **Function**: Currently runs an ensemble of models (VADER, GoEmotions, BERT) in a single process to detect emotions.
- **Refactor Plan**: This will be split into independent containers for each model to improve scalability and isolation.

### **Aggregation Service**
- **Role**: Conversation state tracker.
- **Function**: Tracks emotional arcs over time (Valence/Arousal), detects dynamic rules (e.g., "when I say X I mean Y"), and determines overall mood.

### **Persistence Service**
- **Role**: Data storage.
- **Function**: Saves analysis results and conversation history to PostgreSQL.

### **API Service**
- **Role**: External interface.
- **Function**: REST/WebSocket API for frontend communication.

### **Frontend Service**
- **Role**: User Interface.
- **Function**: React-based UI to display chat, real-time emotion analysis, and debugging logs.

## 3. Data Flow
1. **User Input** -> Frontend -> API -> Ingestion Service
2. **Ingestion Service** -> Redis Stream (`raw_messages`) -> Preprocessing Service
3. **Preprocessing Service** -> Redis Stream (`preprocessed_stream`) -> Analyzers
4. **Analyzers** -> Micro-streams -> **Central Responder** (New Component)
5. **Central Responder** -> Redis Stream (`emotion_stream`) -> Aggregation Service
6. **Aggregation Service** -> `conversation_update_stream` -> Persistence / API

## 4. Key Technologies
- **Docker & Docker Compose**: Orchestration.
- **Redis**: Message broker and temporary state (Streams, Hashes).
- **PostgreSQL**: Long-term storage.
- **Python (FastAPI/Asyncio)**: Backend logic.
- **React**: Frontend.

## 5. Emotion Modeling
The system currently maps emotions to a subset of:
- **GoEmotions (27 classes)**: `admiration`, `amusement`, `anger`, `annoyance`, `approval`, `caring`, `confusion`, `curiosity`, `desire`, `disappointment`, `disapproval`, `disgust`, `embarrassment`, `excitement`, `fear`, `gratitude`, `grief`, `joy`, `love`, `nervousness`, `optimism`, `pride`, `realization`, `relief`, `remorse`, `sadness`, `surprise` + `neutral`.
- **VADER**: Valence (Positive, Negative, Neutral).
- **Basic BERT**: 7 basic emotions (ekman).

## 6. Proposed New Architecture (Task)
To improve modularity and configurability, the analysis layer is being split:
- **VADER Container**: Dedicated sentiment analysis.
- **BERT Container**: Dedicated Transformer-based analysis.
- **GoEmotions Container**: Dedicated fine-grained emotion analysis.
- **Central Responder**: A new orchestrator that:
  - Collects results from all 3 containers.
  - Applies a Weighted Ensemble logic (configurable via JSON).
  - Resolves conflicts (e.g., Sarcasm detection).
  - Outputs the final "Truth" emotion.
