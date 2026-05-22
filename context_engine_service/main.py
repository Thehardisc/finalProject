import asyncio
import json
import time
import uuid
import os
import numpy as np
from typing import Dict, List

from fastapi import FastAPI
import redis.asyncio as aioredis
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContextEngineService")

app = FastAPI()

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")

class ContextEngineService:
    def __init__(self, redis_url: str, qdrant_url: str):
        self.redis = aioredis.from_url(redis_url, decode_responses=True)
        self.qdrant = AsyncQdrantClient(url=qdrant_url)
        self.collection_name = "episodic_memory"
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_size = 384
        
        # Hyperparameters for Working Memory
        self.decay_factor = 0.95
        self.window_size_seconds = 3600  # 1 Hour sliding window

    async def initialize(self):
        """Ensure Qdrant collection exists and is properly indexed."""
        try:
            collections = await self.qdrant.get_collections()
            if not any(c.name == self.collection_name for c in collections.collections):
                await self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE)
                )
                # Create payload index for user_id
                await self.qdrant.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="user_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                logger.info("Qdrant collection and indexes initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")

    async def calculate_user_baseline(self, user_id: str, current_embedding: List[float] = None) -> Dict[str, float]:
        """
        Fetches long-term Episodic Memory from Qdrant.
        Finds the historical baseline valence for similar semantic topics.
        """
        if not current_embedding or not user_id:
            return {"historical_valence": 0.0, "topic_resonance": 0.0}

        try:
            response = await self.qdrant.query_points(
                collection_name=self.collection_name,
                query=current_embedding,
                query_filter=models.Filter(
                    must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
                ),
                limit=5
            )
            search_result = response.points

            if not search_result:
                return {"historical_valence": 0.0, "topic_resonance": 0.0}

            total_score = sum(hit.score for hit in search_result)
            if total_score == 0:
                return {"historical_valence": 0.0, "topic_resonance": 0.0}

            weighted_valence = sum((hit.payload.get("valence", 0) * (hit.score / total_score)) for hit in search_result)
            avg_resonance = total_score / len(search_result)

            return {
                "historical_valence": float(weighted_valence),
                "topic_resonance": float(avg_resonance)
            }
        except Exception as e:
            logger.warning(f"Failed to fetch baseline from Qdrant: {e}")
            return {"historical_valence": 0.0, "topic_resonance": 0.0}

    async def update_working_memory(self, user_id: str, current_valence: float, linguistic_markers: Dict) -> float:
        """
        Updates Redis with current state and calculates exponential volatility.
        """
        if not user_id:
            return 0.0
            
        now = time.time()
        window_cutoff = now - self.window_size_seconds
        
        state_key = f"context:user:{user_id}:state"
        valence_key = f"context:user:{user_id}:valence_window"
        linguistic_key = f"context:user:{user_id}:linguistic_window"

        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.zremrangebyscore(valence_key, "-inf", window_cutoff)
                pipe.zremrangebyscore(linguistic_key, "-inf", window_cutoff)
                
                valence_data = json.dumps({"valence": current_valence, "id": str(uuid.uuid4())})
                pipe.zadd(valence_key, {valence_data: now})
                
                linguistic_data = json.dumps(linguistic_markers)
                pipe.zadd(linguistic_key, {linguistic_data: now})
                
                pipe.zrange(valence_key, 0, -1)
                pipe.hgetall(state_key)
                
                results = await pipe.execute()

            history_raw = results[4]  
            prev_state = results[5]   

            recent_valences = [json.loads(x)["valence"] for x in history_raw]
            current_variance = np.var(recent_valences) if len(recent_valences) > 1 else 0.0
            
            prev_volatility = float(prev_state.get("current_volatility", 0.0))
            new_volatility = (self.decay_factor * prev_volatility) + ((1 - self.decay_factor) * current_variance)

            await self.redis.hset(state_key, mapping={
                "current_volatility": new_volatility,
                "last_message_ts": now,
                "baseline_valence": np.mean(recent_valences) if recent_valences else current_valence
            })

            return new_volatility
        except Exception as e:
            logger.error(f"Redis memory update failed: {e}")
            return 0.0

    async def build_115_dim_context_vector(self, user_id: str, current_valence: float, linguistic_markers: Dict, current_embedding: List[float] = None) -> List[float]:
        """
        Builds the 48-dimension contextual vector to append to the 67-dim base vector.
        """
        baseline_task = self.calculate_user_baseline(user_id, current_embedding)
        memory_task = self.update_working_memory(user_id, current_valence, linguistic_markers)
        
        baseline_data, volatility = await asyncio.gather(baseline_task, memory_task)

        context_features = np.zeros(48, dtype=float)
        context_features[0] = baseline_data["historical_valence"]
        context_features[1] = baseline_data["topic_resonance"]
        context_features[2] = volatility
        context_features[3] = current_valence
        context_features[4] = float(linguistic_markers.get("length", 0))
        context_features[5] = float(linguistic_markers.get("latency_ms", 0))
        
        if current_embedding and len(current_embedding) >= 42:
            context_features[6:48] = current_embedding[:42]

        return context_features.tolist()

    async def save_to_episodic_memory(self, user_id: str, text: str, valence: float, current_embedding: List[float], dominant_emotion: str):
        """Save this interaction to Qdrant for future memory."""
        if not user_id or not current_embedding:
            return
            
        try:
            point_id = str(uuid.uuid4())
            await self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=current_embedding,
                        payload={
                            "user_id": user_id,
                            "timestamp": time.time(),
                            "valence": valence,
                            "dominant_emotion": dominant_emotion,
                            "text": text
                        }
                    )
                ]
            )
        except Exception as e:
            logger.error(f"Failed to save to Qdrant: {e}")

context_engine = ContextEngineService(redis_url=f"redis://{REDIS_HOST}:6379", qdrant_url=f"http://{QDRANT_HOST}:6333")

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Context Engine...")
    await context_engine.initialize()
    asyncio.create_task(redis_listener())

async def redis_listener():
    """Listen to preprocessed_stream and publish named context features."""
    group_name  = "context_engine_group"
    client_id   = f"context_engine_{uuid.uuid4().hex[:6]}"
    stream_name = "preprocessed_stream"

    try:
        await context_engine.redis.xgroup_create(stream_name, group_name, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Group creation error: {e}")

    logger.info(f"Context Engine listening on {stream_name}...")

    while True:
        try:
            messages = await context_engine.redis.xreadgroup(
                group_name, client_id, {stream_name: ">"}, count=10, block=2000
            )
            if messages:
                for stream, msgs in messages:
                    for msg_id, msg_data in msgs:
                        try:
                            text            = msg_data.get("text", "")
                            user_id         = msg_data.get("user_id", "anonymous")
                            conversation_id = msg_data.get("conversation_id", "")
                            raw_msg_id      = msg_data.get("message_id", "")

                            # Read REAL conversation state written by aggregation_service
                            # for the previous message in this conversation.
                            state        = await context_engine.redis.hgetall(f"conversation:{conversation_id}")
                            prev_valence = float(state.get("average_valence", 0.0)) if state else 0.0
                            prev_emotion = (state.get("dominant_emotion", "neutral") if state else "neutral")

                            embedding = context_engine.embedder.encode(text).tolist()

                            linguistic_markers = {"length": len(text), "latency_ms": 0}

                            # Run episodic-memory lookup and working-memory update in parallel
                            baseline_data, volatility = await asyncio.gather(
                                context_engine.calculate_user_baseline(user_id, embedding),
                                context_engine.update_working_memory(user_id, prev_valence, linguistic_markers),
                            )

                            # Persist to Qdrant with REAL valence and emotion (not hardcoded 0/"neutral")
                            asyncio.create_task(
                                context_engine.save_to_episodic_memory(
                                    user_id, text, prev_valence, embedding, prev_emotion
                                )
                            )

                            # Send named scalar features — central_responder uses these directly
                            # without needing to parse a raw 48-dim vector.
                            payload = {
                                "message_id": raw_msg_id,
                                "model_name": "context_engine",
                                "scores": json.dumps({
                                    "historical_valence": baseline_data["historical_valence"],
                                    "topic_resonance":    baseline_data["topic_resonance"],
                                    "volatility":         float(volatility),
                                    "prev_valence":       prev_valence,
                                }),
                                "original_data": json.dumps(msg_data),
                            }

                            await context_engine.redis.xadd("partial_analysis_stream", payload)
                            await context_engine.redis.xack(stream_name, group_name, msg_id)

                        except Exception as msg_err:
                            logger.error(f"Context engine failed on {msg_id}: {msg_err}")
                            try:
                                await context_engine.redis.xack(stream_name, group_name, msg_id)
                            except Exception:
                                pass

        except Exception as e:
            logger.error(f"Redis listener error: {e}")
            await asyncio.sleep(1)

@app.get("/health")
def health():
    return {"status": "ok", "service": "context_engine"}
