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

from shared.utils.logger import get_logger
from shared.constants import (
    CONTEXT_DIM, CDM_N_STATES,
    CTX_CDM_PROBS, CTX_RESIDENCY, CTX_TRANSITION, CTX_ABRUPTNESS,
    CTX_COHERENCE, CTX_ENTROPY, CTX_SPK_DIVERGENCE, CTX_VELOCITY,
    CTX_ACCELERATION, CTX_HIST_VALENCE, CTX_RESONANCE, CTX_VOLATILITY,
    CTX_CURR_VALENCE, CTX_MSG_LENGTH, CTX_LATENCY_MS,
)
from cdm import CDM, observation_from_emotions

logger = get_logger("context_engine")

# Residency normalisation: how many consecutive turns in the same top-1 state
# saturates the residency feature to 1.0.
_RESIDENCY_CAP = 10.0

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

    def _belief_key(self, conversation_id: str) -> str:
        return f"conv:{conversation_id}:cdm_belief"

    def _load_belief(self, raw: str):
        """
        Deserialise the persisted CDM belief blob.

        Returns (belief: np.ndarray[28], top1_idx: int, residency_count: int,
                 last_valence: float). Falls back to a uniform cold-start belief
        on any missing/malformed data.
        """
        if raw:
            try:
                blob = json.loads(raw)
                belief = np.asarray(blob.get("belief", []), dtype=np.float64)
                if belief.shape == (CDM_N_STATES,):
                    return (
                        belief,
                        int(blob.get("top1_idx", -1)),
                        int(blob.get("residency_count", 0)),
                        float(blob.get("last_valence", 0.0)),
                    )
            except Exception:
                pass
        return CDM.uniform_belief(), -1, 0, 0.0

    def _diag_payload(self, diag, residency_norm: float) -> dict:
        """Compact, JSON-serialisable view of a PBSM transition for the frontend."""
        return {
            "cdm_available": True,
            "belief":        [round(p, 5) for p in diag.belief],
            "top3_states":   diag.top3_states,
            "top3_probs":    [round(p, 5) for p in diag.top3_probs],
            "top1_name":     diag.top1_name,
            "top2_name":     diag.top2_name,
            "top3_name":     diag.top3_name,
            "momentum":      round(diag.state_momentum, 5),
            "abruptness":    round(diag.entry_abruptness, 5),
            "catalyst":      diag.catalyst_state,
            "catalyst_obs":  diag.catalyst_obs,
            "residency":     round(residency_norm, 5),
        }

    async def build_cdm_context_vector(
        self,
        conversation_id: str,
        user_id: str,
        current_valence: float,
        last_emotions: Dict,
        linguistic_markers: Dict,
        current_embedding: List[float] = None,
    ):
        """
        Build the 44-dim CDM context vector and rich diagnostics.

        The Parallel Belief State Machine advances one step from the persisted
        belief using the *previous* turn's emotion distribution as the
        observation (the current turn's emotions don't exist yet at
        preprocessing time). Episodic + working memory fill the scalar tail.

        Returns (context_vector: List[float], cdm_diag: dict).
        """
        # Episodic memory + working-memory volatility, in parallel
        baseline_data, volatility = await asyncio.gather(
            self.calculate_user_baseline(user_id, current_embedding),
            self.update_working_memory(user_id, current_valence, linguistic_markers),
        )

        # ── PBSM belief update ──────────────────────────────────────────────
        belief_key = self._belief_key(conversation_id)
        prev_belief, prev_top1, residency_count, last_valence = self._load_belief(
            await self.redis.get(belief_key)
        )
        residency_norm = min(residency_count / _RESIDENCY_CAP, 1.0)

        observation = observation_from_emotions(last_emotions or {})
        diag = CDM.transition(prev_belief, observation, residency_norm)

        new_top1 = diag.top3_states[0]
        new_residency_count = residency_count + 1 if new_top1 == prev_top1 else 0
        new_residency_norm  = min(new_residency_count / _RESIDENCY_CAP, 1.0)

        # ── Assemble the 44-dim context vector ──────────────────────────────
        ctx = np.zeros(CONTEXT_DIM, dtype=float)
        ctx[CTX_CDM_PROBS]      = diag.belief
        ctx[CTX_RESIDENCY]      = new_residency_norm
        ctx[CTX_TRANSITION]     = [s / (CDM_N_STATES - 1) for s in diag.top3_states]
        ctx[CTX_ABRUPTNESS]     = diag.entry_abruptness
        ctx[CTX_COHERENCE]      = baseline_data["topic_resonance"]
        ctx[CTX_ENTROPY]        = self._entropy(observation)
        ctx[CTX_SPK_DIVERGENCE] = 0.0
        ctx[CTX_VELOCITY]       = float(np.clip(current_valence - last_valence, -1.0, 1.0))
        ctx[CTX_ACCELERATION]   = 0.0
        ctx[CTX_HIST_VALENCE]   = baseline_data["historical_valence"]
        ctx[CTX_RESONANCE]      = baseline_data["topic_resonance"]
        ctx[CTX_VOLATILITY]     = float(volatility)
        ctx[CTX_CURR_VALENCE]   = current_valence
        ctx[CTX_MSG_LENGTH]     = min(float(linguistic_markers.get("length", 0)) / 280.0, 1.0)
        ctx[CTX_LATENCY_MS]     = min(float(linguistic_markers.get("latency_ms", 0)) / 60000.0, 1.0)

        # Persist the advanced belief for the next turn
        await self.redis.set(belief_key, json.dumps({
            "belief":          [float(p) for p in diag.belief],
            "top1_idx":        int(new_top1),
            "residency_count": new_residency_count,
            "last_valence":    current_valence,
        }), ex=86400 * 7)

        return ctx.tolist(), self._diag_payload(diag, new_residency_norm)

    @staticmethod
    def _entropy(dist: np.ndarray) -> float:
        """Shannon entropy of a distribution, normalised to [0, 1]."""
        p = np.asarray(dist, dtype=np.float64)
        s = p.sum()
        if s <= 0:
            return 1.0
        p = p / s
        nz = p[p > 0]
        h = float(-np.sum(nz * np.log(nz)))
        return h / np.log(len(p)) if len(p) > 1 else 0.0

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
            logger.error("xgroup_create failed", extra={"stream": stream_name, "error": str(e)})

    logger.info("context_engine_listening", extra={"event": "stream_listen_start", "stream": stream_name, "group": group_name})

    while True:
        try:
            messages = await context_engine.redis.xreadgroup(
                group_name, client_id, {stream_name: ">"}, count=10, block=2000
            )
            if messages:
                for stream, msgs in messages:
                    for msg_id, msg_data in msgs:
                        text            = msg_data.get("text", "")
                        user_id         = msg_data.get("user_id", "anonymous")
                        conversation_id = msg_data.get("conversation_id", "")
                        raw_msg_id      = msg_data.get("message_id", "")
                        mlog = logger.bind(message_id=raw_msg_id, conversation_id=conversation_id, user_id=user_id)
                        try:

                            # Read REAL conversation state written by aggregation_service
                            # for the previous message in this conversation.
                            state        = await context_engine.redis.hgetall(f"conversation:{conversation_id}")
                            prev_valence = float(state.get("average_valence", 0.0)) if state else 0.0
                            prev_emotion = (state.get("dominant_emotion", "neutral") if state else "neutral")

                            # Previous turn's full GoEmotions distribution is the PBSM
                            # observation (the current turn's emotions don't exist yet).
                            last_emotions = {}
                            if state and state.get("last_message_emotions"):
                                try:
                                    last_emotions = json.loads(state["last_message_emotions"])
                                except Exception:
                                    last_emotions = {}

                            embedding = context_engine.embedder.encode(text).tolist()

                            linguistic_markers = {"length": len(text), "latency_ms": 0}

                            # Advance the PBSM and build the 44-dim CDM context vector.
                            context_vector, cdm_diag = await context_engine.build_cdm_context_vector(
                                conversation_id, user_id, prev_valence,
                                last_emotions, linguistic_markers, embedding,
                            )

                            # Persist to Qdrant with REAL valence and emotion (not hardcoded 0/"neutral")
                            asyncio.create_task(
                                context_engine.save_to_episodic_memory(
                                    user_id, text, prev_valence, embedding, prev_emotion
                                )
                            )

                            mlog.warning(
                                "[DIAG-1-CTX-PUBLISH] "
                                f"dim={len(context_vector)} "
                                f"belief_sum={sum(context_vector[0:CDM_N_STATES]):.4f} "
                                f"top3={cdm_diag.get('top3_states')}"
                            )

                            # Publish the CDM context vector + rich diagnostics.
                            # central_responder injects context_vector into the
                            # feature vector's [67:111] block and forwards cdm_diag.
                            payload = {
                                "message_id": raw_msg_id,
                                "model_name": "context_engine",
                                "context_vector": json.dumps(context_vector),
                                "cdm_diag": json.dumps(cdm_diag),
                                "scores": json.dumps({
                                    "historical_valence": context_vector[CTX_HIST_VALENCE],
                                    "topic_resonance":    context_vector[CTX_RESONANCE],
                                    "volatility":         context_vector[CTX_VOLATILITY],
                                    "prev_valence":       prev_valence,
                                }),
                                "original_data": json.dumps(msg_data),
                            }

                            await context_engine.redis.xadd("partial_analysis_stream", payload)
                            await context_engine.redis.xack(stream_name, group_name, msg_id)

                        except Exception as msg_err:
                            mlog.error(
                                "context_engine_message_failed",
                                extra={"event": "ce_message_failed", "error_class": type(msg_err).__name__, "error": str(msg_err)},
                            )
                            try:
                                await context_engine.redis.xack(stream_name, group_name, msg_id)
                            except Exception as ack_err:
                                mlog.warning(
                                    "xack_failed_after_error",
                                    extra={"event": "xack_failed", "error": str(ack_err)},
                                )

        except Exception as e:
            logger.error("redis_listener_loop_error", extra={"event": "ce_listener_error", "error": str(e)})
            await asyncio.sleep(1)

@app.get("/health")
def health():
    return {"status": "ok", "service": "context_engine"}
