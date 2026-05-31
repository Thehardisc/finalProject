"""
context_engine_service/main.py — Context Engine Service (CDM edition).

Listens to preprocessed_stream, builds a 23-dim context vector per message,
and publishes it to partial_analysis_stream for the central_responder_service.

Context vector layout (23 dims — must match shared/constants.py CONTEXT_DIM):
  [0:7]    CDM state probabilities (7 latent conversation states)
  [7]      state_residency         (consecutive messages in current state, normalised)
  [8:11]   transition_path         (last 3 state indices / 7)
  [11]     entry_abruptness        (fractional state-index jump, 0-1)
  [12]     topic_coherence         (cosine similarity vs previous message embedding)
  [13]     emotion_entropy         (Shannon entropy of previous-turn GoEmotions dist.)
  [14]     speaker_divergence      (std of per-speaker EMA valences)
  [15]     velocity                (Δvalence vs previous message)
  [16]     acceleration            (Δ²valence — is the shift intensifying?)
  [17]     historical_valence      (Qdrant episodic memory, weighted by topic similarity)
  [18]     topic_resonance         (Qdrant average cosine score)
  [19]     volatility              (EMA of variance, user working-memory window)
  [20]     current_valence         (EMA valence of whole conversation)
  [21]     message_length          (char count)
  [22]     latency_ms              (time since previous message)
"""

import asyncio
import json
import sys
import time
import uuid
import os
import numpy as np
from typing import Dict, List

# Make cdm.py importable from any launch directory (project root or service dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdm import CDM
from shared.module_registry import register_module
from shared.constants import (
    CONTEXT_DIM,
    CTX_HIST_VALENCE, CTX_RESONANCE, CTX_VOLATILITY, CTX_CURR_VALENCE,
)

from fastapi import FastAPI
import redis.asyncio as aioredis
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContextEngineService")

app = FastAPI()

REDIS_HOST  = os.getenv("REDIS_HOST",  "redis")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
STREAM_MAXLEN = int(os.getenv("REDIS_STREAM_MAXLEN", 10_000))

# GoEmotions label order — must match shared/constants.py EMOTION_LABELS
_GOEMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
_N_GOEMO = len(_GOEMOTION_LABELS)  # 28


def _emotion_entropy(emotions: Dict) -> float:
    """Shannon entropy (nats) of a GoEmotions probability distribution."""
    probs = np.array([float(emotions.get(e, 0.0)) for e in _GOEMOTION_LABELS])
    s = probs.sum()
    probs = probs / s if s > 0 else np.ones(_N_GOEMO) / _N_GOEMO
    probs_nz = probs[probs > 0]
    return float(-np.sum(probs_nz * np.log(probs_nz)))


class ContextEngineService:
    def __init__(self, redis_url: str, qdrant_url: str):
        self.redis  = aioredis.from_url(redis_url, decode_responses=True)
        self.qdrant = AsyncQdrantClient(url=qdrant_url)
        self.collection_name = "episodic_memory"
        self.embedder    = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_size = 384

        self.decay_factor       = 0.95
        self.window_size_seconds = 3600  # 1-hour sliding window for volatility

    async def initialize(self):
        """Ensure Qdrant collection exists and is properly indexed."""
        try:
            collections = await self.qdrant.get_collections()
            if not any(c.name == self.collection_name for c in collections.collections):
                await self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE)
                )
                await self.qdrant.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="user_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                logger.info("Qdrant collection and indexes initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")

    async def calculate_user_baseline(self, user_id: str, current_embedding: List[float] = None) -> Dict[str, float]:
        """Fetch long-term episodic memory from Qdrant for this user and topic."""
        if not current_embedding or not user_id:
            return {"historical_valence": 0.0, "topic_resonance": 0.0}

        try:
            response = await asyncio.wait_for(
                self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=current_embedding,
                    query_filter=models.Filter(
                        must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
                    ),
                    limit=5,
                ),
                timeout=0.5,  # 500 ms circuit breaker
            )
            search_result = response.points

            if not search_result:
                return {"historical_valence": 0.0, "topic_resonance": 0.0}

            total_score = sum(hit.score for hit in search_result)
            if total_score == 0:
                return {"historical_valence": 0.0, "topic_resonance": 0.0}

            weighted_valence = sum(
                hit.payload.get("valence", 0) * (hit.score / total_score)
                for hit in search_result
            )
            return {
                "historical_valence": float(weighted_valence),
                "topic_resonance":    float(total_score / len(search_result)),
            }
        except asyncio.TimeoutError:
            logger.warning(f"Qdrant query timed out for user {user_id} — using zero baseline")
            return {"historical_valence": 0.0, "topic_resonance": 0.0}
        except Exception as e:
            logger.warning(f"Failed to fetch baseline from Qdrant: {e}")
            return {"historical_valence": 0.0, "topic_resonance": 0.0}

    async def update_working_memory(self, user_id: str, current_valence: float, linguistic_markers: Dict) -> float:
        """Update Redis working-memory window and return EMA volatility."""
        if not user_id:
            return 0.0

        now = time.time()
        window_cutoff   = now - self.window_size_seconds
        state_key       = f"context:user:{user_id}:state"
        valence_key     = f"context:user:{user_id}:valence_window"
        linguistic_key  = f"context:user:{user_id}:linguistic_window"

        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.zremrangebyscore(valence_key,    "-inf", window_cutoff)
                pipe.zremrangebyscore(linguistic_key, "-inf", window_cutoff)
                pipe.zadd(valence_key,    {json.dumps({"valence": current_valence, "id": str(uuid.uuid4())}): now})
                pipe.zadd(linguistic_key, {json.dumps(linguistic_markers): now})
                pipe.zrange(valence_key, 0, -1)
                pipe.hgetall(state_key)
                results = await pipe.execute()

            history_raw = results[4]
            prev_state  = results[5]

            recent_valences = [json.loads(x)["valence"] for x in history_raw]
            current_variance = np.var(recent_valences) if len(recent_valences) > 1 else 0.0
            prev_volatility  = float(prev_state.get("current_volatility", 0.0))
            new_volatility   = self.decay_factor * prev_volatility + (1 - self.decay_factor) * current_variance

            await self.redis.hset(state_key, mapping={
                "current_volatility": float(new_volatility),
                "last_message_ts":    now,
                "baseline_valence":   float(np.mean(recent_valences)) if recent_valences else float(current_valence),
            })
            return new_volatility
        except Exception as e:
            logger.error(f"Redis memory update failed: {e}")
            return 0.0

    async def _store_embedding(self, key: str, embedding: List[float]):
        """Persist embedding in Redis for the next message's topic-coherence computation."""
        try:
            await self.redis.set(key, json.dumps(embedding), ex=86400 * 7)
        except Exception as e:
            logger.warning(f"Failed to store embedding at {key}: {e}")

    async def build_context_vector(
        self,
        conversation_id:    str,
        user_id:            str,
        current_valence:    float,
        linguistic_markers: Dict,
        current_embedding:  List[float],
        last_emotions:      Dict,
        is_continuation:    bool = False,
    ) -> List[float]:
        """
        Build the 23-dim CDM context vector for the current message.

        is_continuation=True: rapid-fire fragment of the same conversational turn.
        Dispatches to _build_continuation_vector() which keeps the CDM state as-is,
        zeros velocity/acceleration, and skips Qdrant and valence-history writes.

        Reads Redis state written by aggregation_service for the PREVIOUS message,
        so all features capture momentum and trajectory rather than instantaneous values.
        """
        if is_continuation:
            return await self._build_continuation_vector(
                conversation_id, user_id, linguistic_markers, current_embedding, last_emotions
            )

        # Qdrant + working-memory are independent — run in parallel
        baseline_task = self.calculate_user_baseline(user_id, current_embedding)
        memory_task   = self.update_working_memory(user_id, current_valence, linguistic_markers)
        baseline_data, volatility = await asyncio.gather(baseline_task, memory_task)

        # ── Read conversation hash (written by aggregation_service for T-1) ────
        state = await self.redis.hgetall(f"conversation:{conversation_id}")

        # ── Velocity & Acceleration ────────────────────────────────────────────
        hist_raw = await self.redis.lrange(f"conv:{conversation_id}:valence_hist", 0, 2)
        hist = [float(x) for x in hist_raw]
        velocity     = hist[0] - hist[1] if len(hist) >= 2 else 0.0
        acceleration = (velocity - (hist[1] - hist[2])) if len(hist) >= 3 else 0.0

        # ── Speaker Divergence ─────────────────────────────────────────────────
        spk_vals = []
        for k, v in state.items():
            if k.startswith("spk:"):
                try:
                    spk_vals.append(float(v))
                except ValueError:
                    pass
        speaker_divergence = float(np.std(spk_vals)) if len(spk_vals) >= 2 else 0.0

        # ── Topic Coherence ────────────────────────────────────────────────────
        last_embed_key  = f"conv:{conversation_id}:last_embed"
        last_embed_json = await self.redis.get(last_embed_key)
        topic_coherence = 0.0
        if last_embed_json and current_embedding:
            try:
                le    = np.array(json.loads(last_embed_json), dtype=np.float32)
                ce    = np.array(current_embedding,           dtype=np.float32)
                denom = np.linalg.norm(le) * np.linalg.norm(ce)
                if denom > 0:
                    topic_coherence = float(np.dot(le, ce) / denom)
            except Exception:
                pass
        asyncio.create_task(self._store_embedding(last_embed_key, current_embedding))

        # ── Emotion Entropy (previous turn's GoEmotions distribution) ─────────
        emotion_entropy = _emotion_entropy(last_emotions)

        # ── CDM DFSM ───────────────────────────────────────────────────────────
        state_hist_key  = f"conv:{conversation_id}:state_hist"
        cdm_state_key   = f"conv:{conversation_id}:cdm_state"

        raw_prev = await self.redis.get(cdm_state_key)
        prev_state_idx = int(raw_prev) if raw_prev is not None else 0

        current_state_idx = CDM.transition(
            prev_state_idx, velocity, emotion_entropy, speaker_divergence, topic_coherence
        )
        cdm_vec = CDM.one_hot(current_state_idx)

        # ── State Residency & Transition Path ──────────────────────────────────
        state_hist_raw = await self.redis.lrange(state_hist_key, 0, 2)
        state_hist     = [int(x) for x in state_hist_raw]

        residency = 0
        for s in state_hist:
            if s == current_state_idx:
                residency += 1
            else:
                break
        state_residency  = min(residency / 10.0, 1.0)
        entry_abruptness = CDM.entry_abruptness(prev_state_idx, current_state_idx)

        transition_path = [s / 7.0 for s in state_hist[:3]]
        while len(transition_path) < 3:
            transition_path.append(current_state_idx / 7.0)

        await self.redis.set(cdm_state_key, str(current_state_idx), ex=86400 * 7)
        await self.redis.lpush(state_hist_key, str(current_state_idx))
        await self.redis.ltrim(state_hist_key, 0, 9)
        await self.redis.expire(state_hist_key, 86400 * 7)

        ctx = np.zeros(CONTEXT_DIM, dtype=np.float64)
        ctx[0:7]  = cdm_vec
        ctx[7]    = state_residency
        ctx[8:11] = transition_path[:3]
        ctx[11]   = entry_abruptness
        ctx[12]   = topic_coherence
        ctx[13]   = emotion_entropy
        ctx[14]   = speaker_divergence
        ctx[15]   = velocity
        ctx[16]   = acceleration
        ctx[17]   = baseline_data["historical_valence"]
        ctx[18]   = baseline_data["topic_resonance"]
        ctx[19]   = volatility
        ctx[20]   = current_valence
        ctx[21]   = float(linguistic_markers.get("length", 0))
        ctx[22]   = float(linguistic_markers.get("latency_ms", 0))
        return ctx.tolist()

    async def _build_continuation_vector(
        self,
        conversation_id:    str,
        user_id:            str,
        linguistic_markers: Dict,
        current_embedding:  List[float],
        last_emotions:      Dict,
    ) -> List[float]:
        """
        Lightweight 23-dim vector for is_continuation=True fragments.

        Kept  (semantic content still changes per fragment):
          topic_coherence, emotion_entropy, speaker_divergence, embedding storage.

        Suppressed (prevents DFSM pollution from Enter-key bursts):
          CDM.transition()  — CDM state held as-is
          velocity / acceleration — set to 0.0
          state_hist write  — residency counts the fragment correctly
          valence_hist write — no new emotional turn
          Qdrant call        — skipped (no new episodic anchor)
        """
        cdm_state_key  = f"conv:{conversation_id}:cdm_state"
        state_hist_key = f"conv:{conversation_id}:state_hist"

        raw_state     = await self.redis.get(cdm_state_key)
        current_state = int(raw_state) if raw_state is not None else 0
        cdm_vec       = CDM.one_hot(current_state)

        state_hist_raw = await self.redis.lrange(state_hist_key, 0, 2)
        state_hist     = [int(x) for x in state_hist_raw]
        # Count residency including the current fragment
        residency = 1
        for s in state_hist:
            if s == current_state:
                residency += 1
            else:
                break
        state_residency = min(residency / 10.0, 1.0)

        transition_path = [s / 7.0 for s in state_hist[:3]]
        while len(transition_path) < 3:
            transition_path.append(current_state / 7.0)

        # Read existing volatility without writing a new data point
        try:
            user_state = await self.redis.hgetall(f"context:user:{user_id}:state")
            volatility = float(user_state.get("current_volatility", 0.0))
        except Exception:
            volatility = 0.0

        conv_hash = await self.redis.hgetall(f"conversation:{conversation_id}")
        current_ema_valence = float(conv_hash.get("average_valence", 0.0)) if conv_hash else 0.0

        spk_vals = []
        for k, v in conv_hash.items():
            if k.startswith("spk:"):
                try:
                    spk_vals.append(float(v))
                except ValueError:
                    pass
        speaker_divergence = float(np.std(spk_vals)) if len(spk_vals) >= 2 else 0.0

        last_embed_key  = f"conv:{conversation_id}:last_embed"
        last_embed_json = await self.redis.get(last_embed_key)
        topic_coherence = 0.0
        if last_embed_json and current_embedding:
            try:
                le    = np.array(json.loads(last_embed_json), dtype=np.float32)
                ce    = np.array(current_embedding,           dtype=np.float32)
                denom = np.linalg.norm(le) * np.linalg.norm(ce)
                if denom > 0:
                    topic_coherence = float(np.dot(le, ce) / denom)
            except Exception:
                pass
        asyncio.create_task(self._store_embedding(last_embed_key, current_embedding))

        emotion_entropy = _emotion_entropy(last_emotions)

        ctx = np.zeros(CONTEXT_DIM, dtype=np.float64)
        ctx[0:7]  = cdm_vec
        ctx[7]    = state_residency
        ctx[8:11] = transition_path[:3]
        ctx[11]   = 0.0              # entry_abruptness: no transition
        ctx[12]   = topic_coherence
        ctx[13]   = emotion_entropy
        ctx[14]   = speaker_divergence
        ctx[15]   = 0.0              # velocity: no new turn
        ctx[16]   = 0.0              # acceleration: no new turn
        ctx[17]   = 0.0              # historical_valence: Qdrant skipped
        ctx[18]   = 0.0              # topic_resonance: Qdrant skipped
        ctx[19]   = volatility
        ctx[20]   = current_ema_valence
        ctx[21]   = float(linguistic_markers.get("length", 0))
        ctx[22]   = float(linguistic_markers.get("latency_ms", 0))
        return ctx.tolist()

    async def save_to_episodic_memory(
        self, user_id: str, text: str, valence: float,
        current_embedding: List[float], dominant_emotion: str
    ):
        """Save this interaction to Qdrant for future episodic memory retrieval."""
        if not user_id or not current_embedding:
            return

        try:
            await self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=current_embedding,
                        payload={
                            "user_id":          user_id,
                            "timestamp":        time.time(),
                            "valence":          valence,
                            "dominant_emotion": dominant_emotion,
                            "text":             text,
                        }
                    )
                ]
            )
        except Exception as e:
            logger.error(f"Failed to save to Qdrant: {e}")


context_engine = ContextEngineService(
    redis_url=f"redis://{REDIS_HOST}:6379",
    qdrant_url=f"http://{QDRANT_HOST}:6333",
)


@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Context Engine (CDM 23-dim)...")
    await context_engine.initialize()
    await register_module(
        context_engine.redis,
        "context_engine",
        required=False,
        schema="context_vector",
    )
    asyncio.create_task(redis_listener())


async def redis_listener():
    """Read preprocessed_stream, build CDM context vectors, publish to partial_analysis_stream."""
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
            # PEL recovery — reclaim messages idle >30 s (crash-before-ACK protection)
            try:
                _, stale, _ = await context_engine.redis.xautoclaim(
                    stream_name, group_name, client_id,
                    min_idle_time=30_000, start_id='0-0', count=10
                )
                messages = [(stream_name, stale)] if stale else None
            except Exception:
                messages = None

            if not messages:
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

                            # Previous conversation state written by aggregation_service
                            state        = await context_engine.redis.hgetall(f"conversation:{conversation_id}")
                            prev_valence = float(state.get("average_valence", 0.0)) if state else 0.0
                            prev_emotion = state.get("dominant_emotion", "neutral") if state else "neutral"

                            # Previous-turn GoEmotions distribution (for entropy feature)
                            try:
                                last_emotions = json.loads(state.get("last_message_emotions", "{}") or "{}")
                            except json.JSONDecodeError:
                                last_emotions = {}

                            raw_emb = await asyncio.get_running_loop().run_in_executor(
                                None, context_engine.embedder.encode, text
                            )
                            embedding = raw_emb.tolist()

                            linguistic_markers = {
                                "length":     len(text),
                                "latency_ms": float(msg_data.get("latency_ms", 0)),
                            }

                            try:
                                context_vector = await asyncio.wait_for(
                                    context_engine.build_context_vector(
                                        conversation_id,
                                        user_id,
                                        prev_valence,
                                        linguistic_markers,
                                        embedding,
                                        last_emotions,
                                    ),
                                    timeout=1.5,  # overall circuit breaker
                                )
                            except asyncio.TimeoutError:
                                logger.warning(
                                    f"build_context_vector timed out for {raw_msg_id} "
                                    f"— publishing zero {CONTEXT_DIM}-dim vector"
                                )
                                context_vector = [0.0] * CONTEXT_DIM

                            asyncio.create_task(
                                context_engine.save_to_episodic_memory(
                                    user_id, text, prev_valence, embedding, prev_emotion
                                )
                            )

                            # Publish raw vector + named scalars so consumers never
                            # need to guess indices. If CONTEXT_DIM changes, only
                            # shared/constants.py and this block need updating.
                            # ── DIAG-1: prove vector is not all-zeros before publish ──
                            _nz = sum(1 for v in context_vector if v != 0.0)
                            logger.warning(
                                f"[DIAG-1] CE publish msg={raw_msg_id} "
                                f"dim={len(context_vector)} nonzero={_nz} "
                                f"cdm={[round(float(v),2) for v in context_vector[0:7]]} "
                                f"hist_val={float(context_vector[CTX_HIST_VALENCE]):.4f} "
                                f"cur_val={float(context_vector[CTX_CURR_VALENCE]):.4f}"
                            )
                            await context_engine.redis.xadd(
                                "partial_analysis_stream",
                                {
                                    "message_id":          raw_msg_id,
                                    "model_name":          "context_engine",
                                    "context_vector":      json.dumps(context_vector),
                                    "original_data":       json.dumps(msg_data),
                                    "ctx_dim":             str(len(context_vector)),
                                    "ctx_hist_valence":    str(context_vector[CTX_HIST_VALENCE]),
                                    "ctx_topic_resonance": str(context_vector[CTX_RESONANCE]),
                                    "ctx_volatility":      str(context_vector[CTX_VOLATILITY]),
                                    "ctx_curr_valence":    str(context_vector[CTX_CURR_VALENCE]),
                                },
                                maxlen=STREAM_MAXLEN, approximate=True,
                            )
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
