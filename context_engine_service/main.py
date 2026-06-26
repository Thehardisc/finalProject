import asyncio
import json
import sys
import time
import uuid
import os
import numpy as np
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdm import CDM, N_CDM_STATES, reload_hmm as _cdm_reload_hmm
import train_cdm_hmm
from appraisal import compute_appraisal
from shared.module_registry import register_module
from shared.constants import (
    CDM_CTX_DIM, N_CDM_STATES as _N,
    CTX_CDM_PROBS,
    CTX_HIST_POS, CTX_HIST_NEU, CTX_HIST_NEG, CTX_RESONANCE, CTX_VOLATILITY, CTX_CURR_VALENCE,
    CTX_RESIDENCY, CTX_TRANSITION, CTX_ABRUPTNESS, CTX_COHERENCE,
    CTX_ENTROPY, CTX_SPK_DIVERGENCE, CTX_VELOCITY, CTX_ACCELERATION,
    CTX_MSG_LENGTH, CTX_LATENCY_MS,
    CTX_HMM_CONF, CTX_HMM_ENTROPY, CTX_HMM_EMISSION, CTX_HMM_NEXT3, CTX_INTENT_STAB,
)

from fastapi import FastAPI
import redis.asyncio as aioredis
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from shared.utils.logger import get_logger

logger = get_logger("context_engine")

app = FastAPI()

REDIS_HOST  = os.getenv("REDIS_HOST",  "redis")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
STREAM_MAXLEN = int(os.getenv("REDIS_STREAM_MAXLEN", 10_000))

_GOEMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
_N_GOEMO = len(_GOEMOTION_LABELS)  # 28


def _emotion_entropy(emotions: Dict) -> float:
    probs = np.array([float(emotions.get(e, 0.0)) for e in _GOEMOTION_LABELS])
    s = probs.sum()
    probs = probs / s if s > 0 else np.ones(_N_GOEMO) / _N_GOEMO
    probs_nz = probs[probs > 0]
    return float(-np.sum(probs_nz * np.log(probs_nz)))


class ContextEngineService:
    LUA_VOLATILITY = """
    local valence_key = KEYS[1]
    local linguistic_key = KEYS[2]
    local state_key = KEYS[3]
    
    local cutoff = tonumber(ARGV[1])
    local now = tonumber(ARGV[2])
    local current_valence = tonumber(ARGV[3])
    local decay_factor = tonumber(ARGV[4])
    local uuid_str = ARGV[5]
    local ling_json = ARGV[6]
    
    redis.call("ZREMRANGEBYSCORE", valence_key, "-inf", cutoff)
    redis.call("ZREMRANGEBYSCORE", linguistic_key, "-inf", cutoff)
    
    local v_json = '{"valence": ' .. current_valence .. ', "id": "' .. uuid_str .. '"}'
    redis.call("ZADD", valence_key, now, v_json)
    redis.call("ZADD", linguistic_key, now, ling_json)
    
    local history_raw = redis.call("ZRANGE", valence_key, 0, -1)
    local prev_volatility = tonumber(redis.call("HGET", state_key, "current_volatility")) or 0.0
    
    local sum = 0
    local count = 0
    for i, v in ipairs(history_raw) do
        local decoded = cjson.decode(v)
        if decoded and decoded["valence"] then
            sum = sum + decoded["valence"]
            count = count + 1
        end
    end
    
    local mean = 0
    local current_variance = 0
    if count > 0 then
        mean = sum / count
        if count > 1 then
            local sum_sq = 0
            for i, v in ipairs(history_raw) do
                local decoded = cjson.decode(v)
                if decoded and decoded["valence"] then
                    local val = decoded["valence"]
                    sum_sq = sum_sq + (val - mean) * (val - mean)
                end
            end
            current_variance = sum_sq / count
        end
    else
        mean = current_valence
    end
    
    local new_volatility = (decay_factor * prev_volatility) + ((1 - decay_factor) * current_variance)
    
    redis.call("HSET", state_key, "current_volatility", tostring(new_volatility))
    redis.call("HSET", state_key, "last_message_ts", tostring(now))
    redis.call("HSET", state_key, "baseline_valence", tostring(mean))
    
    return tostring(new_volatility)
    """

    def __init__(self, redis_url: str, qdrant_url: str):
        self.redis  = aioredis.from_url(redis_url, decode_responses=True)
        self.qdrant = AsyncQdrantClient(url=qdrant_url)
        self.collection_name = "episodic_memory"
        
        self.embedder              = None  # lazy-loaded at startup
        self.vector_size           = 384
        self.decay_factor          = 0.95
        self.window_size_seconds   = 3600
        
        self.qdrant_write_queue = asyncio.Queue(maxsize=1000)
        self.redis_write_queue = asyncio.Queue(maxsize=1000)

    async def initialize(self):
        try:
            from sentence_transformers import SentenceTransformer
            # Load embedder synchronously here but within the startup hook to not block module parse
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
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
            asyncio.create_task(self._qdrant_worker())
            asyncio.create_task(self._redis_worker())
        except Exception as e:
            logger.error(f"Failed to initialize ContextEngine: {e}", exc_info=True)

    async def _qdrant_worker(self):
        while True:
            try:
                task = await self.qdrant_write_queue.get()
                await self.save_to_episodic_memory(**task)
                self.qdrant_write_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Qdrant worker error: {e}")

    async def _redis_worker(self):
        while True:
            try:
                task = await self.redis_write_queue.get()
                await self._store_embedding(task["key"], task["embedding"])
                self.redis_write_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Redis worker error: {e}")

    async def calculate_user_baseline(self, user_id: str, current_embedding: List[float] = None) -> Dict[str, float]:
        zero_res = {"historical_pos": 0.0, "historical_neu": 0.0, "historical_neg": 0.0, "topic_resonance": 0.0}
        if not current_embedding or not user_id:
            return zero_res

        try:
            search_result = await asyncio.wait_for(
                self.qdrant.search(
                    collection_name=self.collection_name,
                    query_vector=current_embedding,
                    query_filter=models.Filter(
                        must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
                    ),
                    limit=5,
                ),
                timeout=1.5,  # Increased from 0.5s to 1.5s
            )

            if not search_result:
                return zero_res

            total_score = sum(hit.score for hit in search_result)
            if total_score == 0:
                return zero_res

            pos, neu, neg = 0.0, 0.0, 0.0
            _emo_pos = {'admiration', 'amusement', 'approval', 'caring', 'desire', 'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'}
            _emo_neg = {'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'}

            for hit in search_result:
                w = hit.score / total_score
                emo = hit.payload.get("dominant_emotion", "neutral")
                if emo in _emo_pos:
                    pos += w
                elif emo in _emo_neg:
                    neg += w
                else:
                    neu += w

            return {
                "historical_pos": float(pos),
                "historical_neu": float(neu),
                "historical_neg": float(neg),
                "topic_resonance": float(total_score / len(search_result)),
            }
        except asyncio.TimeoutError:
            logger.warning(f"Qdrant query timed out for user {user_id} — using zero baseline")
            return zero_res
        except Exception as e:
            logger.warning(f"Failed to fetch baseline from Qdrant: {e}")
            return zero_res

    async def update_working_memory(self, user_id: str, current_valence: float, linguistic_markers: Dict) -> float:
        if not user_id:
            return 0.0

        now            = time.time()
        window_cutoff  = now - self.window_size_seconds
        valence_key    = f"context:user:{user_id}:valence_window"
        linguistic_key = f"context:user:{user_id}:linguistic_window"
        state_key      = f"context:user:{user_id}:state"

        try:
            _valence = float(current_valence)  # numpy 2.0 scalars are not JSON-serialisable
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.zremrangebyscore(valence_key,    "-inf", window_cutoff)
                pipe.zremrangebyscore(linguistic_key, "-inf", window_cutoff)
                pipe.zadd(valence_key,    {json.dumps({"valence": _valence, "id": str(uuid.uuid4())}): now})
                pipe.zadd(linguistic_key, {json.dumps(linguistic_markers): now})
                pipe.zrange(valence_key, 0, -1)
                pipe.hgetall(state_key)
                results = await pipe.execute()

            history_raw = results[4]
            prev_state  = results[5]

            recent_valences  = [json.loads(x)["valence"] for x in history_raw]
            current_variance = float(np.var(recent_valences)) if len(recent_valences) > 1 else 0.0
            prev_volatility  = float(prev_state.get("current_volatility", 0.0))
            new_volatility   = float(self.decay_factor * prev_volatility + (1 - self.decay_factor) * current_variance)
            baseline_valence = float(np.mean(recent_valences)) if recent_valences else _valence

            _USER_TTL = 86400  # 24-hour expiry on user working-memory keys
            async with self.redis.pipeline(transaction=True) as pipe2:
                pipe2.hset(state_key, mapping={
                    "current_volatility": new_volatility,
                    "last_message_ts":    float(now),
                    "baseline_valence":   baseline_valence,
                })
                pipe2.expire(state_key,      _USER_TTL)
                pipe2.expire(valence_key,    _USER_TTL)
                pipe2.expire(linguistic_key, _USER_TTL)
                await pipe2.execute()
            return new_volatility
        except Exception as e:
            logger.error(f"Redis working-memory update failed: {e}", exc_info=True)
            return 0.0

    async def _store_embedding(self, key: str, embedding: List[float]):
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
        message_text:       str = "",
        top_emotion:        str = "neutral",
    ) -> List[float]:
        if is_continuation:
            return await self._build_continuation_vector(
                conversation_id, user_id, linguistic_markers, current_embedding, last_emotions
            )

        baseline_task = self.calculate_user_baseline(user_id, current_embedding)
        memory_task   = self.update_working_memory(user_id, current_valence, linguistic_markers)
        baseline_data, volatility = await asyncio.gather(baseline_task, memory_task)

        _spk            = f"conv:{conversation_id}:spk:{user_id}"
        last_embed_key  = f"{_spk}:last_embed"
        state_hist_key  = f"{_spk}:state_hist"
        cdm_state_key   = f"{_spk}:cdm_state"
        hmm_alpha_key   = f"{_spk}:hmm_alpha"
        intent_stab_key = f"{_spk}:intent_stab"

        # Batch all 7 independent Redis reads into one round-trip
        (
            state, spk_seq_raw, last_embed_json,
            raw_prev, hmm_raw, stab_raw, state_hist_raw
        ) = await asyncio.gather(
            self.redis.hgetall(f"conversation:{conversation_id}"),
            self.redis.get(f"{_spk}:valence_seq"),
            self.redis.get(last_embed_key),
            self.redis.get(cdm_state_key),
            self.redis.get(hmm_alpha_key),
            self.redis.get(intent_stab_key),
            self.redis.lrange(state_hist_key, 0, 2),
        )

        # Derived values — all computed from the gathered results above
        spk_seq      = json.loads(spk_seq_raw) if spk_seq_raw else []
        velocity     = float(spk_seq[-1] - spk_seq[-2]) if len(spk_seq) >= 2 else 0.0
        acceleration = float((spk_seq[-1] - spk_seq[-2]) - (spk_seq[-2] - spk_seq[-3])) if len(spk_seq) >= 3 else 0.0

        spk_vals = []
        for k, v in state.items():
            if k.startswith("spk:"):
                try:
                    spk_vals.append(float(v))
                except ValueError:
                    pass
        speaker_divergence = float(np.std(spk_vals)) if len(spk_vals) >= 2 else 0.0

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
        try:
            self.redis_write_queue.put_nowait({"key": last_embed_key, "embedding": current_embedding})
        except asyncio.QueueFull:
            pass

        emotion_entropy = _emotion_entropy(last_emotions)
        prev_state_idx  = int(raw_prev) if raw_prev is not None else 0
        hmm_alpha       = np.array(json.loads(hmm_raw), dtype=np.float64) if hmm_raw else None
        prev_stab_cnt   = int(stab_raw) if stab_raw else 0
        state_hist      = [int(x) for x in state_hist_raw]

        diag = CDM.transition(
            prev_state_idx, velocity, emotion_entropy, speaker_divergence, topic_coherence,
            hmm_alpha=hmm_alpha,
            top_emotion=top_emotion,
            message_text=message_text,
        )
        current_state_idx = diag.current_state
        cdm_vec           = diag.one_hot
        entry_abruptness  = 1.0 - diag.transition_probability

        residency = 1
        for s in state_hist:
            if s == current_state_idx:
                residency += 1
            else:
                break
        state_residency = min(residency / 10.0, 1.0)

        transition_path = [s / float(N_CDM_STATES) for s in state_hist[:3]]
        while len(transition_path) < 3:
            transition_path.append(current_state_idx / float(N_CDM_STATES))

        new_stab_cnt = prev_stab_cnt + 1 if current_state_idx == prev_state_idx else 1
        intent_stability = min(new_stab_cnt / 10.0, 1.0)

        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.set(cdm_state_key,  str(current_state_idx), ex=86400 * 7)
            pipe.lpush(state_hist_key, str(current_state_idx))
            pipe.ltrim(state_hist_key, 0, 9)
            pipe.expire(state_hist_key, 86400 * 7)
            pipe.set(hmm_alpha_key,   json.dumps([float(a) for a in diag.hmm_alpha_next]), ex=86400 * 7)
            pipe.set(intent_stab_key, str(new_stab_cnt), ex=86400 * 7)
            await pipe.execute()

        ctx = np.zeros(CDM_CTX_DIM, dtype=np.float64)
        ctx[CTX_CDM_PROBS]      = cdm_vec
        ctx[CTX_RESIDENCY]      = state_residency
        ctx[CTX_TRANSITION]     = transition_path[:3]
        ctx[CTX_ABRUPTNESS]     = entry_abruptness
        ctx[CTX_COHERENCE]      = topic_coherence
        ctx[CTX_ENTROPY]        = emotion_entropy
        ctx[CTX_SPK_DIVERGENCE] = speaker_divergence
        ctx[CTX_VELOCITY]       = velocity
        ctx[CTX_ACCELERATION]   = acceleration
        ctx[CTX_HIST_POS]       = baseline_data["historical_pos"]
        ctx[CTX_HIST_NEU]       = baseline_data["historical_neu"]
        ctx[CTX_HIST_NEG]       = baseline_data["historical_neg"]
        ctx[CTX_RESONANCE]      = baseline_data["topic_resonance"]
        ctx[CTX_VOLATILITY]     = volatility
        ctx[CTX_CURR_VALENCE]   = current_valence
        ctx[CTX_MSG_LENGTH]     = float(linguistic_markers.get("length", 0))
        ctx[CTX_LATENCY_MS]     = float(linguistic_markers.get("latency_ms", 0))
        ctx[CTX_HMM_CONF]       = diag.hmm_confidence
        ctx[CTX_HMM_ENTROPY]    = diag.hmm_state_entropy
        ctx[CTX_HMM_EMISSION]   = diag.hmm_emission_logprob
        ctx[CTX_HMM_NEXT3]      = diag.hmm_top3_next[:3]
        ctx[CTX_INTENT_STAB]    = intent_stability
        return ctx.tolist()

    async def _build_continuation_vector(
        self,
        conversation_id:    str,
        user_id:            str,
        linguistic_markers: Dict,
        current_embedding:  List[float],
        last_emotions:      Dict,
    ) -> List[float]:
        _spk           = f"conv:{conversation_id}:spk:{user_id}"
        cdm_state_key  = f"{_spk}:cdm_state"
        state_hist_key = f"{_spk}:state_hist"

        raw_state     = await self.redis.get(cdm_state_key)
        current_state = int(raw_state) if raw_state is not None else 0
        cdm_vec       = CDM.one_hot(current_state)

        state_hist_raw = await self.redis.lrange(state_hist_key, 0, 2)
        state_hist = [int(x) for x in state_hist_raw]
        residency  = 1
        for s in state_hist:
            if s == current_state:
                residency += 1
            else:
                break
        state_residency = min(residency / 10.0, 1.0)

        transition_path = [s / float(N_CDM_STATES) for s in state_hist[:3]]
        while len(transition_path) < 3:
            transition_path.append(current_state / float(N_CDM_STATES))

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

        last_embed_key  = f"{_spk}:last_embed"
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

        try:
            self.redis_write_queue.put_nowait({"key": last_embed_key, "embedding": current_embedding})
        except asyncio.QueueFull:
            pass

        emotion_entropy = _emotion_entropy(last_emotions)

        hmm_alpha_key   = f"{_spk}:hmm_alpha"
        intent_stab_key = f"{_spk}:intent_stab"
        hmm_raw, stab_raw = await asyncio.gather(
            self.redis.get(hmm_alpha_key),
            self.redis.get(intent_stab_key),
        )
        hmm_alpha        = np.array(json.loads(hmm_raw), dtype=np.float64) if hmm_raw else np.ones(N_CDM_STATES) / N_CDM_STATES
        intent_stability = min(int(stab_raw or 0) / 10.0, 1.0)
        hmm_conf         = float(hmm_alpha.max())
        hmm_ent          = float(-np.sum(hmm_alpha * np.log(hmm_alpha + 1e-12)))
        top3_next        = CDM.get_next_probs(current_state)

        ctx = np.zeros(CDM_CTX_DIM, dtype=np.float64)
        ctx[CTX_CDM_PROBS]      = cdm_vec
        ctx[CTX_RESIDENCY]      = state_residency
        ctx[CTX_TRANSITION]     = transition_path[:3]
        ctx[CTX_COHERENCE]      = topic_coherence
        ctx[CTX_ENTROPY]        = emotion_entropy
        ctx[CTX_SPK_DIVERGENCE] = speaker_divergence
        ctx[CTX_VOLATILITY]     = volatility
        ctx[CTX_CURR_VALENCE]   = current_ema_valence
        ctx[CTX_MSG_LENGTH]     = float(linguistic_markers.get("length", 0))
        ctx[CTX_LATENCY_MS]     = float(linguistic_markers.get("latency_ms", 0))
        ctx[CTX_HMM_CONF]       = hmm_conf
        ctx[CTX_HMM_ENTROPY]    = hmm_ent
        ctx[CTX_HMM_NEXT3]      = top3_next[:3]
        ctx[CTX_INTENT_STAB]    = intent_stability
        return ctx.tolist()

    async def save_to_episodic_memory(
        self, user_id: str, text: str, valence: float,
        embedding: List[float], dominant_emotion: str
    ):
        if not user_id or not embedding:
            return

        try:
            await asyncio.wait_for(
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=[
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                "user_id":          user_id,
                                "timestamp":        time.time(),
                                "valence":          float(valence),
                                "dominant_emotion": dominant_emotion,
                                "text":             text[:500],  # cap to avoid large payloads
                            }
                        )
                    ]
                ),
                timeout=3.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Qdrant upsert timed out — episodic memory point dropped")
        except Exception as e:
            logger.error(f"Failed to save to Qdrant: {type(e).__name__}: {e}")


context_engine = ContextEngineService(
    redis_url=f"redis://{REDIS_HOST}:6379",
    qdrant_url=f"http://{QDRANT_HOST}:6333",
)


async def _background_hmm_training():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, train_cdm_hmm.ensure_model)
    _cdm_reload_hmm()
    logger.info("CDM HMM reloaded after background training.")


@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Context Engine (CDM v4.0 — 15 intent states, 40-dim)...")
    _cdm_reload_hmm()
    await context_engine.initialize()
    
    await register_module(
        context_engine.redis,
        "context_engine",
        required=False,
        schema="context_vector",
    )
    
    asyncio.create_task(redis_listener())
    asyncio.create_task(_dlq_drainer())

    if os.getenv("TRAIN_HMM_ON_STARTUP", "false").lower() == "true":
        asyncio.create_task(_background_hmm_training())
        logger.info("HMM training started in background (TRAIN_HMM_ON_STARTUP=True).")
    else:
        logger.info("Skipping HMM training (TRAIN_HMM_ON_STARTUP is disabled).")


async def _dlq_drainer():
    dlq_name   = "preprocessed_stream_dlq"
    group_name = "context_engine_dlq_group"
    # Fixed consumer name: avoids leaking dead consumers into the group on restarts.
    # xautoclaim reclaims any PEL from a prior run of this same consumer name.
    client_id  = "dlq_drainer_1"

    try:
        await context_engine.redis.xgroup_create(dlq_name, group_name, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.warning(f"DLQ group create: {e}")

    logger.info("DLQ drainer started — monitoring preprocessed_stream_dlq.")

    while True:
        try:
            try:
                _, stale, _ = await context_engine.redis.xautoclaim(
                    dlq_name, group_name, client_id,
                    min_idle_time=30_000, start_id="0-0", count=10,
                )
                messages = [(dlq_name, stale)] if stale else None
            except Exception:
                messages = None

            if not messages:
                messages = await context_engine.redis.xreadgroup(
                    group_name, client_id, {dlq_name: ">"}, count=10, block=5000
                )

            if messages:
                for _, msgs in messages:
                    for msg_id, data in msgs:
                        try:
                            raw   = json.loads(data.get("raw_data", "{}"))
                            mid   = raw.get("message_id", "?")
                            cid   = raw.get("conversation_id", "?")
                            error = data.get("error", "unknown")
                        except Exception:
                            mid, cid, error = "?", "?", data.get("error", "unknown")
                        logger.error(
                            "dlq_message_drained",
                            extra={
                                "event":           "dlq_message_drained",
                                "message_id":      mid,
                                "conversation_id": cid,
                                "error":           error,
                            },
                        )
                        await context_engine.redis.xack(dlq_name, group_name, msg_id)
        except Exception as e:
            if "Timeout" in str(e) or "timeout" in str(e):
                continue
            logger.warning(f"DLQ drainer loop error: {e}")
            await asyncio.sleep(5)


async def redis_listener():
    group_name  = "context_engine_group"
    client_id   = f"context_engine_{uuid.uuid4().hex[:6]}"
    stream_name = "preprocessed_stream"
    dlq_name    = "preprocessed_stream_dlq"

    try:
        await context_engine.redis.xgroup_create(stream_name, group_name, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error("xgroup_create failed", extra={"stream": stream_name, "error": str(e)})

    logger.info("context_engine_listening", extra={"event": "stream_listen_start", "stream": stream_name, "group": group_name})

    last_stale_id = '0-0'

    while True:
        try:
            try:
                next_stale_id, stale, _ = await context_engine.redis.xautoclaim(
                    stream_name, group_name, client_id,
                    min_idle_time=30_000, start_id=last_stale_id, count=10
                )
                last_stale_id = next_stale_id if next_stale_id != '0-0' else '0-0'
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
                        text            = msg_data.get("text", "")
                        user_id         = msg_data.get("user_id", "anonymous")
                        conversation_id = msg_data.get("conversation_id", "")
                        raw_msg_id      = msg_data.get("message_id", "")
                        is_continuation = str(msg_data.get("is_continuation", "false")).lower() == "true"
                        mlog = logger.bind(message_id=raw_msg_id, conversation_id=conversation_id, user_id=user_id)
                        try:
                            state        = await context_engine.redis.hgetall(f"conversation:{conversation_id}")
                            prev_valence = float(state.get("average_valence", 0.0)) if state else 0.0
                            prev_emotion = state.get("dominant_emotion", "neutral") if state else "neutral"

                            try:
                                raw_emotions = json.loads(state.get("last_message_emotions", "{}") or "{}")
                                last_emotions = {}
                                for k, v in raw_emotions.items():
                                    if k in _GOEMOTION_LABELS:
                                        try:
                                            last_emotions[k] = float(v)
                                        except (ValueError, TypeError):
                                            pass
                            except Exception:
                                last_emotions = {}

                            raw_emb = await asyncio.get_running_loop().run_in_executor(
                                None, context_engine.embedder.encode, text
                            )
                            embedding = raw_emb.tolist()

                            _now = time.time()
                            _last_ts = float(state.get("last_updated", 0.0) or 0.0) if state else 0.0
                            inter_msg_latency_ms = (_now - _last_ts) * 1000.0 if _last_ts > 0 else 0.0

                            linguistic_markers = {
                                "length":     len(text),
                                "latency_ms": inter_msg_latency_ms,
                            }

                            top_emotion = max(last_emotions, key=last_emotions.get) if last_emotions else "neutral"

                            # Read T-1's embedding BEFORE build_context_vector queues
                            # the write of T's embedding.  This guarantees we get T-1's
                            # vector to pair with T-1's emotional labels in Qdrant.
                            # Per-speaker: only this user's last embed, not the other speaker's.
                            _prev_embed_raw = await context_engine.redis.get(
                                f"conv:{conversation_id}:spk:{user_id}:last_embed"
                            )

                            try:
                                context_vector = await asyncio.wait_for(
                                    context_engine.build_context_vector(
                                        conversation_id,
                                        user_id,
                                        prev_valence,
                                        linguistic_markers,
                                        embedding,
                                        last_emotions,
                                        is_continuation=is_continuation,
                                        message_text=text,
                                        top_emotion=top_emotion,
                                    ),
                                    timeout=1.5,
                                )
                            except asyncio.TimeoutError:
                                logger.warning(
                                    f"build_context_vector timed out for {raw_msg_id} "
                                    f"— publishing zero {CDM_CTX_DIM}-dim vector"
                                )
                                context_vector = [0.0] * CDM_CTX_DIM

                            # Episodic memory: save T-1's embedding with T-1's emotional
                            # outcome.  Saving T's embedding with T-1's labels (the
                            # original bug) caused Qdrant to store mismatched text↔emotion
                            # pairs, corrupting hist_pos/hist_neg for future queries.
                            # Skip on the first message (no previous embedding yet).
                            if _prev_embed_raw:
                                try:
                                    context_engine.qdrant_write_queue.put_nowait({
                                        "user_id":          user_id,
                                        "text":             "",   # T-1's text not recoverable here
                                        "valence":          prev_valence,
                                        "embedding":        json.loads(_prev_embed_raw),
                                        "dominant_emotion": prev_emotion,
                                    })
                                except asyncio.QueueFull:
                                    logger.warning(f"Qdrant write queue full for msg {raw_msg_id}, dropping episodic memory.")

                            _nz = sum(1 for v in context_vector if v != 0.0)
                            logger.debug(
                                f"[DIAG-1] CE publish msg={raw_msg_id} "
                                f"dim={len(context_vector)} nonzero={_nz} "
                                f"cdm={[round(float(v),2) for v in context_vector[0:N_CDM_STATES]]} "
                                f"hist_pos={float(context_vector[CTX_HIST_POS]):.4f} "
                                f"cur_val={float(context_vector[CTX_CURR_VALENCE]):.4f} "
                                f"hmm_conf={float(context_vector[CTX_HMM_CONF]):.3f}"
                            )
                            
                            _appraisal = compute_appraisal({
                                "entry_abruptness":  float(context_vector[CTX_ABRUPTNESS]),
                                "current_valence":   float(context_vector[CTX_CURR_VALENCE]),
                                "topic_resonance":   float(context_vector[CTX_RESONANCE]),
                                "volatility":        float(context_vector[CTX_VOLATILITY]),
                                "speaker_divergence": float(context_vector[CTX_SPK_DIVERGENCE]),
                            })

                            await context_engine.redis.xadd(
                                "partial_analysis_stream",
                                {
                                    "message_id":               raw_msg_id,
                                    "model_name":               "context_engine",
                                    "context_vector":           json.dumps(context_vector),
                                    "original_data":            json.dumps(msg_data),
                                    "ctx_dim":                  str(len(context_vector)),
                                    "ctx_hist_pos":             str(context_vector[CTX_HIST_POS]),
                                    "ctx_hist_neu":             str(context_vector[CTX_HIST_NEU]),
                                    "ctx_hist_neg":             str(context_vector[CTX_HIST_NEG]),
                                    "ctx_topic_resonance":      str(context_vector[CTX_RESONANCE]),
                                    "ctx_volatility":           str(context_vector[CTX_VOLATILITY]),
                                    "ctx_curr_valence":         str(context_vector[CTX_CURR_VALENCE]),
                                    "appraisal_novelty":        str(_appraisal["novelty"]),
                                    "appraisal_goal_congruence": str(_appraisal["goal_congruence"]),
                                    "appraisal_coping":         str(_appraisal["coping"]),
                                },
                                maxlen=STREAM_MAXLEN, approximate=True,
                            )
                            await context_engine.redis.xack(stream_name, group_name, msg_id)

                        except Exception as msg_err:
                            mlog.error(
                                "context_engine_message_failed",
                                extra={"event": "ce_message_failed", "error_class": type(msg_err).__name__, "error": str(msg_err)},
                            )
                            try:
                                await context_engine.redis.xadd(dlq_name, {
                                    "raw_data": json.dumps(msg_data),
                                    "error": str(msg_err)
                                }, maxlen=STREAM_MAXLEN, approximate=True)
                                await context_engine.redis.xack(stream_name, group_name, msg_id)
                            except Exception as ack_err:
                                mlog.warning(
                                    "xack_failed_after_error",
                                    extra={"event": "xack_failed", "error": str(ack_err)},
                                )

        except Exception as e:
            if "NOGROUP" in str(e):
                try:
                    await context_engine.redis.xgroup_create(stream_name, group_name, mkstream=True)
                    logger.warning("Re-created consumer group after NOGROUP error.",
                                   extra={"stream": stream_name})
                except Exception as cg_err:
                    if "BUSYGROUP" not in str(cg_err):
                        logger.error("xgroup_create_failed", extra={"error": str(cg_err)})
            else:
                logger.error("redis_listener_loop_error", extra={"event": "ce_listener_error", "error": str(e)})
            await asyncio.sleep(1)


@app.get("/health")
def health():
    return {"status": "ok", "service": "context_engine"}
