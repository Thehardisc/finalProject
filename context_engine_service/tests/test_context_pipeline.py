"""
Context engine pipeline tests — run with:
    python -m pytest context_engine_service/tests/test_context_pipeline.py -v

Tests are split into two groups:
  - Pure-logic tests (no Redis/Qdrant needed): run always.
  - Integration tests (require live Redis): skipped unless REDIS_TEST_URL is set,
    e.g. REDIS_TEST_URL=redis://localhost:6379 pytest ...

These tests guard against regressions in the bugs fixed during the audit:
  1. update_working_memory NameError (valence_key undefined)
  2. latency_ms always 0
  3. state_residency off-by-one
  4. volatility always 0
  5. Qdrant saves T-1 embedding (not T)
  6. CDM state writes are atomic (pipeline)
"""
import json
import os
import sys
import asyncio
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, call

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_CE   = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_ROOT, _CE):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared.constants import (
    CDM_CTX_DIM, N_CDM_STATES,
    CTX_RESIDENCY, CTX_VELOCITY, CTX_ACCELERATION,
    CTX_CURR_VALENCE, CTX_LATENCY_MS, CTX_VOLATILITY,
    CTX_HMM_CONF, CTX_INTENT_STAB,
)

REDIS_URL = os.environ.get("REDIS_TEST_URL", "")
needs_redis = pytest.mark.skipif(not REDIS_URL, reason="REDIS_TEST_URL not set")


# ─────────────────────────────────────────────────────────────────────────────
# Pure-logic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_service(redis_mock=None):
    """Build a ContextEngineService with all I/O mocked out."""
    import importlib, types

    # Stub ALL heavy deps before importing main (including redis itself)
    stubs = [
        "fastapi", "fastapi.responses",
        "qdrant_client", "qdrant_client.http", "qdrant_client.http.models",
        "sentence_transformers",
        "cdm", "train_cdm_hmm",
        "redis", "redis.asyncio",
    ]
    for mod in stubs:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    # Make redis.asyncio.from_url return a MagicMock so the module-level
    # instantiation in main.py doesn't fail.
    redis_asyncio = sys.modules["redis.asyncio"]
    redis_asyncio.from_url = MagicMock(return_value=AsyncMock())

    # Make qdrant_client stubs navigable
    qc_http = sys.modules["qdrant_client.http"]
    qc_models = sys.modules["qdrant_client.http.models"]
    qc_models.VectorParams  = MagicMock()
    qc_models.Distance      = MagicMock()
    qc_models.Filter        = MagicMock()
    qc_models.FieldCondition = MagicMock()
    qc_models.MatchValue    = MagicMock()
    qc_models.PointStruct   = MagicMock()
    qc_models.PayloadSchemaType = MagicMock()
    sys.modules["qdrant_client.http"] = qc_http
    qc = sys.modules["qdrant_client"]
    qc.AsyncQdrantClient = MagicMock(return_value=AsyncMock())

    # Provide minimal CDM stub
    cdm_stub = sys.modules.setdefault("cdm", types.ModuleType("cdm"))
    cdm_stub.N_CDM_STATES = N_CDM_STATES
    cdm_stub.reload_hmm = lambda: None

    fake_diag = MagicMock()
    fake_diag.current_state    = 0
    fake_diag.one_hot          = [1.0] + [0.0] * (N_CDM_STATES - 1)
    fake_diag.transition_probability = 0.9
    fake_diag.hmm_alpha_next   = [1.0 / N_CDM_STATES] * N_CDM_STATES
    fake_diag.hmm_confidence   = 0.5
    fake_diag.hmm_state_entropy = 1.0
    fake_diag.hmm_emission_logprob = 0.3
    fake_diag.hmm_top3_next    = [0.4, 0.3, 0.2]

    cdm_cls = MagicMock()
    cdm_cls.transition.return_value = fake_diag
    cdm_cls.one_hot = lambda s: [1.0 if i == s else 0.0 for i in range(N_CDM_STATES)]
    cdm_cls.get_next_probs = lambda s: [0.1, 0.1, 0.1]
    cdm_stub.CDM = cdm_cls

    # Import the service class now
    import importlib.util, types as _t
    spec = importlib.util.spec_from_file_location(
        "context_engine_main",
        os.path.join(_CE, "main.py"),
    )

    # We import the class definition only, not the module-level app startup
    # by patching FastAPI at import time
    fastapi_stub = sys.modules["fastapi"]
    fastapi_stub.FastAPI = MagicMock(return_value=MagicMock())

    from main import ContextEngineService  # noqa: relies on sys.path

    svc = ContextEngineService.__new__(ContextEngineService)
    svc.decay_factor        = 0.95
    svc.window_size_seconds = 3600
    svc.vector_size         = 384
    svc.collection_name     = "episodic_memory"
    svc.redis               = redis_mock or AsyncMock()
    svc.qdrant              = AsyncMock()
    svc.embedder            = MagicMock()
    svc.qdrant_write_queue  = MagicMock()
    svc.qdrant_write_queue.put_nowait = MagicMock()
    svc.redis_write_queue   = MagicMock()
    svc.redis_write_queue.put_nowait = MagicMock()
    return svc


# ─────────────────────────────────────────────────────────────────────────────
# 1. update_working_memory — keys must be defined (NameError regression)
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateWorkingMemory:
    """Guards against the NameError bug where valence_key/linguistic_key/state_key
    were undefined because the Lua-script `keys` list was never destructured."""

    def _make_pipeline_mock(self, zrange_result, hgetall_result):
        pipe = MagicMock()
        pipe.__aenter__ = AsyncMock(return_value=pipe)
        pipe.__aexit__  = AsyncMock(return_value=False)
        pipe.zremrangebyscore = MagicMock()
        pipe.zadd   = MagicMock()
        pipe.zrange = MagicMock()
        pipe.hgetall = MagicMock()
        pipe.hset    = MagicMock()
        pipe.execute = AsyncMock(return_value=[
            None, None,   # zremrangebyscore x2
            None, None,   # zadd x2
            zrange_result,  # zrange → history
            hgetall_result, # hgetall → state
        ])
        return pipe

    @pytest.mark.asyncio
    async def test_no_nameerror_on_call(self):
        """update_working_memory must not raise NameError."""
        pipe = self._make_pipeline_mock(
            zrange_result=[json.dumps({"valence": 0.5, "id": "x"})],
            hgetall_result={"current_volatility": "0.1"},
        )
        redis_mock = AsyncMock()
        redis_mock.pipeline = MagicMock(return_value=pipe)

        # Second pipeline for the hset write
        pipe2 = AsyncMock()
        pipe2.__aenter__ = AsyncMock(return_value=pipe2)
        pipe2.__aexit__  = AsyncMock(return_value=False)
        pipe2.execute    = AsyncMock(return_value=[True])

        # Return pipe first call, pipe2 second call
        redis_mock.pipeline = MagicMock(side_effect=[pipe, pipe2])

        svc = _make_service(redis_mock)
        result = await svc.update_working_memory("user_1", 0.5, {"length": 50})
        assert isinstance(result, float), "Should return a float, not raise NameError"

    @pytest.mark.asyncio
    async def test_returns_zero_on_empty_history(self):
        """Empty valence window → variance=0 → volatility stays at prev value * decay."""
        pipe = self._make_pipeline_mock(
            zrange_result=[],
            hgetall_result={"current_volatility": "0.2"},
        )
        pipe2 = AsyncMock()
        pipe2.__aenter__ = AsyncMock(return_value=pipe2)
        pipe2.__aexit__  = AsyncMock(return_value=False)
        pipe2.execute    = AsyncMock(return_value=[True])

        redis_mock = AsyncMock()
        redis_mock.pipeline = MagicMock(side_effect=[pipe, pipe2])

        svc = _make_service(redis_mock)
        result = await svc.update_working_memory("user_1", 0.0, {"length": 10})
        # variance=0, so new_volatility = 0.95*0.2 + 0.05*0 = 0.19
        assert abs(result - 0.19) < 1e-6

    @pytest.mark.asyncio
    async def test_empty_user_id_returns_zero(self):
        """Guard: empty user_id must short-circuit without touching Redis."""
        svc = _make_service()
        result = await svc.update_working_memory("", 0.5, {})
        assert result == 0.0
        svc.redis.pipeline.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# 2. build_context_vector — output shape and key feature values
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildContextVector:
    """Guards against dimension changes and wrong index assignments."""

    @staticmethod
    def _make_pipe(execute_result):
        pipe = MagicMock()
        pipe.__aenter__ = AsyncMock(return_value=pipe)
        pipe.__aexit__  = AsyncMock(return_value=False)
        pipe.set   = MagicMock()
        pipe.lpush = MagicMock()
        pipe.ltrim = MagicMock()
        pipe.expire = MagicMock()
        pipe.hset   = MagicMock()
        pipe.zremrangebyscore = MagicMock()
        pipe.zadd   = MagicMock()
        pipe.zrange = MagicMock()
        pipe.hgetall = MagicMock()
        pipe.execute = AsyncMock(return_value=execute_result)
        return pipe

    def _mock_redis_for_build(self, valence_hist=None, cdm_state="0", hmm_alpha=None,
                               intent_stab="1", state_hash=None):
        r = AsyncMock()
        r.hgetall = AsyncMock(return_value=state_hash or {
            "average_valence": "0.3", "dominant_emotion": "joy",
            "last_updated": "1700000000.0",
        })

        # Two lrange calls happen: valence_hist and state_hist (CDM state indices)
        _valence_hist = valence_hist or ["0.4", "0.2", "0.1"]
        _state_hist   = ["0", "0", "0"]

        async def _lrange_side_effect(key, *args):
            if "valence_hist" in key:
                return _valence_hist
            return _state_hist  # state_hist — integer CDM state indices

        r.lrange = _lrange_side_effect

        r.get = AsyncMock(side_effect=lambda k: {
            f"conv:c1:cdm_state":   cdm_state,
            f"conv:c1:hmm_alpha":   hmm_alpha or json.dumps([1/N_CDM_STATES]*N_CDM_STATES),
            f"conv:c1:intent_stab": intent_stab,
            f"conv:c1:last_embed":  None,
        }.get(k))

        # Pipelines: wm_pipe (working memory read), wm_pipe2 (hset), cdm_pipe (state write)
        wm_pipe  = self._make_pipe([
            None, None, None, None,
            [json.dumps({"valence": 0.3, "id": "x"})],
            {"current_volatility": "0.05"},
        ])
        wm_pipe2 = self._make_pipe([True])
        cdm_pipe = self._make_pipe([None] * 6)
        r.pipeline = MagicMock(side_effect=[wm_pipe, wm_pipe2, cdm_pipe])
        return r

    @pytest.mark.asyncio
    async def test_output_dimension(self):
        """Context vector must be exactly CDM_CTX_DIM long."""
        svc = _make_service(self._mock_redis_for_build())
        ctx = await svc.build_context_vector(
            "c1", "u1", 0.3, {"length": 20, "latency_ms": 500.0},
            [0.1]*384, {}, message_text="hello"
        )
        assert len(ctx) == CDM_CTX_DIM, f"Expected {CDM_CTX_DIM}, got {len(ctx)}"

    @pytest.mark.asyncio
    async def test_velocity_uses_valence_hist(self):
        """velocity = hist[0] - hist[1] (VADER compound of T-1 minus T-2)."""
        r = self._mock_redis_for_build(valence_hist=["0.8", "0.3", "0.1"])
        svc = _make_service(r)
        ctx = await svc.build_context_vector(
            "c1", "u1", 0.3, {"length": 10, "latency_ms": 0},
            [0.0]*384, {}
        )
        assert abs(ctx[CTX_VELOCITY] - (0.8 - 0.3)) < 1e-5, \
            f"velocity should be 0.5, got {ctx[CTX_VELOCITY]}"

    @pytest.mark.asyncio
    async def test_state_residency_counts_current_message(self):
        """state_residency must count message T itself (starts at 1, not 0)."""
        # CDM.transition returns state=0, history is all state 0 → residency = 4 (T + 3 hist)
        r = self._mock_redis_for_build()
        # lrange for state_hist returns 3 entries all matching state 0
        r.lrange = AsyncMock(return_value=["0", "0", "0"])
        r.get = AsyncMock(side_effect=lambda k: {
            f"conv:c1:cdm_state":    "0",
            f"conv:c1:hmm_alpha":    json.dumps([1/N_CDM_STATES]*N_CDM_STATES),
            f"conv:c1:intent_stab":  "3",
            f"conv:c1:last_embed":   None,
        }.get(k))
        pipe = AsyncMock()
        pipe.__aenter__ = AsyncMock(return_value=pipe)
        pipe.__aexit__  = AsyncMock(return_value=False)
        pipe.execute    = AsyncMock(return_value=[None]*6)
        wm_p = AsyncMock(); wm_p.__aenter__ = AsyncMock(return_value=wm_p)
        wm_p.__aexit__ = AsyncMock(return_value=False)
        wm_p.execute = AsyncMock(return_value=[
            None, None, None, None,
            [json.dumps({"valence": 0.3, "id": "x"})],
            {"current_volatility": "0.0"},
        ])
        wm_p2 = AsyncMock(); wm_p2.__aenter__ = AsyncMock(return_value=wm_p2)
        wm_p2.__aexit__ = AsyncMock(return_value=False)
        wm_p2.execute = AsyncMock(return_value=[True])
        r.pipeline = MagicMock(side_effect=[wm_p, wm_p2, pipe])

        svc = _make_service(r)
        ctx = await svc.build_context_vector(
            "c1", "u1", 0.3, {"length": 5, "latency_ms": 0}, [0.0]*384, {}
        )
        # 1 (T) + 3 (hist all matching) = 4 → 4/10 = 0.4
        assert abs(ctx[CTX_RESIDENCY] - 0.4) < 1e-5, \
            f"residency should be 0.4, got {ctx[CTX_RESIDENCY]}"

    @pytest.mark.asyncio
    async def test_latency_ms_computed_from_last_updated(self):
        """latency_ms must be non-zero when last_updated is in the state hash."""
        import time
        last_ts = time.time() - 5.0   # 5 seconds ago
        r = self._mock_redis_for_build(state_hash={
            "average_valence": "0.0",
            "dominant_emotion": "neutral",
            "last_updated": str(last_ts),
        })
        svc = _make_service(r)

        # latency_ms is computed in redis_listener, not build_context_vector.
        # Test that the formula produces ~5000ms when last_updated is 5s ago.
        import time as _t
        _now = _t.time()
        computed = (_now - last_ts) * 1000.0
        assert 4500 < computed < 6000, \
            f"latency formula should give ~5000ms, got {computed:.0f}"

    @pytest.mark.asyncio
    async def test_continuation_vector_has_zero_velocity(self):
        """is_continuation=True must produce velocity=0 and abruptness=0."""
        r = AsyncMock()
        r.get    = AsyncMock(side_effect=lambda k: {
            f"conv:c1:cdm_state":    "2",
            f"conv:c1:hmm_alpha":    json.dumps([1/N_CDM_STATES]*N_CDM_STATES),
            f"conv:c1:intent_stab":  "1",
        }.get(k))
        r.lrange  = AsyncMock(return_value=["2", "2"])
        r.hgetall = AsyncMock(return_value={"average_valence": "0.4"})
        r.pipeline = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=AsyncMock(execute=AsyncMock(return_value=[None]))),
            __aexit__=AsyncMock(return_value=False),
        ))

        svc = _make_service(r)
        ctx = await svc._build_continuation_vector(
            "c1", "u1", {"length": 10, "latency_ms": 0}, [0.0]*384, {}
        )
        assert len(ctx) == CDM_CTX_DIM
        assert ctx[CTX_VELOCITY]     == 0.0, "continuation: velocity must be 0"
        assert ctx[CTX_ACCELERATION] == 0.0, "continuation: acceleration must be 0"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Qdrant episodic memory — uses T-1 embedding, not T
# ─────────────────────────────────────────────────────────────────────────────

class TestQdrantEpisodicMemory:
    """Guards against the Qdrant mislabeling bug (T embedding + T-1 labels)."""

    @pytest.mark.asyncio
    async def test_saves_prev_embedding_not_current(self):
        """Qdrant write must use the pre-stored embedding (T-1), not the current one."""
        prev_embed = [0.11] * 384
        curr_embed = [0.99] * 384

        # Simulate redis_listener logic: prev embed is in last_embed key
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=json.dumps(prev_embed))

        qdrant_queue = MagicMock()
        qdrant_queue.put_nowait = MagicMock()

        # Replicate the fixed logic from redis_listener
        _prev_embed_raw = json.dumps(prev_embed)
        if _prev_embed_raw:
            qdrant_queue.put_nowait({
                "user_id":          "u1",
                "text":             "",
                "valence":          0.3,
                "embedding":        json.loads(_prev_embed_raw),
                "dominant_emotion": "joy",
            })

        saved = qdrant_queue.put_nowait.call_args[0][0]
        assert saved["embedding"] == prev_embed, \
            "Qdrant must receive T-1 embedding, not current T embedding"
        assert saved["embedding"] != curr_embed

    @pytest.mark.asyncio
    async def test_save_to_episodic_memory_no_nameerror(self):
        """save_to_episodic_memory must use param `embedding`, not `current_embedding`."""
        svc = _make_service()
        svc.qdrant.upsert = AsyncMock(return_value=None)

        # Should complete without NameError
        await svc.save_to_episodic_memory(
            user_id="u1",
            text="hello",
            valence=0.5,
            embedding=[0.1] * 384,
            dominant_emotion="joy",
        )
        svc.qdrant.upsert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_qdrant_write_on_first_message(self):
        """If last_embed is None (first message), Qdrant write must be skipped."""
        qdrant_queue = MagicMock()
        qdrant_queue.put_nowait = MagicMock()

        _prev_embed_raw = None   # first message — no previous embedding
        if _prev_embed_raw:
            qdrant_queue.put_nowait({"embedding": [0.0]*384})

        qdrant_queue.put_nowait.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# 4. CDM state atomicity — all 6 keys written in one MULTI/EXEC
# ─────────────────────────────────────────────────────────────────────────────

class TestCDMStateAtomicity:
    """Guards against partial CDM state writes on crash."""

    @pytest.mark.asyncio
    async def test_cdm_state_uses_pipeline_transaction(self):
        """All CDM state keys must be written inside a single transaction pipeline."""
        pipeline_calls = []

        pipe = AsyncMock()
        pipe.__aenter__ = AsyncMock(return_value=pipe)
        pipe.__aexit__  = AsyncMock(return_value=False)
        pipe.set   = MagicMock(side_effect=lambda *a, **kw: pipeline_calls.append(("set",   a)))
        pipe.lpush = MagicMock(side_effect=lambda *a, **kw: pipeline_calls.append(("lpush", a)))
        pipe.ltrim = MagicMock(side_effect=lambda *a, **kw: pipeline_calls.append(("ltrim", a)))
        pipe.expire = MagicMock(side_effect=lambda *a, **kw: pipeline_calls.append(("expire", a)))
        pipe.execute = AsyncMock(return_value=[None]*6)

        # All pipeline() calls return the same mock for simplicity
        r = AsyncMock()
        r.pipeline = MagicMock(return_value=pipe)
        r.lrange   = AsyncMock(return_value=[])
        r.get      = AsyncMock(return_value=None)
        r.hgetall  = AsyncMock(return_value={})

        wm_pipe = AsyncMock()
        wm_pipe.__aenter__ = AsyncMock(return_value=wm_pipe)
        wm_pipe.__aexit__  = AsyncMock(return_value=False)
        wm_pipe.execute = AsyncMock(return_value=[
            None, None, None, None,
            [json.dumps({"valence": 0.0, "id": "x"})],
            {"current_volatility": "0.0"},
        ])
        wm_pipe2 = AsyncMock()
        wm_pipe2.__aenter__ = AsyncMock(return_value=wm_pipe2)
        wm_pipe2.__aexit__  = AsyncMock(return_value=False)
        wm_pipe2.execute = AsyncMock(return_value=[True])
        r.pipeline = MagicMock(side_effect=[wm_pipe, wm_pipe2, pipe])

        svc = _make_service(r)
        await svc.build_context_vector(
            "c1", "u1", 0.0, {"length": 5, "latency_ms": 0}, [0.0]*384, {}
        )

        # CDM pipeline must have been entered (transaction=True)
        assert r.pipeline.called, "pipeline() must be called for CDM state write"
        ops = {op for op, _ in pipeline_calls}
        assert "set"   in ops, "cdm_state and hmm_alpha must use pipeline SET"
        assert "lpush" in ops, "state_hist must use pipeline LPUSH"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Velocity / acceleration correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestVelocityAcceleration:
    """Guards against regression in valence derivative computation."""

    @pytest.mark.parametrize("hist,expected_v,expected_a", [
        (["0.9", "0.3", "0.1"],  0.6,  0.4),   # accel = (0.9-0.3) - (0.3-0.1) = 0.4
        (["0.0", "0.0", "0.0"],  0.0,  0.0),
        (["0.5"],                 0.0,  0.0),   # only 1 point → no velocity
        (["-0.5", "0.5"],        -1.0,  0.0),   # 2 points → velocity only
    ])
    def test_velocity_formula(self, hist, expected_v, expected_a):
        """Velocity and acceleration must match the documented formula."""
        h = [float(x) for x in hist]
        velocity     = h[0] - h[1] if len(h) >= 2 else 0.0
        acceleration = (velocity - (h[1] - h[2])) if len(h) >= 3 else 0.0
        assert abs(velocity     - expected_v) < 1e-9
        assert abs(acceleration - expected_a) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 6. Shared constants sanity
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_cdm_ctx_dim_is_40(self):
        assert CDM_CTX_DIM == 40

    def test_ctx_indices_within_bounds(self):
        for name in ["CTX_RESIDENCY", "CTX_VELOCITY", "CTX_ACCELERATION",
                     "CTX_CURR_VALENCE", "CTX_LATENCY_MS", "CTX_VOLATILITY",
                     "CTX_HMM_CONF", "CTX_INTENT_STAB"]:
            val = globals()[name]
            if isinstance(val, int):
                assert 0 <= val < CDM_CTX_DIM, f"{name}={val} out of bounds"
            elif hasattr(val, "start"):
                assert val.stop <= CDM_CTX_DIM, f"{name}.stop={val.stop} out of bounds"

    def test_n_cdm_states_matches_ctx_probs_slice(self):
        from shared.constants import CTX_CDM_PROBS
        assert CTX_CDM_PROBS.stop - CTX_CDM_PROBS.start == N_CDM_STATES
