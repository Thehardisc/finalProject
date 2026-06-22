# context_engine Knowledge Base
Last updated: 2026-06-21

## Architecture
**CDM HMM**: 15 intent states × 28 observations (4 dialog acts × 7 emotion groups).
Model file: `context_engine_service/models/cdm_hmm.pkl`

**15 CDM States** (N_CDM_STATES=15):
NEUTRAL, QUESTION, AGREEMENT, DISAGREEMENT, EMPATHY, WARMTH, TENSION, CONFLICT,
ARGUMENT, WITHDRAWAL, FRUSTRATION, PRAISE, RECONCILIATION, TOPIC_SHIFT, SUSTAINED_AFFECT

**40-dim CDM context vector** (indices within CDM block, feature offset=39):
- [0:15]  CDM intent one-hot (N_CDM_STATES)
- [15]    state_residency
- [16:19] transition_path (last 3 state indices / N_CDM_STATES)
- [19]    entry_abruptness
- [20]    topic_coherence
- [21]    emotion_entropy
- [22]    speaker_divergence (conversation-level, std across speakers)
- [23]    velocity (Δvalence per-speaker)
- [24]    acceleration (Δ²valence per-speaker)
- [25]    hist_valence_pos (Qdrant episodic memory)
- [26]    hist_valence_neu
- [27]    hist_valence_neg
- [28]    topic_resonance
- [29]    volatility
- [30]    current_valence
- [31]    message_length
- [32]    latency_ms
- [33]    hmm_conf (CTX_HMM_CONF — drives ctx_weight in GatingEnsembleNet)
- [34]    hmm_entropy
- [35]    hmm_emission_logprob
- [36:39] hmm_top3_next_probs
- [39]    intent_stability

**Per-speaker Redis keys** (FIXED 2026-06-20, was per-conversation):
```
conv:{cid}:spk:{uid}:cdm_state
conv:{cid}:spk:{uid}:hmm_alpha
conv:{cid}:spk:{uid}:state_hist
conv:{cid}:spk:{uid}:intent_stab
conv:{cid}:spk:{uid}:last_embed
```
Velocity/acceleration read from `conv:{cid}:spk:{uid}:valence_seq` (written by aggregation_service).

**speaker_divergence**: KEPT at conversation level (std of valence across all spk: entries in `conversation:{cid}` hash) — correctly measures inter-speaker difference.

**Qdrant episodic memory**:
- Collection: per conversation_id
- Stores sentence embeddings for context retrieval
- hist_valence_pos/neu/neg populated from Qdrant search results
- last_embed written after each message to `conv:{cid}:spk:{uid}:last_embed`

**Session reset cleanup** (api_service/routes/conversations.py):
```python
async for key in r.scan_iter(f"conv:{conversation_id}:spk:*"):
    await r.delete(key)
```

## Known Issues
- **Fixed 2026-06-20**: UnboundLocalError `'_spk' referenced before assignment` — `_spk` was defined after first use. Fix: moved `_spk = f"conv:{conversation_id}:spk:{user_id}"` to very top of `build_context_vector()`.
- **ctx gate low during training (0.096)**: stale pre-fix caches caused GatingEnsembleNet to distrust ctx block. Fix: cache rebuild in progress.
- OPTIONAL_TIMEOUT_MS=1000 — if context_engine is slow, aggregation proceeds with zero ctx vector.

## Improvement Queue
- **[Med]** Add CDM state transition logging at DEBUG level for traceability.
- **[Med]** Expose HMM confidence (hmm_conf) in WebSocket payload for UI display.
- **[Low]** CDM state cards UI: show current state + transition probabilities in frontend.

## Cross-Agent Dependencies
- Provides: 40-dim CDM context vector [42:82] to **meta_learner**
- Reads: `conv:{cid}:spk:{uid}:valence_seq` from **infra** (written by aggregation_service)
- Depends on: **pipeline** for message delivery (optional module, OPTIONAL_TIMEOUT_MS)
- Depends on: **infra** for Redis + Qdrant configuration

## Inter-Agent Requests (Pending)
*None*

## Recent History
- 2026-06-21: UnboundLocalError fix verified via docker logs
- 2026-06-21: Per-speaker isolation tested — keys created correctly for each user_id
- 2026-06-20: Speaker_divergence preserved at conversation level (correct behavior)
