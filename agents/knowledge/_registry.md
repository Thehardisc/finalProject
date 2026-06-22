# InnerLink Agent Registry
_Updated: 2026-06-21_

## Agent Status

| Agent | Domain | Health | Last active | Notes |
|-------|--------|--------|-------------|-------|
| pipeline | ingestion_service + preprocessing_service | OK | 2026-06-21 | start.sh fix; burst ACK fix |
| nlp | vader + bert + goemotions | OK | 2026-06-22 | Circular labeling FIXED — GoE gate 0.245, goe_confidence added |
| meta_learner | GatingEnsembleNet + background trainer | OK | 2026-06-22 | acc=0.6644 F1=0.629; GoE gate=0.245, ctx gate=0.243 (healthy) |
| context_engine | CDM HMM + Qdrant + appraisal | OK | 2026-06-21 | Appraisal null leak FIXED (3f04707) |
| trajectory | EmotionalDialogueEncoder (EDE) LSTM | OK | 2026-06-21 | top-3 acc 81%; VAD mismatch FIXED |
| api_frontend | api_service REST/WS + frontend React | OK | 2026-06-21 | Auth gaps fixed; all panels render |
| infra | Redis + Docker + shared/ | OK | 2026-06-21 | ingestion Dockerfile path fixed |
| aggregation | aggregation_service conversation state | OK | 2026-06-21 | EMA fix; rules TTL fix |
| persistence | persistence_service + PostgreSQL | OK | 2026-06-21 | NOGROUP recovery; DLQ present |
| llm | llm_reasoning_service | UNKNOWN | 2026-06-18 | No full audit since session 10 |

Legend: **OK** / **DEGRADED** / **UNKNOWN**

---

## System-Wide Invariants

- `FEATURE_DIM = 116`. Layout: VADER[0:4] BERT[4:11] GoE[11:39] VAD[39:42] CDM[42:82] Traj[82:110] Sarcasm[110] Dynamics[111:113] Appraisal[113:116].
- `model_name` keys must be exactly `"vader"`, `"basic_bert"`, `"go_emotions"`, `"context_engine"` — wrong key → silent neutral fallback.
- `shared/constants.py` is single source of truth. Use `CTX_*` named constants — never hardcode integers.
- `trainer/cycle.py` feature building MUST match `meta_learner.py:build_feature_vector` exactly. Mismatch → `X has N features but StandardScaler expecting M`.
- GoEmotions gate hard-capped at ≤0.50 via `.clamp(max=0.50)` in `GatingEnsembleNet.forward()`.
- `CDM_CTX_DIM=40` (context_engine vector). `CONTEXT_DIM=74` (full context block in GatingEnsembleNet).
- `gate_weights_alpha` in WebSocket payload: [vader, bert, goe, vad, ctx] — 5 elements. Frontend shows only [0:3].

---

## Open Cross-Cutting Issues

| ID | Agents | Description | Priority |
|----|--------|-------------|----------|
| XSS-001 | nlp + meta_learner + persistence | GoEmotions circular labeling — GoE direct samples reduced 37%→21%, model deployed acc=0.6644. DB label contamination (persistence) still open. | MEDIUM |
| XSS-002 | meta_learner + persistence | Contaminated "neutral" rows from bug period in DB → trainer learns neutral bias. Need SQL cleanup. | HIGH |
| XSS-003 | llm | Full audit of llm_reasoning_service pending (no review since session 10). | MEDIUM |

---

## Recent Cross-Agent Tasks

| Date | Task | Agents | Outcome |
|------|------|--------|---------|
| 2026-06-22 | GoEmotions circular labeling fix + trainer cycle | nlp + meta_learner | COMPLETE. GoE 1000→400/class, cache v4, 9768 samples (21.1%). Trainer: acc=0.6644, F1=0.629, DEPLOYED. Gate: goe=0.245, ctx=0.243. DB label contamination still open (XSS-001 MEDIUM). |
| 2026-06-21 | GoEmotions circular labeling status + next steps | nlp + meta_learner | Fix path: `_GOE_DIRECT_PER_CLASS=400`, delete `goemotions_direct_cache.pkl`. New issue ISS-N005 (goe_confidence not exposed). Cross-request to persistence re: is_verified count. |

---

## Agent Invocation Reference

```bash
# Python CLI (from project root)
python agents/run.py "describe the task"
python agents/run.py --agent pipeline "debug Redis stream lag"
python agents/run.py --agents nlp,meta_learner "check feature vector parity"
python agents/run.py --status

# Claude Code slash commands
/project:agent "describe the task"         # head agent auto-routes
/project:agent-pipeline "debug task"       # direct sub-agent
/project:agent-nlp "diagnose GoEmotions"
/project:agent-meta_learner "retrain"
/project:agent-context_engine "CDM audit"
/project:agent-trajectory "LSTM debug"
/project:agent-api_frontend "auth issue"
/project:agent-infra "Redis health"
/project:agent-aggregation "EMA valence"
/project:agent-persistence "DB cleanup"
/project:agent-llm "audit service"
```
