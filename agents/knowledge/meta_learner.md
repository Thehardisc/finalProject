# meta_learner Knowledge Base
Last updated: 2026-06-21

## Architecture
**GatingEnsembleNet** (`central_responder_service/trainer/models.py`)
- Inputs: 5 streams [vader, bert, goe, vad, ctx]
- Gate weights α: [vader, bert, goe, vad, ctx] (5 floats, sum=1)
- GoE gate hard-capped at ≤0.50 via `.clamp(max=0.50)` in forward()
- ctx block = x[:, ML_DIM:FEATURE_DIM] = 74-dim (CDM40 + Traj28 + Sarcasm1 + Dynamics2 + Appraisal3)

**116-dim Feature Vector** (`meta_learner.py:build_feature_vector`):
| Block        | Indices  | Dims | Source                                    |
|-------------|----------|------|------------------------------------------|
| VADER        | [0:4]    | 4    | vader_neg/neu/pos/compound                |
| BERT Ekman   | [4:11]   | 7    | 7 Ekman labels                            |
| GoEmotions   | [11:39]  | 28   | 28 labels from EMOTION_LABELS             |
| VAD lexicon  | [39:42]  | 3    | valence, arousal, dominance (Warriner)    |
| CDM Context  | [42:82]  | 40   | context_engine 40-dim vector              |
| Traj Prior   | [82:110] | 28   | EDE predicted_next from prev message      |
| Sarcasm      | [110]    | 1    | sarcasm_classifier score [0,1]            |
| Dynamics     | [111:113]| 2    | inertia + contagion (Kuppens/Kramer)      |
| Appraisal    | [113:116]| 3    | novelty, goal_congruence, coping (Scherer)|

Constants: ML_DIM=42, CONTEXT_DIM=74, FEATURE_DIM=116

**Key invariant**: `trainer/cycle.py` feature building and `meta_learner.py:build_feature_vector` MUST produce identical 116-dim vectors. Mismatch → `X has N features but StandardScaler expecting M`.

**Training pipeline** (`trainer/cycle.py`):
- Datasets: MELD (6000), EmpatheticDialogues (25000), dair-ai/emotion (16K), csv_local (7506), GoEmotions direct (remainder)
- Caches: `.cache/*.pkl` on host, bind-mounted to `/app/models/`
- Gate: new model hot-reloaded only if test accuracy ≥ ACCURACY_GATE (default 0.40)
- Retrain interval: RETRAIN_INTERVAL_SECONDS (default 1800s)
- Model output: `models/meta_weights.pkl` (+ `.sha256`, `_meta.json`, `.ready`)

**Model files** (bind-mounted, no rebuild needed):
- `central_responder_service/models/meta_weights.pkl`
- `central_responder_service/models/meta_weights_meta.json`

## Known Issues
- **relief class F1=0.118** [ISS-C008, OPEN]: Only 23 test samples → near-random precision. Inherent data scarcity; no fix without more relief-labeled real data.
- **grief class F1=0.310** [ISS-C009, OPEN]: 139 test samples, low precision (0.201). Rare class, hard to improve without data augmentation.

## Improvement Queue
- **[Med]** Use `goe_confidence` in `meta_learner.py:predict_with_meta_learner` to dynamically suppress GoE gate when max GoE score < 0.15.
- **[Med]** Add emoji_scores as extra signal block (would require feature vector expansion to 144 dims — coordinate with nlp agent).
- **[Med]** Increase ACCURACY_GATE to 0.50 once model stabilises above that level.
- **[Low]** Expose per-class training F1 in meta_weights_meta.json for monitoring.

## Cross-Agent Dependencies
- Depends on: **nlp** for VADER[0:4], BERT[4:11], GoE[11:39] blocks
- Depends on: **context_engine** for CDM context vector [42:82]
- Depends on: **trajectory** for prior [82:110]
- Depends on: **infra** for model file paths and pkl lifecycle

## Inter-Agent Requests (Pending)
*None*

## Recent History
- 2026-06-22: **Trainer cycle complete — model DEPLOYED.** GoE 1000→400/class (9768 samples, 21.1% of training set). Acc 0.7003→0.6644 (↓3.59% — expected: old accuracy inflated by circular labeling). Gate weights: vader:0.167 bert:0.193 goe:0.245 vad:0.152 ctx:0.243. ctx gate now 0.243 (was 0.096) — confirmed stale cache issue resolved.
- 2026-06-21: ctx gate = 0.096 training vs 0.240 test — identified root cause as stale caches
- 2026-06-20: Deleted meld_features_cache.pkl, empathetic_train_cache.pkl, csv_local_cache.pkl, empathetic_goe_cache.npy, meld_goe_cache.npy
- 2026-06-20: per-speaker CDM isolation fix deployed; trainer rebuild started
