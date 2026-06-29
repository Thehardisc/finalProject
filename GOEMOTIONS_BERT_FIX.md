# GoEmotions / BERT Fusion — Problem Analysis & Fix Plan

## Summary

The system produces wrong or weak emotion labels for sentences containing intensifiers
before emotion words (e.g. "I'm **absolutely** furious right now"). Root cause: the
GoEmotions model is trained on Reddit data where "absolutely" is overwhelmingly an
approval/agreement signal — not an intensifier. BERT, trained on a different corpus,
handles these sentences correctly at very high confidence.

---

## Evidence

| Sentence | GoEmotions (top) | BERT (top) | System output |
|---|---|---|---|
| "I'm absolutely furious right now" | **approval 22.9%** ❌ | **anger 98.7%** ✓ | annoyance |
| "I hate this so much" | anger 66% ✓ | anger 74% ✓ | anger |
| "This makes me so angry" | anger 70% ✓ | anger 99% ✓ | anger |

GoEmotions gets "absolutely furious" completely wrong at very low confidence (22.9%).
BERT is 98.7% certain it is anger. The system currently ignores this and outputs "annoyance"
because the existing BERT guard only fires when the meta-learner prediction is outside
BERT's subtype set — and "annoyance" IS a valid anger sub-type, so the guard stays silent.

---

## Root Cause Chain

```
GoEmotions (Reddit-trained)
  "absolutely" → approval context on Reddit ("absolutely!", "absolutely right!")
  "absolutely furious" → model has never seen this pattern → guesses approval at 22%

Feature vector built with noisy GoEmotions block (approval dominant, low confidence)
  └─ Meta-learner trained primarily on GoEmotions signal (28 dims vs BERT's 7)
       └─ Outputs "annoyance" (weakly correct cluster, but wrong confidence level)

BERT guard in predict_with_meta_learner:
  Fires only when pred_label NOT in BERT's subtype set
  "annoyance" ∈ anger subtypes → guard SILENTLY accepts wrong output
  BERT's 98.7% anger signal is completely ignored
```

---

## All Identified Issues

### Issue 1 — GoEmotions input noise (core problem)
When GoEmotions is confused (max score < ~0.35), feeding its raw distribution into the
feature vector poisons the neural network input. A 22% "approval" signal across 28
classes is barely above uniform noise (1/28 = 3.6%), yet it gets the same feature weight
as a 70% confident prediction.

**Affected patterns:** sentences with intensifiers before emotion words ("absolutely
furious", "completely devastated", "totally disgusted"), first-person direct statements
("I'm furious"), and rare emotion word combinations that don't appear in Reddit data.

### Issue 2 — BERT guard doesn't protect against GoEmotions uncertainty
The guard fires only when `pred_label not in BERT_SUBTYPES[bert_top]`. When the
meta-learner still manages to land in the right cluster (e.g., "annoyance" for anger),
the guard stays silent even when BERT is 99% confident and GoEmotions is noise.

### Issue 3 — BERT anchor in training penalizes valid sub-type predictions
In `trainer/models.py`, the BERT anchor loss fires for ALL high-confidence BERT
predictions. When BERT=joy(0.8) and the model correctly predicts "love" (a joy sub-type),
the anchor still creates a loss pushing the model toward "joy" instead. This trains the
model to prefer coarser Ekman labels over fine-grained correct ones.

### Issue 4 — Architectural information asymmetry
GoEmotions gets `Linear(28→64)`, BERT gets `Linear(7→64)`. GoEmotions carries 4× more
input signal. The load-balance loss (`5e-4`) prevents total collapse but can't overcome
the information gap — GoEmotions dominates gate weights even when its signal is noise.

---

## Additional Ideas Considered

| Idea | Description | Verdict |
|---|---|---|
| BERT-GoE affinity weighting (always-on) | Multiply every GoE score by its BERT Ekman affinity weight before encoding | Strong — works continuously, not just on edge cases |
| Explicit disagreement feature | Add 1-dim BERT-GoE conflict signal: `bert_conf × (1 - GoE_in_BERT_subtype_mass)` | Good — lets the network learn to act on conflict |
| GoE entropy as gate modifier | Reduce GoE gate weight when GoE entropy is very high | Moderate — treats symptom not cause |
| Replace GoEmotions model | Swap `bhadresh-savani/bert-base-go-emotion` for a model trained on more direct speech | High impact but large change, needs evaluation |
| Temperature scaling on GoE outputs | Sharpen GoE probability distribution | Won't fix wrong label, just calibration |

---

## Master Fix Plan

**Principle:** When GoEmotions is noisy (uncertain), boost BERT's influence.
All three changes implement this principle at different layers of the stack.

---

### Change 1 — Pre-correct GoEmotions input in `build_feature_vector`
**File:** `central_responder_service/meta_learner.py`

**When:** `goe_max < 0.35 AND bert_max > 0.80`

**What:** Replace the GoEmotions block in the feature vector with BERT-informed scores.
Take the raw GoEmotions scores within BERT's subtype set, normalize to sum to 1.0,
zero out everything outside the subtype. The neural network gets a coherent GoEmotions
signal instead of noise.

**Effect:** Immediate, no retraining required. The model sees clean input for the cases
where GoEmotions is provably wrong.

**Thresholds:**
- `UNCERTAIN_GOE_THRESH = 0.35` — below this, GoEmotions is near-random for 28 classes
- `CONFIDENT_BERT_THRESH = 0.80` — above this, BERT's cluster assignment is reliable

---

### Change 2 — Fix BERT guard in `predict_with_meta_learner`
**File:** `central_responder_service/meta_learner.py`

**Current behavior:** Guard fires only when `pred_label not in BERT_SUBTYPES[bert_top]`

**New behavior:** Guard also fires when `goe_max < UNCERTAIN_GOE_THRESH AND bert_max > CONFIDENT_BERT_THRESH`, regardless of whether pred_label is in the subtype set. A weak
"annoyance" with BERT screaming anger at 98.7% should not be silently accepted.

**Blend coefficient:** Reduce from `min(bert_max * 0.95, 0.85)` to `min(bert_max * 0.70, 0.65)`
to guide rather than override the neural network's output.

**Effect:** Immediate, no retraining required.

---

### Change 3 — Fix BERT anchor loss in training
**File:** `central_responder_service/trainer/models.py`

**Current behavior:** `bert_anchor_mask = (bert_conf_b > 0.75) & (bert_cidx_b >= 0)`
This fires for ALL high-confidence BERT samples, even when the model prediction is
already a valid GoEmotions sub-type of BERT's Ekman label.

**New behavior:** Only apply the BERT anchor loss when the model's argmax prediction is
NOT in BERT's subtype set. When the model correctly predicts "love" and BERT says "joy",
no anchor loss — love IS joy. Only fire when the model predicts something orthogonal
(e.g., model=approval, BERT=anger).

**Effect:** Next training cycle. Stops the model from being trained to prefer coarse
Ekman labels over fine-grained correct ones.

---

## Implementation Priority

| Priority | Change | Effect | Requires retrain |
|---|---|---|---|
| 1 (immediate) | Change 1 — pre-correct GoE input | Fixes core feature noise | No |
| 2 (immediate) | Change 2 — fix BERT guard | Catches any remaining leakthrough | No |
| 3 (next cycle) | Change 3 — fix BERT anchor | Better model going forward | Yes |

Changes 1 and 2 can be deployed together and tested immediately. Change 3 improves
the quality of the next automatically-triggered training cycle.

---

## Testing Checklist

After implementing Changes 1 and 2, verify these sentences route correctly:

- "I'm absolutely furious right now" → anger (not annoyance/approval)
- "I am completely devastated" → sadness/grief (not neutral)
- "I totally love this!" → love/joy (should still work — GoE is high confidence here)
- "I hate this so much" → anger (should still work — already working)
- "I'm really happy today" → joy (should still work — already working)
- "This is absolutely amazing" → admiration/excitement (GoE should be fine here — approval is correct for Reddit-style approval statements)

The last case is important: "absolutely amazing" should STILL map to approval/admiration.
The fix should only fire when GoEmotions is UNCERTAIN (< 0.35) — high-confidence GoEmotions
predictions should not be touched.

---

## Key Files

| File | Role |
|---|---|
| `central_responder_service/meta_learner.py` | `build_feature_vector` (Change 1), `predict_with_meta_learner` (Change 2) |
| `central_responder_service/trainer/models.py` | `train_gating_network` BERT anchor (Change 3) |
| `shared/constants.py` | `EMOTION_LABELS`, `BERT_LABELS` — label ordering |
| `goemotions_service/main.py` | GoEmotions model source (`bhadresh-savani/bert-base-go-emotion`) |
