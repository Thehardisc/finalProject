# nlp Knowledge Base
Last updated: 2026-06-21

## Architecture
Three parallel NLP services, all reading `preprocessed_stream`, writing to `partial_analysis_stream`.

### VADER service
- Model: VADER lexicon (vaderSentiment)
- Output fields: `vader_neg, vader_neu, vader_pos, vader_compound` (4 floats)
- model_name: `"vader"`
- Feature vector slot: [0:4]

### BERT service
- Model: `j-hartmann/emotion-english-distilroberta-base`
- Output: 7-class Ekman probabilities
- Labels (BERT_LABELS): anger, disgust, fear, joy, neutral, sadness, surprise
- model_name: `"basic_bert"`
- Feature vector slot: [4:11]

### GoEmotions service
- Model: `bhadresh-savani/bert-base-go-emotion`
- Output: 28-class GoEmotions probabilities + emoji_scores dict
- Labels: EMOTION_LABELS in shared/constants.py (28 classes)
- Emoji scoring: `emoji.demojize(text)` → GoEmotions labels
- model_name: `"go_emotions"`
- Feature vector slot: [11:39]
- Note: emoji_scores are forwarded but NOT currently in meta-learner feature vector

## Known Issues
- **GoEmotions circular labeling** [ISS-N003, FIXED 2026-06-22]: Reduced from ~37% to 21.1% of final training set (9768/46323 samples). GoE gate: was inflated (estimated >0.35), now 0.245 in deployed model. Trainer deployed new model with acc=0.6644 (↓3.59% from 0.7003 — old accuracy was inflated by circular labeling). Cache ID v4 active.
- **GoEmotions confidence not exposed** [ISS-N005, FIXED 2026-06-22]: `goe_confidence = max(EMOTION_LABELS scores)` now appended to `analyze()` output. Available in `partial_analysis_stream` event for downstream use.
- emoji_scores computed but unused in meta-learner — potential signal being wasted.

## Improvement Queue
- **[Med]** Use `goe_confidence` in central_responder to dynamically suppress GoE gate when max < 0.15 — requires change in `meta_learner.py:predict_with_meta_learner` or `GatingEnsembleNet`.
- **[Med]** Integrate emoji_scores into feature vector — either as 28-dim additive signal or as separate 28-dim block (would require feature vector expansion from 116 to 144 dims).
- **[Low]** Cache BERT tokenizer per service instance (already done, verify on restart).

## Cross-Agent Dependencies
- Provides: VADER[0:4], BERT[4:11], GoE[11:39] blocks to **meta_learner**
- Provides: GoE 28-dim history to **trajectory** (written by central_responder after aggregation)
- Depends on: **pipeline** for `preprocessed_stream` messages
- Depends on: **infra** for Redis consumer group setup

## Inter-Agent Requests (Pending)
*None*

## Recent History
- 2026-06-22: FIXED ISS-N003 fully — trainer cycle completed, GoE gate 0.245 (healthy), model deployed (acc=0.6644)
- 2026-06-22: FIXED ISS-N005 (goe_confidence added to analyze() output)
- 2026-06-20: Training NLP rebuild in progress — EmpatheticDialogues BERT 12%, GoE 5% (ETA ~40 min)
- 2026-06-20: GoE circular labeling problem identified; dair-ai/emotion (16K) + EmpatheticDialogues (25K) added to reduce bias
