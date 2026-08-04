# InnerLink QA Test Calibration — Portable Prompt

CONTEXT: InnerLink is a real-time emotion-analysis chat system. Messages flow through
parallel NLP analyzers (VADER lexicon, BERT-7-Ekman, GoEmotions-28 blend) into a
meta-learner (GatingEnsembleNet) that fuses them, plus an LSTM trajectory predictor.
The test suite lives in `qa_suite/` (pytest), with per-test thresholds centralized in
`qa_suite/thresholds.py` so every test asserts against a named constant, never a bare
literal. Tests run in 3 modes: offline (no stack needed), slow (large corpora, opt-in),
live/e2e (needs the full docker stack up).

TASK: Implement or update the equivalent tests below in the target system, using the
same methodology: (1) measure the system's actual current baseline for each metric,
(2) set the threshold at baseline minus a safety margin — tight enough to catch a real
regression, loose enough to survive normal noise/retrain variance, (3) re-run the full
suite after every threshold change to confirm nothing that should pass now fails.

**IMPORTANT:** The numeric values below are calibrated to THIS system's specific model
and corpora. Do NOT copy them verbatim into a different model/dataset — re-measure that
system's own baseline first, then apply the same margin logic. The values are included
here as a worked reference for what "reasonable margin" looks like per metric type.

---

## Functional tests (7)

### 1. Single-message emotion analysis
Clear emotion, e.g. "furious", "thank you so much".

- **Metric:** final classifier confidence
- **Threshold:** `CLEAR_CONF_FLOOR = 0.83`
- **Measured baseline:** min passing confidence 0.882 across a 12-sentence clear-emotion
  battery (12/12 pass) — margin 0.05
- **How derived:** run the battery, take the minimum confidence among passing cases,
  subtract a small fixed margin (~0.05 on a 0–1 confidence scale)

### 2. Multi-emotion / mixed-emotion message
E.g. "thrilled but terrified", "hilarious but infuriating".

- **Metric:** minimum score among the secondary (non-dominant) qualifying emotion signal
- **Threshold:** `MIXED_FLOOR = 0.10`
- **Measured baseline:** weakest secondary signal across 3 curated mixed-emotion cases
  was 0.138 — margin 0.038
- **Assertion shape:** at least 2 emotion labels score ≥ floor, AND both expected emotion
  clusters appear in the top-6 raw-ranked labels (two independent checks — floor
  gates noise, top-6 check gates cluster coverage)

### 3. Ambiguous / ironic / sarcastic message
E.g. "ok then", "oh great, exactly what I needed".

- **Metric:** entropy of the emotion-score distribution (for vague, non-sarcastic
  ambiguity) + a dedicated sarcasm classifier score (for irony)
- **Threshold:** `VAGUE_ENTROPY_MIN = 0.55`
- **Measured baseline:** clear (unambiguous) inputs top out at entropy 0.42 — the vague
  threshold must sit above that ceiling; 0.55 already has 0.13 margin, no change needed
- **Sarcasm side:** separate classifier, gate is a mean-score separation between
  sarcastic and sincere batteries (>0.15 gap), not a single scalar threshold

### 4. Negation & intensifiers
E.g. "this is good" vs "this is not good"; "I am happy" vs "I am very happy".

- **Metric A:** valence delta caused by negation
  - Threshold: `NEGATION_MIN_DELTA = 0.5`
  - Measured: 0.782 delta (+0.440 → −0.341) — margin 0.28
- **Metric B:** magnitude ratio caused by an intensifier
  - Threshold: `INTENSIFIER_MIN_RATIO = 1.02`
  - Measured: 1.069 ratio (+0.572 → +0.612) — margin 0.05 (intensifier effects are
    inherently small on this lexicon-based signal, so keep the margin proportionally
    tighter here than elsewhere)

### 5. Emoji analysis
E.g. "lunch" vs "lunch 😀" vs "lunch 😢".

- **Metric:** minimum |valence delta| caused by an emoji, in either direction
- **Threshold:** `EMOJI_MIN_DELTA = 0.25`
- **Measured:** 0.361 (positive emoji), 0.477 (negative emoji) — margin 0.11 below the
  weaker of the two

### 6. Message sequence with shared conversational context
Multi-turn dialogue, valence arc over time.

- **Metric A:** per-turn family-accuracy (predicted emotion falls in the correct broad
  family: joy/anger/sadness/fear/surprise/neutral)
  - Curated corpus (12 dialogues / 53 turns): `CONV_FAMILY_GATE = 0.68`, measured 0.755
    — margin 0.075 (small N, keep extra buffer)
  - Generated corpus (3000 dialogues / 13,250 turns): `CONV_GEN_FAMILY_GATE = 0.88`,
    measured 0.961 — margin 0.081 (large N, statistically stable, tighter is safe)
- **Metric B:** valence-arc trend match (does the conversation's predicted trajectory
  match its scripted arc — escalating/de-escalating/flat/etc.)
  - Threshold: `CONV_ARC_GATE = 0.80`, measured 0.845 (2536/3000) — margin only 0.045,
    kept narrow deliberately because a few specific arc types (celebration, good_news,
    disgust, gratitude) are known weak spots and carry most of the misses
- Also verify structurally: valence actually changes turn-to-turn (context isn't
  stateless), and any trajectory-prediction state (e.g. an LSTM prior) has the right
  dimensionality

### 7. Conversation-level insights / analytics display
Dominant emotion, trend, intensity shown after a conversation.

- **Metric:** latency from triggering analysis to the insights becoming available
- **Threshold:** `ANALYSIS_LATENCY_BUDGET_SEC = 5.0s`
- **Measured:** 0.01s on a 3-turn conversation — this endpoint is a fast Redis+DB round
  trip, so the budget is deliberately generous (not tightly margined like the other
  gates) because cost scales with conversation length and this wasn't load-tested at
  realistic scale. If you have a way to measure this against a long/real conversation
  history, re-derive a tighter number instead of trusting the 5.0s as-is.
- Also assert correctness: returned trend/mood direction actually matches the
  scripted emotional direction of the test conversation (e.g. a written escalating
  negative arc should classify as escalating, not stable)

---

## Non-functional tests (8)

### 1. Response time
- **Metric:** warm end-to-end latency for a single message (ingestion → full pipeline
  result), after a throwaway warm-up request to exclude cold-start/model-load time
- **Threshold:** `PERF_LATENCY_BUDGET_SEC = 3.0s`
- **Measured:** 0.24–0.36s warm — ~9x margin, deliberately loose to absorb container/
  network jitter across environments, while still catching an order-of-magnitude
  regression

### 2. Real-time operation
No perceptible delay in a live chat.

- **Metric:** sustained throughput over a short burst of sequential messages
- No dedicated numeric gate beyond the latency budget above; measured 4.0–4.1 msg/s
  sequential round-trip as a reference point

### 3. Robustness to malformed/garbled text
Typos, slang, leetspeak, emoji spam, excessive punctuation, empty/whitespace-only input.

- **Metric:** fraction of inputs that produce a valid label without crashing
- **Threshold:** must be 100% — this is a boolean must-never-fail gate, not a tunable
  percentage; do not loosen it
- **Measured:** 3000/3000 (100%) on a large auto-generated fuzz corpus, plus a smaller
  hand-written robustness battery, also 100%

### 4. Visual usability
Can a user infer the emotional state from color/UI alone, no text?

- **NOT AUTOMATABLE.** This is a human-perception check. Do not attempt to fake a
  numeric proxy for it — flag it explicitly as requiring manual UX review instead of
  silently skipping it or inventing a meaningless metric.

### 5. Stability over time / no gradual degradation under sustained load
- **Metric:** ratio of average per-message latency in the second half of a sustained
  sequential burst vs the first half
- **Threshold:** `SOAK_REGRESSION_FACTOR = 1.5` (plus a small fixed +0.3s absolute
  tolerance to avoid dividing by near-zero baseline latencies)
- **Measured:** ~1.0–1.1x ratio (essentially flat) — margin comfortably below 1.5x while
  still catching a genuine 50%+ slowdown trend
- Cheaper alternative to a long separate soak loop: extend an existing throughput/
  burst test to also record and compare per-message latency, rather than adding a
  new heavy test

### 6. Availability / graceful degradation when an optional component fails
- **Metric:** pipeline still produces a valid result when a non-critical/optional
  component (in this system: an optional context-enrichment service) is stopped
- No numeric gate — a functional pass/fail assertion (result still produced, no
  crash) plus confirmation the component auto-recovers after restart
- **SAFETY NOTE:** this test mutates live infrastructure (stops/starts a real
  container or service). Make it strictly opt-in (e.g. gated behind an explicit
  environment variable), skipped by default, and always restore the stopped
  component in a try/finally so a test failure can't leave the system degraded.
  Never wire this into a default/CI test run without that opt-in gate.

### 7. Process completeness
A message fully traverses every pipeline stage end-to-end, arriving with all expected
fields.

- **Metric:** pass/fail — message ingested at the entry point is observed emerging from
  the final output stream/queue within a generous timeout, with all required result
  fields present
- No numeric accuracy gate; this is a structural/completeness check

### 8. Data persistence
A message and its analysis result are durably stored and retrievable.

- **Metric:** pass/fail — after posting a message and waiting for the async persistence
  write, fetch it back through the system's normal read path (not by querying the
  database directly) and confirm the content and analysis match what was produced
- No numeric gate; poll with a generous timeout budget to absorb async write lag
  rather than asserting on a fixed sleep

---

## Methodology summary

Apply this pattern to any new metric you add:

- Never hardcode a bare number in a test body — always a named constant in the
  thresholds/config file, with a comment stating the measured baseline it was derived
  from and the date.
- Margin size should scale with corpus size and inherent noise: small hand-curated
  corpora and inherently small effect sizes (e.g. an intensifier's effect on a lexicon
  score) get a bigger relative margin; large auto-generated corpora with many samples
  can be margined tighter.
- Boolean must-never-fail properties (no crash, valid output shape) get a 100%
  threshold, never loosened for convenience.
- After every threshold change, re-run the full suite (all modes/tiers) before
  considering the change done — a threshold that isn't validated against a real passing
  run is a guess, not a calibration.
- Anything that cannot be reduced to an automatable check (subjective human perception,
  aesthetic judgment) should be explicitly labeled as a manual/out-of-scope test rather
  than papered over with a fake metric.
