# qa_suite — Emotion Engine QA

A **self-contained** pytest suite that exercises the InnerLink emotion engine via
input *equivalence classes* plus structural invariants and a light performance
probe. It lives outside the service directories and imports the real pipeline
helpers (`meta_learner`, `shared.constants`) rather than copying them — nothing
under the service folders is modified.

## Why the thresholds differ from the original QA doc

The original document proposed generic SLAs (`confidence > 0.85`, `latency < 200ms`,
fixed mixed-emotion splits). Those don't match this system: `meta_confidence` is a
softmax over **28** GoEmotions classes, and a message crosses ~10 services running
4 transformers + an LSTM. The bands in `thresholds.py` are **calibrated to observed
behaviour** and cited there. Re-derive after model changes with `calibrate.py`.

## Layout

| File | Stack? | Covers |
|---|---|---|
| `conftest.py` | — | sys.path bootstrap, `analyze` fixture, live `post_and_wait`, markers |
| `thresholds.py` | — | calibrated assertion bands (single source) |
| `corpora.py` | — | per-emotion labeled corpus (28 labels) + deterministic fuzz generator |
| `calibrate.py` / `build_goemotions_corpus.py` | — | refresh thresholds / rebuild the real-data sample |
| `data/edge_messages.json` | — | **authored** edge-case messages, each with the emotion it should map to |
| `data/conversations.json` | — | multi-turn conversations with per-turn emotion + expected valence arc |
| `data/goemotions_sample.jsonl` | — | 2000 **real** human-labeled GoEmotions messages |
| `test_functional.py` | no | clear / mixed / vague-ironic / minimal / modifiers / emoji / determinism / boundary |
| `test_invariants.py` | no | feature-vector shape, GoE gate cap ≤0.50, gate-vector shape, score normalisation |
| `test_messages.py` | no (+`@slow` agg) | edge-case messages: strict per-case (default) + aggregate family gate + per-edge report |
| `test_conversation.py` | no (+`@slow` gen) | curated dialogues per-case + **3000 generated** conversations (per-turn accuracy + valence-arc) |
| `test_coverage.py` | no (`@slow`) | per-emotion coverage — aggregate family-accuracy gate + per-family + exact-label report |
| `test_goemotions.py` | no (`@slow`) | benchmark on 2000 real messages vs human gold (top-1 + family accuracy) |
| `test_robustness.py` | no (`@slow`) | **3000 fuzz** inputs (typos/leet/emoji/unicode/…) → never crashes, always a valid label |
| `test_live.py` | **yes** (`@e2e`) | sequential context, sarcasm, full payload contract, latency + throughput |

**~3116 tests.** Default run = **89** content tests (functional, invariants, edge messages, curated
conversations). `@slow` batteries (**3019** cases: 3000 fuzz + 3000 generated-conversation turns +
2000 real GoEmotions + corpus + edge aggregate) are opt-in, ~7 min. Live tests are `@e2e`. Big
labeled corpora use an **aggregate accuracy gate** (prints misses, stays green) rather than per-case
asserts on noisy data.

## Running

### Quick: wrapper scripts

```bash
./qa_suite/run_qa.sh              # offline, 89 content tests | Windows: qa_suite\run_qa.bat
./qa_suite/run_qa.sh slow         # big batteries (~3019, 7m)| qa_suite\run_qa.bat slow
./qa_suite/run_qa.sh live         # @e2e (auto-loads .env)    | qa_suite\run_qa.bat live
./qa_suite/run_qa.sh all          # offline + live, no slow   | qa_suite\run_qa.bat all
./qa_suite/run_qa.sh full         # EVERYTHING                | qa_suite\run_qa.bat full
./qa_suite/run_qa.sh calibrate    # refresh thresholds       | qa_suite\run_qa.bat calibrate
```

To regenerate the real-data sample: `python qa_suite/build_goemotions_corpus.py --n 2000`.

### Full action report (`-vvv`)

Add `-vvv` (no `-s` needed) to print a full block for every action — input, predicted
label + confidence, expected + ✓/✗, VADER, VAD, top-5 emotions, sarcasm, gate weights.
Emitted through pytest's terminal writer so it shows even with capture on. Silent at lower
verbosity, so normal runs stay clean.

```bash
python -m pytest qa_suite/test_functional.py -vvv          # per-action report for one file
./qa_suite/run_qa.sh offline -vvv                          # via the wrapper
```
Example block:
```
━━ test_clear_emotion[I'm absolutely furious] — PASSED · 1 action(s) ━━
  [1] input    : "I'm absolutely furious right now"
      predicted: anger  (conf 0.87)
      expected : one of {anger, annoyance}   ✓
      vader    : compound -0.61  neg 0.50 neu 0.50 pos 0.00
      vad      : valence -0.48 arousal +0.66 dominance +0.49
      top-5    : anger 0.78 · neutral 0.12 · annoyance 0.09 · disapproval 0.01 · fear 0.01
      signals  : sarcasm 0.00   gate [v,b,g,vad,ctx] 0.17 0.17 0.45 0.12 0.09
```
Aggregate tests (coverage/goemotions/edge) also surface their summary line; live tests show
`LIVE in : … / dominant : … (round-trip Xs)`. Note `-vvv` reports **every** action, so the big
batteries print thousands of blocks — target a specific file or use `-k` to keep it readable.
Extra pytest args pass through, e.g. `./qa_suite/run_qa.sh offline -k emoji -v`.
The scripts prefer the project `.venv` and run from the repo root.

### Manual

```bash
# 1. Calibrate thresholds (offline; loads HF models)
python qa_suite/calibrate.py --json /tmp/baseline.json

# 2. Offline gate — no docker needed
python -m pytest qa_suite/test_functional.py qa_suite/test_invariants.py -v

# 3. Live (needs: docker compose up -d, and INTERNAL_API_KEY exported)
export $(grep -v '^#' .env | xargs)
python -m pytest qa_suite/test_live.py -v -m e2e

# Everything
python -m pytest qa_suite/ -v
```

The offline files pass in both meta-learner and rule-based-fallback modes
(confidence-band assertions self-skip without a trained pkl). `test_live.py`
self-skips when the stack is down.

## Notes / scope

- **Modifiers & negation** are asserted on VADER's `compound` only (the
  deterministic surface). VADER has built-in negation/intensifier handling;
  BERT/GoEmotions are model-driven, so a final-label flip is *not* asserted.
- **Out of scope** (per agreement): 50 msg/s sustained load and 4-hour soak.
  The throughput test is a light sequential burst that records msgs/sec.
