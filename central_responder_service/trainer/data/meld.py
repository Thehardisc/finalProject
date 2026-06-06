"""
trainer/data/meld.py — MELD multi-turn conversation feature extraction.
"""

import gc
import json
import os
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np

from shared.constants import (
    EMOTION_LABELS, FEATURE_DIM, CDM_CTX_DIM, PRIOR_DIM, N_CDM_STATES,
    CTX_RESIDENCY, CTX_TRANSITION, CTX_ABRUPTNESS, CTX_COHERENCE,
    CTX_ENTROPY, CTX_SPK_DIVERGENCE, CTX_VELOCITY, CTX_ACCELERATION,
    CTX_HIST_POS, CTX_HIST_NEU, CTX_HIST_NEG, CTX_RESONANCE, CTX_VOLATILITY,
    CTX_CURR_VALENCE, CTX_MSG_LENGTH, CTX_LATENCY_MS,
    CTX_HMM_CONF, CTX_HMM_ENTROPY, CTX_HMM_EMISSION, CTX_HMM_NEXT3,
    CTX_INTENT_STAB,
)
from shared.utils.logger import get_logger
from meta_learner import build_feature_vector
from trainer.utils import _run_batch, _vader
from trainer.data.synthetic import _LABEL_TO_INTENT

logger = get_logger("trainer")

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl"))

# MELD 7 labels → GoEmotions 28 (direct matches)
_MELD_TO_GOEMOTION: dict = {
    "anger":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "joy":      "joy",
    "neutral":  "neutral",
    "sadness":  "sadness",
    "surprise": "surprise",
}

_MELD_VALENCE: dict = {
    "joy":      0.80,
    "surprise": 0.15,
    "neutral":  0.00,
    "anger":   -0.75,
    "disgust": -0.70,
    "fear":    -0.65,
    "sadness": -0.70,
}


def _download_meld_raw(out_path: Path, max_rows: int = 3000) -> None:
    """Download MELD and save only (dialogue_id, utterance_id, text, emotion, speaker)
    to a compact JSON. Called BEFORE NLP models are loaded so HuggingFace dataset and
    BERT/GoEmotions don't share RAM simultaneously."""
    try:
        from datasets import load_dataset
        slice_size = min(max_rows * 2, 4000)
        logger.info(f"[MELD] Pre-downloading raw data (train[:{slice_size}]) — no NLP models in memory yet...")
        ds = load_dataset("declare-lab/MELD", split=f"train[:{slice_size}]")
        rows = []
        for row in ds:
            rows.append({
                "d": str(row.get("Dialogue_ID", "")),
                "u": int(row.get("Utterance_ID", 0)),
                "t": str(row.get("Utterance", "")).strip(),
                "e": str(row.get("Emotion",   "neutral")).lower(),
                "s": str(row.get("Speaker",   "")).strip(),
            })
        del ds
        gc.collect()
        with open(out_path, "w") as f:
            json.dump(rows, f)
        logger.info(f"[MELD] Raw cache saved → {out_path} ({len(rows)} rows)")
    except Exception as e:
        logger.warning(f"[MELD] Pre-download failed: {e} — MELD will be skipped.")


def _meld_build_cdm(
    history: list,          # list of dicts: {valence, intent_state, speaker, goe_dist}
    current_valence: float,
    current_intent: int,
    current_speaker: str,
    current_text: str,
) -> np.ndarray:
    """
    Build a real 40-dim CDM context vector from MELD conversation history.
    Uses the same slot layout as context_engine_service so the meta-learner
    trains on vectors that look like real inference-time CDM output.
    """
    ctx = np.zeros(CDM_CTX_DIM, dtype=np.float32)
    n   = len(history)

    # ── CDM intent one-hot [0:15] ──────────────────────────────────────────
    ctx[current_intent] = 1.0

    # ── state_residency [15] — how long in this state ─────────────────────
    streak = 1
    for h in reversed(history):
        if h["intent_state"] == current_intent:
            streak += 1
        else:
            break
    ctx[CTX_RESIDENCY] = min(streak / max(n + 1, 1), 1.0)

    # ── transition_path [16:19] — last 3 intent indices / N_CDM_STATES ────
    recent = [h["intent_state"] for h in history[-3:]]
    for i, s in enumerate(recent):
        ctx[CTX_TRANSITION.start + i] = s / N_CDM_STATES

    # ── entry_abruptness [19] — sudden valence jump ────────────────────────
    prev_val = history[-1]["valence"] if history else 0.0
    ctx[CTX_ABRUPTNESS] = min(abs(current_valence - prev_val), 1.0)

    # ── topic_coherence [20] — stable intent = high coherence ─────────────
    if n > 0:
        same = sum(1 for h in history[-5:] if h["intent_state"] == current_intent)
        ctx[CTX_COHERENCE] = same / min(n, 5)
    else:
        ctx[CTX_COHERENCE] = 0.5

    # ── emotion_entropy [21] — diversity of previous labels ───────────────
    if n > 0:
        ec     = Counter(h["intent_state"] for h in history[-5:])
        total  = sum(ec.values())
        probs  = [c / total for c in ec.values()]
        ent    = -sum(p * np.log(p + 1e-9) for p in probs)
        ctx[CTX_ENTROPY] = float(np.clip(ent / np.log(N_CDM_STATES), 0.0, 1.0))

    # ── speaker_divergence [22] ────────────────────────────────────────────
    if history:
        ctx[CTX_SPK_DIVERGENCE] = float(history[-1]["speaker"] != current_speaker)

    # ── velocity [23] and acceleration [24] ───────────────────────────────
    velocity = current_valence - prev_val
    ctx[CTX_VELOCITY] = float(np.clip(velocity, -1.0, 1.0))
    if len(history) >= 2:
        prev_velocity = history[-1]["valence"] - history[-2]["valence"]
        ctx[CTX_ACCELERATION] = float(np.clip(velocity - prev_velocity, -1.0, 1.0))

    # ── valence history [25:28] ────────────────────────────────────────────
    all_vals = [h["valence"] for h in history] + [current_valence]
    ctx[CTX_HIST_POS] = float(np.mean([v > 0.2  for v in all_vals]))
    ctx[CTX_HIST_NEU] = float(np.mean([abs(v) <= 0.2 for v in all_vals]))
    ctx[CTX_HIST_NEG] = float(np.mean([v < -0.2 for v in all_vals]))

    # ── topic_resonance [28] — reuse coherence as proxy ───────────────────
    ctx[CTX_RESONANCE] = ctx[CTX_COHERENCE]

    # ── volatility [29] — std of valence trajectory ────────────────────────
    if len(all_vals) > 1:
        ctx[CTX_VOLATILITY] = float(np.clip(np.std(all_vals), 0.0, 1.0))

    # ── current_valence [30] ───────────────────────────────────────────────
    ctx[CTX_CURR_VALENCE] = float(np.clip(current_valence, -1.0, 1.0))

    # ── message_length [31] (raw chars — normalized by build_feature_vector)
    ctx[CTX_MSG_LENGTH] = float(len(current_text))

    # ── latency_ms [32] — 0 (not available in MELD) ───────────────────────
    ctx[CTX_LATENCY_MS] = 0.0

    # ── HMM slots [33:39] — moderate confidence since we have real context ─
    ctx[CTX_HMM_CONF]    = 0.65
    ctx[CTX_HMM_ENTROPY] = 0.40
    ctx[CTX_HMM_EMISSION] = -1.0
    ctx[CTX_HMM_NEXT3.start]     = 0.50
    ctx[CTX_HMM_NEXT3.start + 1] = 0.30
    ctx[CTX_HMM_NEXT3.start + 2] = 0.20

    # ── intent_stability [39] — same as residency ─────────────────────────
    ctx[CTX_INTENT_STAB] = ctx[CTX_RESIDENCY]

    return ctx


def extract_meld_features(
    vader_analyzer,
    bert_analyzer,
    goe_analyzer,
    max_utterances: int = 1500,
    batch_size: int = 32,
) -> tuple:
    """
    Build (X, y, has_cdm) from MELD with REAL conversation context.

    Unlike EmpatheticDialogues (topic-level label) or GoEmotions (single sentence),
    MELD provides multi-turn conversations where each utterance's CDM vector is
    built from the actual preceding messages in the same dialogue.

    This teaches the meta-learner:
      - "Previous messages were angry → current ambiguous message = probably anger"
      - "Context is neutral → trust NLP signal directly"
      - When context genuinely changes the answer vs when NLP is sufficient

    has_cdm = True for all MELD samples with at least 1 prior utterance,
    so the CDM block is NOT zeroed during training — the context path gets
    real gradient signal for the first time.
    """
    cache_path = MODEL_PATH.parent / "meld_features_cache.pkl"
    cache_key  = f"meld_ctx_v1_{max_utterances}_{FEATURE_DIM}"
    empty      = np.empty((0, FEATURE_DIM), dtype=np.float32)

    logger.info(f"[MELD] Starting feature extraction (max_utterances={max_utterances})...")
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("cache_key") == cache_key:
                X, y, has_cdm = cached["data"]
                logger.info(f"[MELD] Cache hit — {len(y)} utterances with real context.")
                return np.array(X, dtype=np.float32) if X else empty, y, has_cdm
        except Exception as e:
            logger.warning(f"[MELD] Cache load failed: {e} — recomputing.")

    # ── Pass 1: load raw MELD rows ────────────────────────────────────────────
    raw_json_path = MODEL_PATH.parent / "meld_raw_cache.json"
    raw_rows = []
    if raw_json_path.exists():
        try:
            with open(raw_json_path) as f:
                data = json.load(f)
            for r in data:
                raw_rows.append((r["d"], r["u"], r["t"], r["e"], r["s"]))
            logger.info(f"[MELD] Loaded {len(raw_rows)} rows from pre-downloaded raw cache.")
        except Exception as e:
            logger.warning(f"[MELD] Raw cache read failed: {e} — trying live load.")
            raw_rows = []

    if not raw_rows:
        import urllib.request
        import zipfile
        import csv
        import io
        CSV_URLS = [
            "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
            "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/data/train_sent_emo.csv",
            "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/train_sent_emo.csv",
        ]
        csv_bytes = None
        for url in CSV_URLS:
            try:
                logger.info(f"[MELD] Trying CSV download: {url}")
                with urllib.request.urlopen(url, timeout=120) as resp:
                    size_mb = int(resp.headers.get("Content-Length", 0)) / 1e6
                    logger.info(f"[MELD] Downloading ({size_mb:.1f} MB)...")
                    csv_bytes = resp.read()
                logger.info(f"[MELD] Download complete ({len(csv_bytes)/1e6:.1f} MB).")
                break
            except Exception as e:
                logger.warning(f"[MELD] {url} failed: {e}")

        if not csv_bytes:
            logger.warning("[MELD] All CSV URLs failed — skipping MELD.")
            return empty, [], []

        try:
            if csv_bytes[:2] == b'PK':
                with zipfile.ZipFile(io.BytesIO(csv_bytes)) as zf:
                    csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
                    text_data = zf.read(csv_name).decode("utf-8")
            else:
                text_data = csv_bytes.decode("utf-8")

            reader = csv.DictReader(io.StringIO(text_data))
            for row in reader:
                raw_rows.append((
                    str(row.get("Dialogue_ID", "")),
                    int(row.get("Utterance_ID", 0)),
                    str(row.get("Utterance",    "")).strip(),
                    str(row.get("Emotion",      "neutral")).lower(),
                    str(row.get("Speaker",      "")).strip(),
                ))
            logger.info(f"[MELD] Parsed {len(raw_rows)} raw rows from CSV.")
            save_data = [{"d": d, "u": u, "t": t, "e": e, "s": s}
                         for d, u, t, e, s in raw_rows]
            with open(raw_json_path, "w") as f:
                json.dump(save_data, f)
            logger.info(f"[MELD] Raw cache saved → {raw_json_path}")
        except Exception as e:
            logger.warning(f"[MELD] CSV parse failed: {e} — skipping.")
            return empty, [], []

    # Group and sort by dialogue
    dialogues: dict = {}
    for did, uid, text, emo, spk in raw_rows:
        dialogues.setdefault(did, []).append((uid, text, emo, spk))
    for did in dialogues:
        dialogues[did].sort(key=lambda x: x[0])

    # Flatten to ordered list
    all_rows = []
    for did, utt_list in dialogues.items():
        for uid, text, emo, spk in utt_list:
            all_rows.append((did, uid, text, emo, spk))
    all_rows = all_rows[:max_utterances]
    all_texts = [r[2] for r in all_rows]

    logger.info(f"[MELD] {len(all_rows)} utterances across {len(dialogues)} dialogues. Running batched NLP...")
    t0 = time.time()

    total_texts = len(all_texts)
    vader_outs = []
    _chk = {int(total_texts * p) for p in (0.25, 0.50, 0.75)}
    for i, text in enumerate(all_texts):
        try:
            vader_outs.append({f"vader_{k}": v for k, v in _vader(vader_analyzer, text).items()})
        except Exception:
            vader_outs.append({})
        if i + 1 in _chk:
            logger.info(f"  [MELD/VADER] {i+1}/{total_texts} ({int((i+1)/total_texts*100)}%)")

    bert_outs = _run_batch(bert_analyzer, all_texts, batch_size=batch_size, label="MELD/BERT") if callable(bert_analyzer) else [{} for _ in all_texts]
    goe_outs  = _run_batch(goe_analyzer,  all_texts, batch_size=batch_size, label="MELD/GoE")  if callable(goe_analyzer)  else [{} for _ in all_texts]

    logger.info(f"[MELD] NLP done in {time.time()-t0:.0f}s. Building CDM vectors from conversation history...")

    # ── Pass 2: build features with real context ───────────────────────────
    conv_history: dict = {}

    features, labels, has_cdm_list = [], [], []
    _cdm_total = len(all_rows)
    _cdm_chk = {int(_cdm_total * p) for p in (0.25, 0.50, 0.75)}

    for _cdm_i, ((did, uid, text, meld_label, speaker), vader_out, bert_out, goe_out) in enumerate(zip(
        all_rows, vader_outs, bert_outs, goe_outs
    )):
        if _cdm_i + 1 in _cdm_chk:
            logger.info(f"  [MELD/CDM] {_cdm_i+1}/{_cdm_total} ({int((_cdm_i+1)/_cdm_total*100)}%) — {len(features)} vectors built so far")
        if not text:
            continue

        goemo_label = _MELD_TO_GOEMOTION.get(meld_label, "neutral")
        valence     = _MELD_VALENCE.get(meld_label, 0.0)
        intent_st   = _LABEL_TO_INTENT.get(goemo_label, (0, 0, 0))[0]

        history = conv_history.get(did, [])

        # First utterance in a conversation: no real context yet
        if not history:
            ctx        = np.zeros(CDM_CTX_DIM, dtype=np.float32)
            ctx[intent_st] = 1.0
            ctx[CTX_CURR_VALENCE] = valence
            ctx[CTX_MSG_LENGTH]   = float(len(text))
            ctx[CTX_HMM_CONF]     = 0.30
            prior      = [0.0] * PRIOR_DIM
            real_ctx   = False
        else:
            ctx      = _meld_build_cdm(history, valence, intent_st, speaker, text)
            prior    = history[-1]["goe_dist"]
            real_ctx = True

        fv = build_feature_vector(
            {"vader": vader_out, "basic_bert": bert_out, "go_emotions": goe_out},
            context_vector=ctx,
            trajectory_prior=prior,
        )
        features.append(fv.flatten())
        labels.append(goemo_label)
        has_cdm_list.append(real_ctx)

        # Update history for next utterance
        goe_dist = [float(goe_out.get(e, 0.0)) for e in EMOTION_LABELS]
        conv_history.setdefault(did, []).append({
            "valence":      valence,
            "intent_state": intent_st,
            "speaker":      speaker,
            "goe_dist":     goe_dist,
        })

    real_count = sum(has_cdm_list)
    logger.info(
        f"[MELD] Built {len(features)} feature vectors — "
        f"{real_count} with real context ({100*real_count//max(len(features),1)}%), "
        f"{len(features)-real_count} first-utterance (no context)."
    )

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"cache_key": cache_key, "data": (features, labels, has_cdm_list)}, f)
        logger.info(f"[MELD] Cache saved → {cache_path}")
    except Exception as e:
        logger.warning(f"[MELD] Could not save cache: {e}")

    return np.array(features, dtype=np.float32) if features else empty, labels, has_cdm_list
