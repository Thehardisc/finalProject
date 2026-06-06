"""
trainer/data/synthetic.py — Synthetic context vector generation and sentence loading.
"""

import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from shared.constants import (
    EMOTION_LABELS, FEATURE_DIM, CONTEXT_DIM, CDM_CTX_DIM, PRIOR_DIM, N_CDM_STATES,
)
from shared.utils.logger import get_logger
from meta_learner import build_feature_vector
from trainer.utils import _run_batch, _vader

logger = get_logger("trainer")

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl"))

# ── Label → likely CDM intent state (primary, secondary, tertiary) ───────────────
_LABEL_TO_INTENT: dict = {
    'admiration':     (2,  14,  1),
    'amusement':      (4,  10,  0),
    'approval':       (14,  2, 11),
    'caring':         (12,  1,  9),
    'curiosity':      (10,  3,  0),
    'desire':         (1,  10,  0),
    'excitement':     (2,  14,  4),
    'gratitude':      (14,  1,  2),
    'joy':            (1,   2, 14),
    'love':           (1,  12,  9),
    'optimism':       (14,  1,  2),
    'pride':          (2,  11, 14),
    'relief':         (9,  14,  0),
    'realization':    (0,  10, 11),
    'anger':          (6,   5,  7),
    'annoyance':      (5,  13,  6),
    'disapproval':    (5,  11,  6),
    'disgust':        (6,   5,  0),
    'disappointment': (8,  13,  0),
    'embarrassment':  (8,   0, 13),
    'fear':           (3,   8,  0),
    'grief':          (8,  12,  0),
    'nervousness':    (3,   8, 13),
    'remorse':        (9,   8, 12),
    'sadness':        (8,  12,  0),
    'confusion':      (11,  3,  0),
    'neutral':        (0,  10, 11),
    'surprise':       (4,  10,  0),
}

# Labels with strong, unambiguous emotional intent — higher hmm_confidence
_STRONG_LABELS = frozenset({
    'anger', 'joy', 'love', 'grief', 'admiration', 'fear',
    'disgust', 'gratitude', 'pride', 'remorse', 'sadness', 'excitement',
})

# ── Approximate valence baselines for synthetic context generation ──────────────
_LABEL_BASE_VALENCE: dict = {
    'admiration':    0.62,  'amusement':      0.68,  'approval':    0.52,
    'caring':        0.58,  'curiosity':      0.18,  'desire':      0.42,
    'excitement':    0.78,  'gratitude':      0.72,  'joy':         0.82,
    'love':          0.82,  'optimism':       0.62,  'pride':       0.68,
    'realization':   0.12,  'relief':         0.52,  'surprise':    0.12,
    'anger':        -0.72,  'annoyance':     -0.48,  'disapproval': -0.52,
    'disgust':      -0.68,  'disappointment':-0.58,  'embarrassment':-0.42,
    'fear':         -0.62,  'grief':         -0.82,  'nervousness': -0.42,
    'remorse':      -0.58,  'sadness':       -0.68,
    'confusion':    -0.08,  'neutral':        0.00,
}


def build_synthetic_context_vector(
    label: str = None,
    mode: str = "train",
) -> list:
    """
    Generate a synthetic context vector for static dataset samples.

    The critical invariant: valence-related features are label-correlated but
    NOT deterministically derived from the label.  Three mechanisms enforce this:

      1. Gaussian noise  σ=0.35 on all valence scalars — SNR ≈ 1.8:1 for
         strong emotions (e.g. joy base=0.82, noise keeps [-0.2, 1.8] → clipped)
      2. Adversarial flip (25% chance) — injects the wrong polarity to force
         the network to treat context as a soft prior, not a cheat sheet
      3. Per-feature dropout (15%) — simulates missing sensors / cold starts

    Modes:
      "train"  — full augmentation (items 1–3 above)
      "val"    — moderate noise only (σ=0.20, no adversarial flip)
      "cold"   — all zeros (for live SQL data with no conversation history)
    """
    if mode == "cold":
        return [0.0] * CONTEXT_DIM

    rng = np.random.RandomState()  # unseeded — each call is independent

    # ── CDM intent state: label-correlated when label is known ──────────────────
    if label and label in _LABEL_TO_INTENT:
        primary, secondary, tertiary = _LABEL_TO_INTENT[label]
        # 20% adversarial: random state to prevent lookup-table memorization
        if mode == "train" and rng.random() < 0.20:
            cdm_state = int(rng.randint(0, N_CDM_STATES))
        else:
            p = rng.random()
            cdm_state = primary if p < 0.60 else (secondary if p < 0.85 else tertiary)
    else:
        dirichlet_alpha = [3.0] + [1.0] * (N_CDM_STATES - 1)
        cdm_state = int(rng.choice(N_CDM_STATES, p=rng.dirichlet(dirichlet_alpha)))
    cdm_one_hot     = [0.0] * N_CDM_STATES
    cdm_one_hot[cdm_state] = 1.0

    # ── Temporal / structural scalars (independent of label) ─────────────────
    residency  = float(rng.beta(2.0, 5.0))
    transition = [float(rng.randint(0, N_CDM_STATES) / float(N_CDM_STATES)) for _ in range(3)]
    abruptness = float(rng.beta(1.0, 3.0))

    # ── Semantic scalars (independent of label) ───────────────────────────────
    coherence      = float(rng.beta(3.0, 2.0))
    entropy        = float(rng.beta(2.0, 3.0))
    spk_divergence = float(rng.beta(1.0, 4.0))
    acceleration   = float(np.clip(rng.normal(0.0, 0.12), -1.0, 1.0))
    resonance      = float(rng.beta(2.0, 2.0))
    volatility     = float(rng.beta(1.5, 3.0))
    msg_length     = float(rng.beta(2.0, 5.0))
    latency_norm   = float(rng.beta(1.0, 4.0))

    # ── Valence scalars: label-correlated but heavily noisy ───────────────────
    noise_sigma = 0.35 if mode == "train" else 0.20
    base_val    = _LABEL_BASE_VALENCE.get(label, 0.0) if label else 0.0

    cur_valence  = float(np.clip(base_val + rng.normal(0.0, noise_sigma), -1.0, 1.0))
    velocity     = float(np.clip(rng.normal(0.0, 0.25),                   -1.0, 1.0))

    # Episodic memory: 3-dim sentiment vector (pos, neu, neg) correlated with label
    _pos_base = max(0.0, base_val)
    _neg_base = max(0.0, -base_val)
    hist_pos = float(np.clip(_pos_base * 0.6 + rng.uniform(0.0, 0.3), 0.0, 1.0))
    hist_neu = float(np.clip(0.5 - abs(base_val) * 0.4 + rng.uniform(-0.1, 0.1), 0.0, 1.0))
    hist_neg = float(np.clip(_neg_base * 0.6 + rng.uniform(0.0, 0.3), 0.0, 1.0))

    if mode == "train" and rng.random() < 0.25:
        cur_valence = -cur_valence
        hist_pos, hist_neg = hist_neg, hist_pos

    # ── HMM-derived features: label-correlated ───────────────────────────────
    is_strong   = label in _STRONG_LABELS if label else False
    conf_base   = float(rng.uniform(0.55, 0.85) if is_strong else rng.uniform(0.35, 0.65))
    alpha_raw   = rng.dirichlet([0.5] * N_CDM_STATES)
    alpha_raw[cdm_state] += conf_base * 6.0
    alpha_raw   /= alpha_raw.sum()
    hmm_conf    = float(alpha_raw.max())
    hmm_ent     = float(-np.sum(alpha_raw * np.log(alpha_raw + 1e-12)))
    hmm_emit    = float(rng.beta(3.0, 2.0) if is_strong else rng.beta(2.0, 3.0))
    top3_next   = sorted(rng.dirichlet([1.0] * 3).tolist(), reverse=True)
    intent_stab = float(rng.beta(3.0, 2.0) if label else rng.beta(1.0, 4.0))

    ctx = (
        cdm_one_hot            +   # [0:15]  CDM one-hot (15 intent states)
        [residency]            +   # [15]
        transition             +   # [16:19]
        [abruptness,               # [19]
         coherence,                # [20]
         entropy,                  # [21]
         spk_divergence,           # [22]
         velocity,                 # [23]
         acceleration,             # [24]
         hist_pos,                 # [25]
         hist_neu,                 # [26]
         hist_neg,                 # [27]
         resonance,                # [28]
         volatility,               # [29]
         cur_valence,              # [30]
         msg_length,               # [31]
         latency_norm,             # [32]
         hmm_conf,                 # [33]
         hmm_ent,                  # [34]
         hmm_emit,                 # [35]
        ] + top3_next +            # [36:39]
        [intent_stab]              # [39]
        + [0.0] * PRIOR_DIM        # [40:68] trajectory prior — zeros for static single-turn data
    )

    # Per-feature dropout: 15% of features zeroed on each training sample
    if mode == "train":
        mask = (rng.random(CONTEXT_DIM) > 0.15).astype(float)
        ctx  = [float(v) * m for v, m in zip(ctx, mask)]

    assert len(ctx) == CONTEXT_DIM, f"ctx dim={len(ctx)} != {CONTEXT_DIM}"
    return ctx


# ── Synthetic sentence generation for missing GoEmotions classes ──────────────

_SYNTHETIC_COUNTS: dict = {
    "neutral":     1069,
    "relief":       568,
    "amusement":    568,
    "confusion":    568,
    "realization":  568,
}
_SYNTHETIC_CLASSES = list(_SYNTHETIC_COUNTS.keys())
_BATCH_SIZE        = 250   # max sentences per Claude API call (fits in 4096 tokens)


def _generate_synthetic_sentences() -> dict:
    """
    Call Claude Haiku to generate synthetic training sentences for each missing class.
    Loops in batches of _BATCH_SIZE until the per-class target in _SYNTHETIC_COUNTS is met.
    """
    import anthropic
    client    = anthropic.Anthropic()
    result    = {}
    total_all = sum(_SYNTHETIC_COUNTS.values())
    done_all  = 0

    for label, target in _SYNTHETIC_COUNTS.items():
        sentences: list = []
        n_batches_est = -(-target // _BATCH_SIZE)
        logger.info(
            f"  [Synthetic] '{label}': target={target} sentences "
            f"(~{n_batches_est} batches of ≤{_BATCH_SIZE}, may need more if API returns fewer)"
        )
        batch_num = 0
        while len(sentences) < target:
            batch_num += 1
            needed = min(_BATCH_SIZE, target - len(sentences))
            logger.info(
                f"  [Synthetic]   batch {batch_num} for '{label}' "
                f"— requesting {needed}, have {len(sentences)}/{target} so far..."
            )
            prompt = (
                f"Generate {needed} diverse first-person situational sentences "
                f"expressing the emotion '{label}'. "
                "Style: EmpatheticDialogues 'situation' field — 1-2 sentences describing "
                "a real-life context where someone feels this emotion. "
                "Vary the settings: work, relationships, hobbies, discovery, everyday moments. "
                "One situation per line. No numbering, no bullets."
            )
            msg   = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            batch = [l.strip() for l in msg.content[0].text.splitlines() if l.strip()]
            sentences.extend(batch)
            logger.info(
                f"  [Synthetic]   batch {batch_num} done — "
                f"got {len(batch)}, total so far: {len(sentences)}/{target}"
            )
        result[label] = sentences[:target]
        done_all     += len(result[label])
        logger.info(
            f"  [Synthetic] '{label}' complete: {len(result[label])}/{target} sentences "
            f"[overall {done_all}/{total_all}]"
        )
    return result


def load_synthetic_features(vader_analyzer, bert_analyzer, goe_analyzer, batch_size: int = 32) -> tuple:
    """
    Load (or auto-generate) synthetic sentences for the 5 missing GoEmotions classes
    and process them through the same NLP pipeline as EmpatheticDialogues.

    Sentences are stored in MODEL_PATH.parent/synthetic_sentences.json (bind-mounted,
    persists container rebuilds). Computed features are cached in
    synthetic_features_cache.pkl keyed by the JSON file's MD5 hash — NLP only runs
    once per unique JSON file.
    """
    import hashlib

    json_path   = MODEL_PATH.parent / "synthetic_sentences.json"
    cache_path  = MODEL_PATH.parent / "synthetic_features_cache.pkl"
    empty       = np.empty((0, FEATURE_DIM), dtype=np.float32)

    # ── 1. Obtain JSON (generate if missing) ────────────────────────────────────
    if not json_path.exists() or json_path.stat().st_size == 0:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            logger.warning(
                "[Synthetic] ANTHROPIC_API_KEY is not set — cannot generate synthetic sentences. "
                "Add ANTHROPIC_API_KEY to .env and rebuild to cover: "
                f"{_SYNTHETIC_CLASSES}. Training will proceed with 23/28 classes."
            )
            return empty, [], []

        total_needed = sum(_SYNTHETIC_COUNTS.values())
        logger.info(
            f"[Synthetic] synthetic_sentences.json not found — generating {total_needed} sentences "
            f"across {len(_SYNTHETIC_CLASSES)} classes via Claude Haiku: {_SYNTHETIC_COUNTS}"
        )
        try:
            data = _generate_synthetic_sentences()
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            saved = sum(len(v) for v in data.values())
            logger.info(f"[Synthetic] Saved {saved} sentences → {json_path}")
        except Exception as e:
            logger.warning(f"[Synthetic] Generation failed: {e} — training with 23/28 classes.")
            return empty, [], []
    else:
        with open(json_path) as f:
            data = json.load(f)
        loaded = {k: len(v) for k, v in data.items()}
        logger.info(f"[Synthetic] Loaded existing synthetic_sentences.json: {loaded}")

    # ── 2. Compute syn_hash to check feature cache ──────────────────────────────
    syn_hash = hashlib.md5(json_path.read_bytes()).hexdigest()[:12]

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("syn_hash") == syn_hash and cached.get("feature_dim") == FEATURE_DIM:
                X_syn, y_syn, gs_syn = cached["data"]
                logger.info(
                    f"[Synthetic] Feature cache hit (hash={syn_hash}) — "
                    f"loaded {len(y_syn)} pre-computed feature vectors. Skipping NLP."
                )
                return (
                    np.array(X_syn, dtype=np.float32) if X_syn else empty,
                    y_syn,
                    gs_syn,
                )
            else:
                logger.info("[Synthetic] Feature cache stale (hash or dim mismatch) — recomputing.")
        except Exception as e:
            logger.warning(f"[Synthetic] Cache load failed: {e} — recomputing.")

    # ── 3. Batch NLP processing ─────────────────────────────────────────────────
    ordered_pairs: list = []  # (goemo_label, text)
    for goemo_label, sentences in data.items():
        if goemo_label not in EMOTION_LABELS:
            logger.warning(f"[Synthetic] Unknown label '{goemo_label}' in JSON — skipping.")
            continue
        for text in sentences:
            if text:
                ordered_pairs.append((goemo_label, text))

    total_sentences = len(ordered_pairs)
    logger.info(
        f"[Synthetic] Running batched NLP on {total_sentences} sentences "
        f"(batch_size=32, 3 models)..."
    )
    t0 = time.time()

    all_texts = [t for _, t in ordered_pairs]

    vader_results = []
    for text in all_texts:
        try:
            vader_results.append({f"vader_{k}": v for k, v in _vader(vader_analyzer, text).items()})
        except Exception:
            vader_results.append({})

    bert_results = _run_batch(bert_analyzer, all_texts, batch_size=batch_size, label="Synthetic/BERT") if callable(bert_analyzer) else [{} for _ in all_texts]
    goe_results  = _run_batch(goe_analyzer,  all_texts, batch_size=batch_size, label="Synthetic/GoE")  if callable(goe_analyzer)  else [{} for _ in all_texts]

    elapsed_nlp = time.time() - t0
    rate = total_sentences / elapsed_nlp if elapsed_nlp > 0 else 0
    logger.info(
        f"[Synthetic] Batched NLP done — {total_sentences} sentences in {elapsed_nlp:.0f}s "
        f"({rate:.1f} samples/s)"
    )

    # ── 4. Build feature vectors ─────────────────────────────────────────────────
    features, labels, gs_list = [], [], []
    class_counts: dict = {}
    for (goemo_label, _text), vader_out, bert_out, goe_out in zip(
        ordered_pairs, vader_results, bert_results, goe_results
    ):
        ctx = build_synthetic_context_vector(label=goemo_label, mode="train")
        fv  = build_feature_vector(
            {"vader": vader_out, "basic_bert": bert_out, "go_emotions": goe_out},
            context_vector=ctx[:CDM_CTX_DIM],
            trajectory_prior=ctx[CDM_CTX_DIM:],
        )
        features.append(fv.flatten())
        labels.append(goemo_label)
        gs_list.append(goe_out)
        class_counts[goemo_label] = class_counts.get(goemo_label, 0) + 1

    for lbl, cnt in class_counts.items():
        logger.info(f"  [Synthetic] '{lbl}': {cnt} feature vectors built")

    # ── 5. Save feature cache ────────────────────────────────────────────────────
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({
                "syn_hash":    syn_hash,
                "feature_dim": FEATURE_DIM,
                "data":        (features, labels, gs_list),
            }, f)
        logger.info(f"[Synthetic] Feature cache saved → {cache_path} (hash={syn_hash})")
    except Exception as e:
        logger.warning(f"[Synthetic] Could not save feature cache: {e}")

    return (
        np.array(features, dtype=np.float32) if features else empty,
        labels,
        gs_list,
    )


def pretrain_context_encoder(
    model,
    X_ctx: np.ndarray,
    y_labels: list,
    classes: list,
    has_cdm: np.ndarray = None,
    n_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """
    Pre-train enc_ctx_expert + ctx_prior_head on context → emotion before
    full pipeline training. Forces the context encoder to learn emotion-predictive
    representations rather than noise, giving the Bayesian prior a useful starting point.

    Only trains enc_ctx_expert and ctx_prior_head — all other parameters frozen.
    When has_cdm is provided, only real-CDM samples are used for pretraining.
    """
    import torch
    import torch.nn.functional as F

    # Filter to real-CDM samples when mask is available
    if has_cdm is not None and has_cdm.any():
        X_ctx   = X_ctx[has_cdm]
        y_labels = [y_labels[i] for i, v in enumerate(has_cdm) if v]
        logger.info(f"  [CtxPretrain] Using {len(X_ctx)} real-CDM samples for pretraining.")
    elif has_cdm is not None and not has_cdm.any():
        logger.info("  [CtxPretrain] No real-CDM samples — skipping pretrain.")
        return model

    from shared.constants import ML_DIM
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx.get(y, 0) for y in y_labels])

    X_ctx_t = torch.tensor(X_ctx[:, ML_DIM:].astype(np.float32), device=device)
    y_t     = torch.tensor(y_idx, dtype=torch.long, device=device)

    params  = list(model.enc_ctx_expert.parameters()) + \
              list(model.ctx_prior_head.parameters())
    opt     = torch.optim.Adam(params, lr=lr)

    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        logits = model.ctx_prior_head(model.enc_ctx_expert(X_ctx_t))
        loss   = F.cross_entropy(logits, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 5 == 0:
            logger.info(f"  [CtxPretrain] epoch {epoch+1}/{n_epochs}  loss={loss.item():.4f}")

    logger.info("  [CtxPretrain] Done — enc_ctx_expert primed.")
    model.eval()
    return model
