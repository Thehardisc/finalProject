"""
meta_learner.py v2.0 — Attention-based Mixture-of-Experts Meta-Learner.

Architecture: GatingEnsembleNet
  Three NLP expert encoders (VADER, BERT, GoEmotions) are independently
  projected to a shared d-dimensional space. The 23-dim context vector
  drives a learned soft gate α ∈ ℝ³ that routes attention over the experts.
  An additive context residual ensures context contributes beyond routing.
  No hardcoded thresholds — all gating is learned end-to-end.

Public API (unchanged from v1):
  load_meta_learner()          → GatingNetworkWrapper | sklearn Pipeline | None
  build_feature_vector()       → np.ndarray [1, 62]
  predict_with_meta_learner()  → (emotion, confidence, scores, sarcasm, conflict)
  calculate_feature_impacts()  → {block: attribution}
  apply_context_correction()   → passthrough stub (superseded by native gating)
"""

import os
import pickle
import json
import numpy as np
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from shared.utils.logger import get_logger
from shared.constants import (
    EMOTION_LABELS, VADER_KEYS, BERT_LABELS,
    FEATURE_DIM, CONTEXT_DIM, ML_DIM, CTX_HMM_CONF,
)

logger = get_logger("meta_learner")

# ── Architecture hyperparameters ────────────────────────────────────────────────
D_MODEL  = 64
DROPOUT  = 0.25

# ── Legacy context-correction constants (kept for API compat — not used by gating) ─
_MAX_CTX_WEIGHT   = float(os.environ.get("CONTEXT_WEIGHT",          "0.45"))
_MISMATCH_CTX_CAP = float(os.environ.get("CONTEXT_WEIGHT_MISMATCH", "0.65"))

_POSITIVE_EMOTIONS = frozenset({
    'admiration', 'amusement', 'approval', 'caring', 'curiosity',
    'desire', 'excitement', 'gratitude', 'joy', 'love',
    'optimism', 'pride', 'relief',
})
_NEGATIVE_EMOTIONS = frozenset({
    'anger', 'annoyance', 'disapproval', 'disgust', 'disappointment',
    'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness',
})
_PREV_EMOTION_VALENCE: dict = {
    'anger': -0.8, 'annoyance': -0.6, 'disapproval': -0.7, 'disgust': -0.8,
    'disappointment': -0.6, 'embarrassment': -0.5, 'fear': -0.7, 'grief': -0.9,
    'nervousness': -0.5, 'remorse': -0.7, 'sadness': -0.7,
    'admiration': 0.8, 'amusement': 0.6, 'approval': 0.6, 'caring': 0.7,
    'curiosity': 0.3, 'desire': 0.5, 'excitement': 0.8, 'gratitude': 0.8,
    'joy': 0.9, 'love': 0.9, 'optimism': 0.7, 'pride': 0.7,
    'relief': 0.6, 'neutral': 0.0, 'confusion': -0.1,
    'realization': 0.1, 'surprise': 0.2,
}

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl")
DEFAULT_META_PATH  = DEFAULT_MODEL_PATH.replace(".pkl", "_meta.json")


# ── Neural architecture ─────────────────────────────────────────────────────────

class GatingEnsembleNet(nn.Module):
    """
    Bayesian Context-First architecture for 28-class emotion classification.

    Forward pass:
        ctx_weight = sigmoid(ctx_alpha * hmm_conf - ctx_bias)   per-sample [0,1]
        ctx_logits = CtxPriorHead(enc_ctx_expert(x_c))          P(emotion | history)
        nlp_logits = NLPHead(Σ_k α_k · enc_k(x_k))             P(emotion | text)
        logits     = ctx_weight * ctx_logits + (1-ctx_weight) * nlp_logits / nlp_temp

    When hmm_conf → 1.0 (HMM certain):  ctx_weight → high  → context dominates
    When hmm_conf → 0.0 (cold start):   ctx_weight → ~0.12 → NLP dominates

    ctx_alpha, ctx_bias, nlp_temp are learnable — the network adapts the
    confidence curve to the actual quality of the HMM signal in the training data.
    """

    def __init__(
        self,
        n_classes: int = len(EMOTION_LABELS),
        d: int = D_MODEL,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        # NLP modality encoders
        self.enc_vader = self._make_encoder(4,  d)
        self.enc_bert  = self._make_encoder(7,  d)
        self.enc_goe   = self._make_encoder(28, d)

        # Context routing encoder: heavy dropout resists synthetic-data leakage.
        # Used to route the NLP gate and provide the context residual.
        self.enc_context = nn.Sequential(
            nn.Linear(CONTEXT_DIM, d * 2),
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(dropout + 0.15),
            nn.Linear(d * 2, d),
            nn.GELU(),
        )

        # Context expert encoder: lighter, learns emotion-predictive representation.
        self.enc_ctx_expert = self._make_encoder(CONTEXT_DIM, d)

        # Context prior head: enc_ctx_expert → emotion prior logits
        self.ctx_prior_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, n_classes),
        )

        # NLP gate: routes attention over 3 NLP experts [vader, bert, goe]
        self.gate_nlp = nn.Sequential(
            nn.Linear(d, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

        # NLP likelihood head
        self.nlp_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, n_classes),
        )

        # Context residual: contributes to NLP repr for smoother convergence
        self.ctx_residual = nn.Linear(d, d)

        # Bayesian combination parameters (all learnable)
        # ctx_weight = sigmoid(ctx_alpha * hmm_conf - ctx_bias)
        # defaults: hmm_conf=0 → sigmoid(-2)≈0.12, hmm_conf=1 → sigmoid(1)≈0.73
        self.ctx_alpha = nn.Parameter(torch.tensor(3.0))
        self.ctx_bias  = nn.Parameter(torch.tensor(2.0))
        self.nlp_temp  = nn.Parameter(torch.tensor(1.5))

    @staticmethod
    def _make_encoder(in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, FEATURE_DIM]
               x[:,  0:4]       — VADER
               x[:,  4:11]      — BERT
               x[:, 11:ML_DIM]  — GoEmotions
               x[:, ML_DIM:]    — Context (CONTEXT_DIM dims)

        Returns:
            logits:       [B, n_classes]
            gate_weights: [B, 4]  [vader, bert, goe, context] — sum to 1 per sample
        """
        x_v = x[:, 0:4]
        x_b = x[:, 4:11]
        x_g = x[:, 11:ML_DIM]
        x_c = x[:, ML_DIM:FEATURE_DIM]

        # HMM confidence — drives context vs NLP balance
        hmm_conf   = x_c[:, CTX_HMM_CONF].unsqueeze(1).clamp(0.0, 1.0)  # [B, 1]
        ctx_weight = torch.sigmoid(
            self.ctx_alpha.clamp(0.5, 6.0) * hmm_conf
            - self.ctx_bias.clamp(0.5, 4.0)
        )  # [B, 1]

        # Context path
        c_h              = self.enc_context(x_c)
        e_c              = self.enc_ctx_expert(x_c)
        ctx_prior_logits = self.ctx_prior_head(e_c)           # [B, n_classes]

        # NLP path
        e_v       = self.enc_vader(x_v)
        e_b       = self.enc_bert(x_b)
        e_g       = self.enc_goe(x_g)
        alpha_nlp = F.softmax(self.gate_nlp(c_h), dim=-1)    # [B, 3]
        nlp_mix   = (alpha_nlp.unsqueeze(-1) * torch.stack([e_v, e_b, e_g], dim=1)).sum(dim=1)
        nlp_repr  = nlp_mix + 0.3 * self.ctx_residual(c_h)
        nlp_logits = self.nlp_head(nlp_repr) / self.nlp_temp.clamp(0.5, 3.0)  # [B, n_classes]

        # Bayesian logit averaging — equivalent to geometric mean of distributions
        logits = ctx_weight * ctx_prior_logits + (1.0 - ctx_weight) * nlp_logits

        # Gate weights for logging: NLP experts share (1-ctx_weight), context gets ctx_weight
        nlp_w     = 1.0 - ctx_weight                          # [B, 1]
        alpha_out = torch.cat([
            alpha_nlp * nlp_w,  # [B, 3]: vader, bert, goe weighted by NLP fraction
            ctx_weight,         # [B, 1]: context weight
        ], dim=1)               # [B, 4]

        return logits, alpha_out


class GatingNetworkWrapper:
    """
    sklearn-compatible wrapper around GatingEnsembleNet.

    Exposes .predict(X), .predict_proba(X), .classes_ — identical interface
    to the previous sklearn MLPClassifier Pipeline, so every existing caller
    (calculate_feature_impacts, load_meta_learner probe, main.py) works
    without modification.

    An additional .get_gate_weights(X) method returns α [n, 3] for the
    Pipeline Inspector / interpretability logging.
    """

    def __init__(
        self,
        model: GatingEnsembleNet,
        classes: List[str],
        scaler: StandardScaler,
    ):
        self.model_   = model
        self.classes_ = np.array(classes)
        self.scaler_  = scaler
        self._device  = next(model.parameters()).device

    # ── sklearn interface ──────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler_.transform(X.reshape(-1, FEATURE_DIM))
        t   = torch.tensor(X_s, dtype=torch.float32, device=self._device)
        self.model_.eval()
        with torch.no_grad():
            logits, _ = self.model_(t)
            # Apply calibration temperature if stored
            T = getattr(self, '_temperature', 1.0)
            proba = F.softmax(logits / T, dim=-1).cpu().numpy()
        return proba  # [n, n_classes]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    # ── Extended API ───────────────────────────────────────────────────────────

    def get_gate_weights(self, X: np.ndarray) -> np.ndarray:
        """Return α routing weights [n, 4] — order: [vader, bert, goe, context]."""
        X_s = self.scaler_.transform(X.reshape(-1, FEATURE_DIM))
        t   = torch.tensor(X_s, dtype=torch.float32, device=self._device)
        self.model_.eval()
        with torch.no_grad():
            _, alpha = self.model_(t)
        return alpha.cpu().numpy()


# ── Public API ──────────────────────────────────────────────────────────────────

def load_meta_learner(model_path: str = DEFAULT_MODEL_PATH) -> Optional[object]:
    """
    Load the trained model from disk.

    Accepts both:
      - GatingNetworkWrapper  (v2 — preferred)
      - sklearn Pipeline      (v1 — backward compat while transitioning)

    Returns None on any failure — caller falls back to rule-based mode.
    """
    try:
        if not os.path.exists(model_path):
            logger.warning(f"No model file at '{model_path}'. Fallback mode.")
            return None

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        if not hasattr(model, 'predict_proba'):
            logger.warning("Loaded object has no predict_proba. Fallback mode.")
            return None

        # Dimension probe — catches stale pkl files after FEATURE_DIM changes
        try:
            dummy = np.zeros((1, FEATURE_DIM))
            model.predict(dummy)
        except Exception as e:
            logger.warning(f"Model incompatible with {FEATURE_DIM}-dim probe: {e}. Deleting stale pkl.")
            try:
                os.remove(model_path)
            except OSError:
                pass
            return None

        _log_metadata()

        # Load calibration temperature from meta.json if available
        if isinstance(model, GatingNetworkWrapper):
            try:
                meta_path = model_path.replace(".pkl", "_meta.json") if isinstance(model_path, str) else str(model_path).replace(".pkl", "_meta.json")
                with open(meta_path) as f:
                    meta = json.load(f)
                T = float(meta.get("calibration_temperature", 1.0))
                model._temperature = T
                logger.info(f"   Calibration T   : {T:.4f}")
            except Exception:
                model._temperature = 1.0
            logger.info("Meta-learner v2 (GatingEnsembleNet + Bayesian ctx-first) loaded.")
        else:
            logger.info("Meta-learner v1 (sklearn Pipeline) loaded — retrain to upgrade.")

        return model

    except Exception as e:
        logger.warning(f"Failed to load model: {e}. Fallback mode.")
        return None


def build_feature_vector(
    model_outputs: dict,
    context_vector: list = None,
) -> np.ndarray:
    """
    Assemble a fixed 62-dim float32 feature vector.

    Layout:
      [0:4]   VADER (4)
      [4:11]  BERT Ekman (7)
      [11:39] GoEmotions (28)
      [39:62] Context Engine (23)
    """
    vader_scores      = model_outputs.get("vader",       {})
    bert_scores       = model_outputs.get("basic_bert",  {})
    goemotions_scores = model_outputs.get("go_emotions", {})

    vec = []
    for k in VADER_KEYS:
        vec.append(float(vader_scores.get(k, 0.0)))
    for k in BERT_LABELS:
        vec.append(float(bert_scores.get(k, 0.0)))
    for k in EMOTION_LABELS:
        vec.append(float(goemotions_scores.get(k, 0.0)))

    ctx = (
        context_vector
        if (context_vector and len(context_vector) == CONTEXT_DIM)
        else [0.0] * CONTEXT_DIM
    )
    vec.extend(ctx)

    return np.array(vec, dtype=np.float32).reshape(1, -1)


def predict_with_meta_learner(
    model,
    feature_vector: np.ndarray,
) -> Tuple[str, float, dict, float, Optional[str], Optional[list]]:
    """
    Run inference with the meta-learner.

    Args:
        model:          GatingNetworkWrapper or sklearn Pipeline
        feature_vector: [1, 62] from build_feature_vector()

    Returns:
        (dominant_emotion, confidence, all_scores, sarcasm_score, conflict_desc, gate_alpha)
        gate_alpha is [vader_w, bert_w, goe_w] for v2, None for v1/fallback.
        On any error returns ("neutral", 0.0, {}, 0.0, None, None).
    """
    if model is None:
        goe = feature_vector[0, 11:ML_DIM]
        if goe.max() > 0:
            idx    = int(goe.argmax())
            scores = {e: float(goe[i]) for i, e in enumerate(EMOTION_LABELS)}
            return EMOTION_LABELS[idx], float(goe[idx]), scores, 0.0, None, None
        return "neutral", 1.0, {"neutral": 1.0}, 0.0, None, None

    try:
        pred_label = model.predict(feature_vector)[0]
        proba      = model.predict_proba(feature_vector)[0]
        classes    = model.classes_

        all_scores   = {str(k): float(v) for k, v in zip(classes, proba)}
        label_idx    = list(classes).index(pred_label)
        confidence   = float(proba[label_idx])

        gate_alpha = None
        # ── DIAG-3: gate weights — promoted to WARNING so they always appear ──
        if isinstance(model, GatingNetworkWrapper):
            alpha      = model.get_gate_weights(feature_vector)[0]
            gate_alpha = [round(float(a), 4) for a in alpha]
            ctx_vec    = feature_vector[0, ML_DIM:]  # context block [ML_DIM:FEATURE_DIM]
            ctx_L2     = float(np.linalg.norm(ctx_vec))
            ctx_mean   = float(np.mean(ctx_vec))
            ctx_max    = float(np.max(np.abs(ctx_vec)))
            logger.debug(
                f"[DIAG-3] Gate α — vader:{alpha[0]:.3f}  bert:{alpha[1]:.3f}  "
                f"goe:{alpha[2]:.3f}  ctx:{alpha[3]:.3f}  (sum={sum(alpha):.3f})  "
                f"ctx_L2={ctx_L2:.4f}  ctx_mean={ctx_mean:.4f}  ctx_max={ctx_max:.4f}"
            )
        else:
            logger.debug(
                f"[DIAG-3] Model type={type(model).__name__} — NOT GatingNetworkWrapper; "
                f"no gate α available (stale pkl?)"
            )

        sarcasm_score, conflict_desc = detect_emotional_conflicts(feature_vector)
        return pred_label, confidence, all_scores, sarcasm_score, conflict_desc, gate_alpha

    except Exception as e:
        logger.warning(f"Predict error: {e}. Returning neutral.")
        return "neutral", 0.0, {}, 0.0, None, None


def detect_emotional_conflicts(vec: np.ndarray) -> Tuple[float, Optional[str]]:
    """
    Heuristic sarcasm / cognitive-dissonance detector.

    Uses index-based access to the flat feature vector — indices mirror the
    VADER_KEYS / BERT_LABELS / EMOTION_LABELS ordering in build_feature_vector.
    Returns (sarcasm_score [0.0–1.0], conflict_description | None).
    """
    try:
        v = vec.flatten()

        # VADER block [0:4]: neg=0, neu=1, pos=2, compound=3
        v_pos = float(v[2])
        v_cmp = float(v[3])

        # BERT Ekman [4:11]: anger=4, disgust=5, fear=6, joy=7, neutral=8, sadness=9, surprise=10
        bert_joy     = float(v[7])
        bert_neutral = float(v[8])

        # GoEmotions [11:39] — EMOTION_LABELS ordering
        # annoyance=idx3, disapproval=idx10, disgust=idx11
        emo_annoyance   = float(v[11 + 3])
        emo_disapproval = float(v[11 + 10])
        emo_disgust     = float(v[11 + 11])

        neg_emo     = max(emo_annoyance, emo_disapproval, emo_disgust)
        pos_text    = (v_pos + bert_joy) / 2.0
        sarcasm_score = 0.0
        conflict_desc = None

        if pos_text > 0.6 and neg_emo > 0.4:
            sarcasm_score = min(pos_text, neg_emo) * 1.2
            conflict_desc = (
                "Cognitive Dissonance: high-fidelity positive text "
                "paired with dismissive visual cues."
            )
        elif v_cmp > 0.8 and neg_emo > 0.2:
            sarcasm_score = 0.5 + neg_emo
            conflict_desc = (
                "Sarcasm detected: semantic praise contradicts visual frustration."
            )
        elif bert_neutral > 0.7 and (emo_annoyance > 0.1 or emo_disapproval > 0.1):
            sarcasm_score = 0.4
            conflict_desc = (
                "Passive-aggression suspected: formal 'Neutral' text "
                "with underlying emoji tension."
            )

        return min(sarcasm_score, 1.0), conflict_desc

    except Exception:
        return 0.0, None


def calculate_feature_impacts(
    model,
    feature_vector: np.ndarray,
    predicted_emotion: str,
) -> dict:
    """
    Block-level feature attribution via leave-one-out sensitivity.

    Measures Δproba when each modality block is zeroed.  Four forward passes;
    negligible latency.  Model-agnostic — works for both v1 and v2.
    """
    try:
        classes = list(model.classes_)
        if predicted_emotion not in classes:
            return {}
        class_idx  = classes.index(predicted_emotion)
        base_proba = float(model.predict_proba(feature_vector)[0][class_idx])

        blocks = {
            "VADER":      slice(0,      4),
            "BERT":       slice(4,      11),
            "GoEmotions": slice(11,     ML_DIM),
            "Context":    slice(ML_DIM, FEATURE_DIM),
        }
        impacts = {}
        for name, sl in blocks.items():
            x_masked          = feature_vector.copy()
            x_masked[0, sl]   = 0.0
            masked_proba      = float(model.predict_proba(x_masked)[0][class_idx])
            impacts[name]     = abs(base_proba - masked_proba)

        total = sum(impacts.values())
        if total > 0:
            return {k: round(v / total, 4) for k, v in impacts.items()}
        return {k: 0.25 for k in impacts}

    except Exception as e:
        logger.warning(f"Failed to calculate impacts: {e}")
        return {}


def apply_context_correction(
    scores: dict,
    context_vector: list,
    prev_emotion: str = "neutral",
) -> Tuple[dict, float]:
    """
    Passthrough stub — superseded by native context gating in GatingEnsembleNet.

    The v2 network learns valence-based routing end-to-end; post-hoc additive
    blending with hard thresholds is no longer needed.  This stub is kept so
    main.py compiles without modification during Phase 2 migration.
    """
    return scores, 0.0


# ── Private helpers ─────────────────────────────────────────────────────────────

def _log_metadata() -> None:
    try:
        if os.path.exists(DEFAULT_META_PATH):
            with open(DEFAULT_META_PATH) as f:
                meta = json.load(f)
            logger.info(f"   Trained at      : {meta.get('trained_at',          'unknown')}")
            logger.info(f"   Training samples: {meta.get('training_samples',    '?')}")
            logger.info(f"   Val accuracy    : {meta.get('validation_accuracy', '?')}")
            logger.info(f"   Test accuracy   : {meta.get('test_accuracy',       '?')}")
    except Exception:
        pass
