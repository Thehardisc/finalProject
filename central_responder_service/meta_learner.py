import io
import os
import pickle
import json
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


from shared.utils.logger import get_logger
from shared.constants import (
    EMOTION_LABELS,
    VADER_KEYS,
    BERT_LABELS,
    FEATURE_DIM,
    CONTEXT_DIM,
    CDM_CTX_DIM,
    PRIOR_DIM,
    ML_DIM,
    CTX_CDM_PROBS,
    CTX_RESIDENCY,
    CTX_TRANSITION,
    CTX_COHERENCE,
    CTX_ENTROPY,
    CTX_RESONANCE,
    CTX_HMM_CONF,
    CTX_MSG_LENGTH,
    CTX_LATENCY_MS,
    CTX_HMM_EMISSION,
    CTX_HMM_ENTROPY,
    CTX_HMM_NEXT3,
    CTX_INTENT_STAB,
    CTX_CURR_VALENCE,
)

logger = get_logger("meta_learner")

D_MODEL = 64
DROPOUT = 0.25

# Cap on the context expert's share of the final logits. The historical 0.50 cap
# let a leak-trained context head dominate live traffic; 0.20 is a mitigation until
# a model trained on honest context features (real verified samples) justifies more.
CTX_WEIGHT_CAP = float(os.environ.get("CTX_WEIGHT_CAP", "0.20"))

# Decision boundary for the combined sarcasm evidence: scores above it fire the
# inversion (and mark the verdict sarcasm-backed downstream). The no-neg-cue cap
# below pins capped scores AT this boundary so they can never invert.
SARCASM_INVERSION_THRESHOLD = float(os.environ.get("SARCASM_INVERSION_THRESHOLD", "0.40"))

# CDM/HMM state features are masked out of the model input (train AND inference).
# In every bootstrap dataset these were synthesized FROM the gold label
# (trainer/data/empathetic.py:_GOEMO_TO_CDM_STATE → ctx one-hot), so a model can
# only ever learn them as label leakage. Until training data with honestly-derived
# states exists (verified live samples), the model must not see these slots.
# Kept: valence/speaker/text-derived scalars, which are computed the same way in
# training and live. The context_engine still publishes the full vector — the
# Pipeline Inspector and snapshots are unaffected; only the model input is masked.
_CDM_KEEP = np.ones(CDM_CTX_DIM, dtype=np.float32)
for _masked in (CTX_CDM_PROBS, CTX_RESIDENCY, CTX_TRANSITION, CTX_COHERENCE,
                CTX_ENTROPY, CTX_RESONANCE, CTX_HMM_CONF, CTX_HMM_ENTROPY,
                CTX_HMM_EMISSION, CTX_HMM_NEXT3, CTX_INTENT_STAB):
    _CDM_KEEP[_masked] = 0.0


def apply_context_mask(X):
    """Zero label-leaked CDM/HMM state columns of feature array(s) in place.

    Accepts [FEATURE_DIM] or [N, FEATURE_DIM]. Idempotent — used both by
    build_feature_vector (fresh vectors) and trainer/cycle.py (cached vectors
    built before the mask existed).
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X[ML_DIM:ML_DIM + CDM_CTX_DIM] *= _CDM_KEEP
    else:
        X[:, ML_DIM:ML_DIM + CDM_CTX_DIM] *= _CDM_KEEP
    return X

_GOE_TO_EKMAN = {
    "admiration": "joy",    "amusement": "joy",       "anger": "anger",
    "annoyance":  "anger",  "approval":  "joy",       "caring": "joy",
    "confusion":  "surprise","curiosity": "surprise",  "desire": "joy",
    "disappointment": "sadness", "disapproval": "disgust", "disgust": "disgust",
    "embarrassment": "fear","excitement": "joy",       "fear": "fear",
    "gratitude":  "joy",    "grief": "sadness",       "joy": "joy",
    "love":       "joy",    "nervousness": "fear",    "optimism": "joy",
    "pride":      "joy",    "realization": "surprise", "relief": "joy",
    "remorse":    "sadness","sadness": "sadness",     "surprise": "surprise",
    "neutral":    "neutral",
}

_EMOTION_VAD = {
    'admiration':     ( 0.72,  0.28,  0.52), 'amusement':      ( 0.76,  0.54,  0.42),
    'anger':          (-0.64,  0.82,  0.62), 'annoyance':      (-0.48,  0.54,  0.42),
    'approval':       ( 0.60,  0.28,  0.44), 'caring':         ( 0.74,  0.28,  0.36),
    'confusion':      (-0.14,  0.36, -0.16), 'curiosity':      ( 0.38,  0.52,  0.30),
    'desire':         ( 0.56,  0.70,  0.38), 'disappointment': (-0.62,  0.30, -0.32),
    'disapproval':    (-0.52,  0.44,  0.36), 'disgust':        (-0.72,  0.54,  0.44),
    'embarrassment':  (-0.42,  0.48, -0.40), 'excitement':     ( 0.72,  0.90,  0.52),
    'fear':           (-0.72,  0.78, -0.64), 'gratitude':      ( 0.84,  0.28,  0.30),
    'grief':          (-0.86, -0.20, -0.60), 'joy':            ( 0.90,  0.72,  0.60),
    'love':           ( 0.88,  0.52,  0.44), 'nervousness':    (-0.44,  0.72, -0.50),
    'optimism':       ( 0.76,  0.48,  0.52), 'pride':          ( 0.76,  0.58,  0.72),
    'realization':    ( 0.20,  0.46,  0.16), 'relief':         ( 0.68, -0.30,  0.32),
    'remorse':        (-0.60, -0.24, -0.44), 'sadness':        (-0.72, -0.40, -0.54),
    'surprise':       ( 0.26,  0.72,  0.16), 'neutral':        ( 0.00,  0.00,  0.00),
}
_VAD_ARRAY = np.array([_EMOTION_VAD[k] for k in EMOTION_LABELS], dtype=np.float32)

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl")
DEFAULT_META_PATH  = DEFAULT_MODEL_PATH.replace(".pkl", "_meta.json")


class _CPUUnpickler(pickle.Unpickler):
    # Deserializes pkl files saved on MPS/CUDA into CPU tensors.
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _compute_vad_from_goe(goe_vec):
    total = goe_vec.sum()
    if total < 1e-6:
        return 0.0, 0.0, 0.0
    vad = (goe_vec / total) @ _VAD_ARRAY
    return tuple(float(np.clip(v, -1.0, 1.0)) for v in vad)


def _compute_inertia(goe_vec, prior_vec):
    norm_g, norm_p = float(np.linalg.norm(goe_vec)), float(np.linalg.norm(prior_vec))
    if norm_g < 1e-6 or norm_p < 1e-6:
        return 0.0
    return float(np.clip(np.dot(goe_vec, prior_vec) / (norm_g * norm_p), -1.0, 1.0))


def _compute_novelty(goe_vec, prior_vec):
    norm_g = float(np.linalg.norm(goe_vec))
    if norm_g < 1e-6:
        return 0.0
    norm_p = float(np.linalg.norm(prior_vec))
    if norm_p < 1e-6:
        p = goe_vec / norm_g
        return float(np.clip(-np.sum(p * np.log(p + 1e-9)) / np.log(len(EMOTION_LABELS)), 0.0, 1.0))
    return float(np.clip(1.0 - np.dot(goe_vec, prior_vec) / (norm_g * norm_p), 0.0, 1.0))


class GatingEnsembleNet(nn.Module):
    def __init__(self, n_classes=len(EMOTION_LABELS), d=D_MODEL, dropout=DROPOUT):
        super().__init__()
        self.enc_vader      = self._make_encoder(4,  d)
        self.enc_bert       = self._make_encoder(7,  d)
        self.enc_goe        = self._make_encoder(28, d)
        self.enc_vad        = self._make_encoder(3,  d)
        self.enc_context    = nn.Sequential(
            nn.Linear(CONTEXT_DIM, d * 2), nn.LayerNorm(d * 2), nn.GELU(),
            nn.Dropout(dropout + 0.15),
            nn.Linear(d * 2, d), nn.GELU(),
        )
        self.enc_ctx_expert = self._make_encoder(CONTEXT_DIM, d)
        self.ctx_prior_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))
        self.gate_nlp       = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 4))
        self.nlp_head       = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d * 2, n_classes),
        )
        self.ctx_residual = nn.Linear(d, d)
        self.ctx_alpha = nn.Parameter(torch.tensor(2.0))
        self.ctx_bias  = nn.Parameter(torch.tensor(2.5))
        self.nlp_temp  = nn.Parameter(torch.tensor(1.5))

    @staticmethod
    def _make_encoder(in_dim, out_dim):
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x):
        x_v, x_b, x_g = x[:, 0:4], x[:, 4:11], x[:, 11:39]
        x_vad, x_c    = x[:, 39:42], x[:, 42:]

        hmm_conf      = x_c[:, CTX_HMM_CONF].unsqueeze(1).clamp(0.0, 1.0)
        ctx_weight    = torch.sigmoid(
            self.ctx_alpha.clamp(0.5, 4.0) * hmm_conf - self.ctx_bias.clamp(1.0, 4.0)
        ).clamp(max=CTX_WEIGHT_CAP)
        ctx_available = (x_c.abs().sum(dim=1, keepdim=True) > 0.01).float()

        c_h              = self.enc_context(x_c)
        ctx_prior_logits = self.ctx_prior_head(self.enc_ctx_expert(x_c))

        e_v, e_b, e_g, e_vad = (
            self.enc_vader(x_v), self.enc_bert(x_b), self.enc_goe(x_g), self.enc_vad(x_vad)
        )
        ctx_gate_raw = F.softmax(self.gate_nlp(c_h * ctx_available), dim=-1)
        goe_capped   = ctx_gate_raw[:, 2:3].clamp(max=0.50)
        others       = torch.cat([ctx_gate_raw[:, :2], ctx_gate_raw[:, 3:]], dim=1)
        others       = others / others.sum(dim=1, keepdim=True).clamp(min=1e-8) * (1.0 - goe_capped)
        ctx_gate     = torch.cat([others[:, :2], goe_capped, others[:, 2:]], dim=1)
        alpha_nlp    = ctx_gate * ctx_available + (1.0 - ctx_available) / 4.0
        nlp_mix      = (alpha_nlp.unsqueeze(-1) * torch.stack([e_v, e_b, e_g, e_vad], dim=1)).sum(dim=1)
        nlp_logits   = (
            self.nlp_head(nlp_mix + 0.3 * ctx_available * self.ctx_residual(c_h))
            / self.nlp_temp.clamp(0.5, 3.0)
        )
        logits    = ctx_weight * ctx_prior_logits + (1.0 - ctx_weight) * nlp_logits
        alpha_out = torch.cat([alpha_nlp * (1.0 - ctx_weight), ctx_weight], dim=1)
        return logits, alpha_out


class GatingNetworkWrapper:
    def __init__(self, model, classes, scaler):
        self.model_   = model
        self.classes_ = np.array(classes)
        self.scaler_  = scaler
        self._device  = next(model.parameters()).device

    def predict_proba(self, X):
        X_s = self.scaler_.transform(X.reshape(-1, FEATURE_DIM))
        t   = torch.tensor(X_s, dtype=torch.float32, device=self._device)
        self.model_.eval()
        with torch.no_grad():
            logits, _ = self.model_(t)
            return F.softmax(logits / getattr(self, '_temperature', 1.0), dim=-1).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def get_gate_weights(self, X):
        X_s = self.scaler_.transform(X.reshape(-1, FEATURE_DIM))
        t   = torch.tensor(X_s, dtype=torch.float32, device=self._device)
        self.model_.eval()
        with torch.no_grad():
            _, alpha = self.model_(t)
        return alpha.cpu().numpy()


def load_meta_learner(model_path=DEFAULT_MODEL_PATH):
    try:
        if not os.path.exists(model_path):
            logger.warning(f"No model file at '{model_path}'. Fallback mode.")
            return None
        with open(model_path, 'rb') as f:
            model = _CPUUnpickler(f).load()
        if not hasattr(model, 'predict_proba'):
            logger.warning("Loaded object has no predict_proba. Fallback mode.")
            return None
        if isinstance(model, GatingNetworkWrapper):
            model.model_.cpu()
            model._device = torch.device("cpu")
        try:
            model.predict(np.zeros((1, FEATURE_DIM)))
        except Exception as e:
            logger.warning(f"Model incompatible with {FEATURE_DIM}-dim probe: {e}. Deleting stale pkl.")
            try:
                os.remove(model_path)
            except OSError:
                pass
            return None
        _log_metadata()
        if isinstance(model, GatingNetworkWrapper):
            try:
                with open(str(model_path).replace(".pkl", "_meta.json")) as f:
                    meta = json.load(f)
                model._temperature = float(meta.get("calibration_temperature", 1.0))
                logger.info(f"   Calibration T: {model._temperature:.4f}")
            except Exception:
                model._temperature = 1.0
            logger.info("Meta-learner v2 (GatingEnsembleNet) loaded.")
        else:
            logger.info("Meta-learner v1 (sklearn Pipeline) loaded — retrain to upgrade.")
        return model
    except Exception as e:
        logger.warning(f"Failed to load model: {e}. Fallback mode.")
        return None


def build_feature_vector(
    model_outputs, context_vector=None, trajectory_prior=None,
    sarcasm_score=0.0, vad_scores=None, dynamics_scores=None, appraisal_scores=None,
):
    vader_scores      = model_outputs.get("vader",       {})
    bert_scores       = model_outputs.get("basic_bert",  {})
    goemotions_scores = model_outputs.get("go_emotions", {})

    def _sf(v, d=0.0):
        try:
            f = float(v)
            return f if np.isfinite(f) else d
        except Exception:
            return d

    goe_vec  = np.array([_sf(goemotions_scores.get(k, 0.0)) for k in EMOTION_LABELS], dtype=np.float32)
    compound = _sf(vader_scores.get("vader_compound", 0.0))

    vec  = [_sf(vader_scores.get(k, 0.0))   for k in VADER_KEYS]
    vec += [_sf(bert_scores.get(k, 0.0))    for k in BERT_LABELS]
    vec += goe_vec.tolist()

    if vad_scores is not None:
        valence   = float(np.clip(vad_scores.get("valence",   0.0), -1.0, 1.0))
        arousal   = float(np.clip(vad_scores.get("arousal",   0.0), -1.0, 1.0))
        dominance = float(np.clip(vad_scores.get("dominance", 0.0), -1.0, 1.0))
    else:
        valence, arousal, dominance = _compute_vad_from_goe(goe_vec)
    vec += [valence, arousal, dominance]

    cdm = list(context_vector if (context_vector is not None and len(context_vector) == CDM_CTX_DIM) else [0.0] * CDM_CTX_DIM)
    cdm[CTX_MSG_LENGTH]   = min(float(cdm[CTX_MSG_LENGTH])  / 500.0,  3.0)
    cdm[CTX_LATENCY_MS]   = min(float(cdm[CTX_LATENCY_MS])  / 2000.0, 3.0)
    cdm[CTX_HMM_EMISSION] = float(np.clip(cdm[CTX_HMM_EMISSION], 0.0, 1.0))
    # Drop the label-leaked CDM/HMM state slots — see _CDM_KEEP above.
    cdm = [float(v) * float(k) for v, k in zip(cdm, _CDM_KEEP)]

    prior_list = trajectory_prior if (trajectory_prior is not None and len(trajectory_prior) == PRIOR_DIM) else [0.0] * PRIOR_DIM
    prior_vec  = np.array(prior_list, dtype=np.float32)
    sarc       = float(np.clip(sarcasm_score, 0.0, 1.0))

    if dynamics_scores is not None:
        inertia   = float(np.clip(dynamics_scores.get("inertia",   0.0), -1.0, 1.0))
        contagion = float(np.clip(dynamics_scores.get("contagion", 0.0), -1.0, 1.0))
    else:
        inertia   = _compute_inertia(goe_vec, prior_vec)
        contagion = float(np.clip(compound * float(cdm[CTX_CURR_VALENCE]), -1.0, 1.0))

    if appraisal_scores is not None:
        novelty         = float(np.clip(appraisal_scores.get("novelty",         0.0),  0.0, 1.0))
        goal_congruence = float(np.clip(appraisal_scores.get("goal_congruence", 0.0), -1.0, 1.0))
        coping          = float(np.clip(appraisal_scores.get("coping",          0.5),  0.0, 1.0))
    else:
        novelty         = _compute_novelty(goe_vec, prior_vec)
        goal_congruence = float(np.clip(valence, -1.0, 1.0))
        coping          = float(np.clip((compound + 1.0) / 2.0, 0.0, 1.0))

    vec += cdm + prior_list + [sarc, inertia, contagion, novelty, goal_congruence, coping]
    return np.array(vec, dtype=np.float32).reshape(1, -1)


# Appends a decision-path event when a trace list is supplied (Pipeline Inspector).
def _t(trace, step, status, detail, label=None, confidence=None):
    if trace is not None:
        trace.append({
            "step": step, "status": status, "detail": detail,
            "label": label, "confidence": round(float(confidence), 4) if confidence is not None else None,
        })


def predict_with_meta_learner(model, feature_vector, trace=None):
    if model is None:
        goe = feature_vector[0, 11:39]
        if goe.max() > 0:
            idx    = int(goe.argmax())
            scores = {e: float(goe[i]) for i, e in enumerate(EMOTION_LABELS)}
            _t(trace, "Rule-based fallback", "applied",
               "No meta-learner loaded — raw GoEmotions argmax used",
               EMOTION_LABELS[idx], goe[idx])
            return EMOTION_LABELS[idx], float(goe[idx]), scores, 0.0, None, None
        # GoE absent (service down / not yet arrived) — fall through to BERT before
        # giving up; every BERT_LABEL is also a valid GoEmotions label.
        bert = feature_vector[0, 4:11]
        if bert.max() > 0:
            idx    = int(bert.argmax())
            label  = BERT_LABELS[idx]
            scores = {e: float(bert[i]) for i, e in enumerate(BERT_LABELS)}
            _t(trace, "Rule-based fallback", "applied",
               "No meta-learner and no GoE signal — raw BERT argmax used",
               label, bert[idx])
            return label, float(bert[idx]), scores, 0.0, None, None
        _t(trace, "Rule-based fallback", "applied",
           "No meta-learner and no model signal at all — neutral guess (low confidence)",
           "neutral", 0.5)
        return "neutral", 0.5, {"neutral": 0.5}, 0.0, None, None

    try:
        # Slot [110] (sarcasm) is all-zero in every bootstrap dataset, so the
        # network's weights on that input are untrained init noise — keep the
        # model input at the zero it was trained with. Sarcasm still acts through
        # the inversion/override layer below, which reads the real score.
        # TODO: drop this once training data carries real sarcasm scores.
        model_input = feature_vector.copy()
        model_input[0, ML_DIM + CDM_CTX_DIM + PRIOR_DIM] = 0.0
        pred_label = model.predict(model_input)[0]
        proba      = model.predict_proba(model_input)[0]
        classes    = model.classes_
        all_scores = {str(k): float(v) for k, v in zip(classes, proba)}
        confidence = float(proba[list(classes).index(pred_label)])

        goe_raw  = feature_vector[0, 11:39]
        goe_max  = float(goe_raw.max())
        goe_top  = EMOTION_LABELS[int(goe_raw.argmax())]
        bert_raw = feature_vector[0, 4:11]
        bert_max = float(bert_raw.max())
        bert_top = BERT_LABELS[int(bert_raw.argmax())]
        _compound     = float(feature_vector[0, 3])
        _sarcasm_slot = float(feature_vector[0, 110])

        def _top3(sc):
            return " · ".join(f"{k} {v:.2f}" for k, v in
                              sorted(((k, v) for k, v in sc.items() if k != "ekman_group"),
                                     key=lambda x: -x[1])[:3])

        _t(trace, "Model inputs", "info",
           f"VADER compound {_compound:+.2f} · BERT top {bert_top} {bert_max:.2f} · "
           f"GoE top {goe_top} {goe_max:.2f} · sarcasm-classifier {_sarcasm_slot:.2f}")

        gate_alpha = None
        if isinstance(model, GatingNetworkWrapper):
            alpha      = model.get_gate_weights(feature_vector)[0]
            gate_alpha = [round(float(a), 4) for a in alpha]
            vad_str    = f"  vad:{alpha[3]:.3f}" if len(alpha) > 4 else ""
            logger.debug(f"[Gate] vader:{alpha[0]:.3f} bert:{alpha[1]:.3f} goe:{alpha[2]:.3f}{vad_str} ctx:{alpha[-1]:.3f}")
            _names = ["VADER", "BERT", "GoE", "VAD", "Context"][:len(gate_alpha)]
            _dom_gate = _names[int(max(range(len(gate_alpha)), key=lambda i: gate_alpha[i]))]
            _t(trace, "Ensemble gates", "info",
               " · ".join(f"{n} {a:.2f}" for n, a in zip(_names, gate_alpha))
               + f" — {_dom_gate} weighted highest (ctx capped at {CTX_WEIGHT_CAP:.2f}, GoE gate capped at 0.50)")

        _flat = " — ⚠ near-flat distribution, verdict is low-signal" if confidence < 0.30 else ""
        _t(trace, "Meta-learner (GatingEnsembleNet)", "info",
           f"top: {_top3(all_scores)}{_flat}", pred_label, confidence)

        guard_conflict = None

        if goe_max >= 0.75 and pred_label != goe_top:
            blend   = min(goe_max * 0.80, 0.65)
            target  = {e: float(goe_raw[i]) for i, e in enumerate(EMOTION_LABELS)}
            blended = {e: (1.0 - blend) * all_scores.get(e, 0.0) + blend * target[e] for e in EMOTION_LABELS}
            _prev = pred_label
            pred_label = max(blended, key=blended.get)
            confidence, all_scores = blended[pred_label], blended
            logger.warning(f"[NLP-guard/GoE] {goe_top}({goe_max:.2f}) → {pred_label}({confidence:.2f})")
            _t(trace, "NLP-guard / GoEmotions", "applied",
               f"Raw GoE {goe_top} ({goe_max:.2f}) ≥ 0.75 contradicted meta '{_prev}' — blended {blend:.0%} toward GoE",
               pred_label, confidence)

        elif (bert_top in EMOTION_LABELS) and bert_max >= 0.80 and pred_label != bert_top:
            _bert_subtypes = {
                "joy":     {"joy","amusement","excitement","admiration","love","approval","caring","gratitude","optimism","pride","desire","relief"},
                "sadness": {"sadness","grief","remorse","disappointment"},
                "anger":   {"anger","annoyance","disapproval"},
                "fear":    {"fear","nervousness"},
                "neutral": {"neutral","realization"},
                "surprise":{"surprise","realization","confusion"},
                "disgust": {"disgust","disapproval"},
            }
            # The guard stands down when the other experts contradict BERT — a lone
            # high-confidence BERT must not veto a meta+GoE consensus (hyperbole idioms).
            vader_compound = float(feature_vector[0, 3])
            consensus      = pred_label == goe_top and goe_max >= 0.50
            cross_conflict = goe_max >= 0.60 and _GOE_TO_EKMAN.get(goe_top, "neutral") != bert_top
            valence_veto   = bert_top in {"fear", "anger", "sadness", "disgust"} and vader_compound >= 0.30
            if consensus or cross_conflict or valence_veto:
                if cross_conflict:
                    guard_conflict = "Expert conflict: BERT contradicts GoEmotions (possible hyperbole)"
                _reasons = [r for r, on in (("meta+GoE consensus", consensus),
                                            ("cross-expert conflict", cross_conflict),
                                            ("positive-valence veto", valence_veto)) if on]
                _t(trace, "NLP-guard / BERT", "suppressed",
                   f"BERT {bert_top} ({bert_max:.2f}) wanted to override, stood down: {', '.join(_reasons)}",
                   pred_label, confidence)
                logger.debug(
                    f"[NLP-guard/BERT] suppressed (consensus={consensus} cross={cross_conflict} "
                    f"valence={valence_veto}) bert={bert_top}({bert_max:.2f}) goe={goe_top}({goe_max:.2f})")
            elif pred_label not in _bert_subtypes.get(bert_top, {bert_top}):
                blend   = min(bert_max * 0.70, 0.55)
                target  = {e: (1.0 if e == bert_top else 0.0) for e in EMOTION_LABELS}
                blended = {e: (1.0 - blend) * all_scores.get(e, 0.0) + blend * target[e] for e in EMOTION_LABELS}
                _prev = pred_label
                pred_label = max(blended, key=blended.get)
                confidence, all_scores = blended[pred_label], blended
                logger.warning(f"[NLP-guard/BERT] {bert_top}({bert_max:.2f}) → {pred_label}({confidence:.2f})")
                _t(trace, "NLP-guard / BERT", "applied",
                   f"Raw BERT {bert_top} ({bert_max:.2f}) ≥ 0.80 outside meta '{_prev}' family — blended {blend:.0%} toward BERT",
                   pred_label, confidence)
            else:
                _t(trace, "NLP-guard / BERT", "skipped",
                   f"BERT {bert_top} ({bert_max:.2f}) confident, but meta '{pred_label}' is within the {bert_top} family — no override needed")

        else:
            _goe_why  = ("meta already agrees with GoE" if pred_label == goe_top
                         else f"GoE max {goe_max:.2f} < 0.75 threshold")
            _bert_why = ("meta already agrees with BERT" if pred_label == bert_top
                         else (f"BERT max {bert_max:.2f} < 0.80 threshold" if bert_max < 0.80
                               else f"BERT top '{bert_top}' is not a GoE label"))
            _t(trace, "NLP-guards", "skipped", f"GoE guard: {_goe_why} · BERT guard: {_bert_why}")

        heuristic_sarcasm, conflict_desc = detect_emotional_conflicts(feature_vector)
        learned_sarcasm = float(feature_vector[0, ML_DIM + CDM_CTX_DIM + PRIOR_DIM])
        # Combined sarcasm evidence: the trained classifier leads, the VADER/GoE
        # heuristic corroborates. Either alone is weak — the heuristic misfires on
        # playful banter (positive wording + mild negative GoE mass), the classifier
        # can miss novel phrasings — but agreement between them is a strong signal.
        # Trace of negative GoE mass (anger/annoyance/disappointment/disapproval/
        # disgust). Real sarcasm leaks some (🙄 → annoyance); sincere gushing
        # ("YESSS I love you SO much") reads ~zero — the classifier's worst
        # false-positive mode, which must not invert a sincere positive.
        _goe_block = feature_vector[0, 11:39]
        neg_cue = float(max(_goe_block[2], _goe_block[3], _goe_block[9],
                            _goe_block[10], _goe_block[11]))
        if learned_sarcasm >= 0.5 and heuristic_sarcasm >= 0.5:
            sarcasm_score = max(learned_sarcasm, heuristic_sarcasm)
        elif learned_sarcasm >= 0.75 and neg_cue >= 0.15:
            # Far above the classifier's tuned operating threshold (0.30) AND a
            # negative undertone exists — heuristic silence must not dilute it,
            # or the inversion is too weak to flip anything (0.93 blended to
            # 0.60 → 12% shift). Moderate scores (the 0.5-0.75 band is noisy,
            # precision ~0.63) still need full heuristic corroboration.
            sarcasm_score = learned_sarcasm
        else:
            sarcasm_score = 0.65 * learned_sarcasm + 0.35 * heuristic_sarcasm
            if neg_cue < 0.15:
                # No negative undertone at all → never invert. The classifier's
                # worst false-positive mode is sincere affection ("i love you"
                # scores 0.93); without any negative GoE trace an inversion can
                # only do harm.
                sarcasm_score = min(sarcasm_score, SARCASM_INVERSION_THRESHOLD)
        if heuristic_sarcasm > 0 or learned_sarcasm > 0:
            _t(trace, "Sarcasm evidence", "info",
               f"classifier {learned_sarcasm:.2f} · heuristic {heuristic_sarcasm:.2f} "
               f"· neg-cue {neg_cue:.2f} → combined {sarcasm_score:.2f} "
               f"(inversion needs > {SARCASM_INVERSION_THRESHOLD:.2f})")
        if conflict_desc is None and guard_conflict:
            conflict_desc = guard_conflict
        if conflict_desc:
            _t(trace, "Conflict detector", "info", conflict_desc)

        if sarcasm_score > SARCASM_INVERSION_THRESHOLD:
            _POS = [EMOTION_LABELS.index(e) for e in ("joy","amusement","admiration","approval","excitement","love","optimism","pride","caring","gratitude") if e in EMOTION_LABELS]
            _NEG = [EMOTION_LABELS.index(e) for e in ("annoyance","disappointment","anger","disapproval","disgust","grief","remorse","sadness") if e in EMOTION_LABELS]
            inv  = min((sarcasm_score - SARCASM_INVERSION_THRESHOLD)
                       / (1.0 - SARCASM_INVERSION_THRESHOLD), 1.0) * 0.6
            probs = np.array([all_scores.get(e, 0.0) for e in EMOTION_LABELS], dtype=np.float32)
            for pos, neg in zip(_POS, _NEG):
                t = probs[pos] * inv
                probs[neg] += t; probs[pos] -= t
            probs = np.clip(probs, 0.0, None)
            s = probs.sum()
            if s > 1e-6:
                probs /= s
            all_scores = {e: float(probs[i]) for i, e in enumerate(EMOTION_LABELS)}
            _prev = pred_label
            pred_label = max(all_scores, key=all_scores.get)
            confidence = all_scores[pred_label]
            _t(trace, "Sarcasm inversion", "applied",
               f"Sarcasm score {sarcasm_score:.2f} > {SARCASM_INVERSION_THRESHOLD:.2f} — shifted {inv:.0%} of positive mass onto paired negatives ('{_prev}' → '{pred_label}')",
               pred_label, confidence)
        else:
            _t(trace, "Sarcasm inversion", "skipped",
               f"conflict score {sarcasm_score:.2f} ≤ {SARCASM_INVERSION_THRESHOLD:.2f} — scores unchanged")

        all_scores["ekman_group"] = _GOE_TO_EKMAN.get(pred_label, "neutral")
        return pred_label, confidence, all_scores, sarcasm_score, conflict_desc, gate_alpha

    except Exception as e:
        logger.warning(f"Predict error: {e}. Returning neutral.")
        return "neutral", 0.0, {}, 0.0, None, None


def detect_emotional_conflicts(vec):
    try:
        v = vec.flatten()
        v_pos, v_cmp     = float(v[2]), float(v[3])
        bert_joy         = float(v[7])
        bert_neutral     = float(v[8])
        neg_emo = max(float(v[11 + 3]), float(v[11 + 10]), float(v[11 + 11]))
        pos_text = (v_pos + bert_joy) / 2.0
        if pos_text > 0.6 and neg_emo > 0.4:
            return min(pos_text, neg_emo) * 1.2, "Cognitive Dissonance: positive text with dismissive cues."
        if v_cmp > 0.8 and neg_emo > 0.2:
            return 0.5 + neg_emo, "Sarcasm detected: semantic praise contradicts frustration."
        if bert_neutral > 0.7 and neg_emo > 0.1:
            return 0.4, "Passive-aggression: formal neutral text with underlying tension."
        # Hyperbole (inverse sarcasm): negative wording, positive intent. Score stays
        # below 0.5 — the >=0.5 path flips positive→negative, the wrong direction here.
        bert_negative = max(float(v[4]), float(v[5]), float(v[6]), float(v[9]))
        goe_raw       = v[11:39]
        goe_top       = EMOTION_LABELS[int(goe_raw.argmax())]
        if bert_negative > 0.5 and v_cmp > 0.3 and _GOE_TO_EKMAN.get(goe_top) == "joy":
            return 0.3, "Hyperbole: negative wording with positive intent."
        return 0.0, None
    except Exception:
        return 0.0, None


def calculate_feature_impacts(model, feature_vector, predicted_emotion):
    try:
        classes = list(model.classes_)
        if predicted_emotion not in classes:
            return {}
        idx        = classes.index(predicted_emotion)
        base_proba = float(model.predict_proba(feature_vector)[0][idx])
        blocks = {"VADER": slice(0, 4), "BERT": slice(4, 11), "GoEmotions": slice(11, ML_DIM), "Context": slice(ML_DIM, FEATURE_DIM)}
        impacts = {}
        for name, sl in blocks.items():
            m = feature_vector.copy(); m[0, sl] = 0.0
            impacts[name] = abs(base_proba - float(model.predict_proba(m)[0][idx]))
        total = sum(impacts.values())
        return {k: round(v / total, 4) for k, v in impacts.items()} if total > 0 else {k: 0.25 for k in impacts}
    except Exception as e:
        logger.warning(f"Failed to calculate impacts: {e}")
        return {}


def apply_context_correction(scores, context_vector, prev_emotion="neutral"):
    return scores, 0.0


def _log_metadata():
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
