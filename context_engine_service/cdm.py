"""
context_engine_service/cdm.py — Conversation Intent Machine v4.0

CDM v4.0 replaces the 14-state momentum-based machine (v3) with a 15-state
intent-based machine grounded in HMM transition dynamics learned from DailyDialog.

States represent *speaker intent* (what the speaker is trying to accomplish)
rather than *conversation dynamics* (how the conversation is moving).

Inference: Bayesian forward filtering (α-pass) over a CategoricalHMM:
    α_t = normalise( (α_{t-1} · A) ⊙ B[:, obs_t] )

Observation encoding:  obs_t = act_idx × 7 + emotion_idx  (0-27)
  act_idx:     0=inform  1=question  2=directive  3=commissive  (heuristic from text)
  emotion_idx: 0=neutral 1=anger 2=disgust 3=fear 4=happiness 5=sadness 6=surprise
               (mapped from previous-turn GoEmotions argmax)

A and B are loaded from models/cdm_hmm.pkl (built by train_cdm_hmm.py).

Public API (backward-compatible with v3 CDM):
    transition(current_state, velocity, entropy, speaker_delta, topic_coherence,
               hmm_alpha=None, top_emotion="neutral", message_text="")
        → StateDiagnostics
    one_hot(state_idx)        → np.ndarray [N_CDM_STATES]
    entry_abruptness(a, b)    → float
    get_state_prior(state)    → dict

    reload_hmm()              → reloads A, B, π from pkl (call after training completes)
"""

from __future__ import annotations

import logging
import os
import pickle
import numpy as np
import spacy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("cdm")

# ── State indices ─────────────────────────────────────────────────────────────
NEUTRAL        =  0
WARMTH         =  1
PRAISE         =  2
HELP_REQUEST   =  3
HUMOR          =  4
TENSION        =  5
CONFLICT       =  6
ARGUMENT       =  7
WITHDRAWAL     =  8
RECONCILIATION =  9
CURIOSITY      = 10
ASSERTIVENESS  = 11
EMPATHY        = 12
FRUSTRATION    = 13
AGREEMENT      = 14

N_CDM_STATES = 15

_STATE_NAMES: List[str] = [
    "NEUTRAL", "WARMTH", "PRAISE", "HELP_REQUEST", "HUMOR",
    "TENSION", "CONFLICT", "ARGUMENT", "WITHDRAWAL", "RECONCILIATION",
    "CURIOSITY", "ASSERTIVENESS", "EMPATHY", "FRUSTRATION", "AGREEMENT",
]

# ── HMM parameter storage ─────────────────────────────────────────────────────
_HMM_PATH = Path(os.environ.get("MODEL_DIR", "/app/models")) / "cdm_hmm.pkl"
_N_OBS    = 28   # 4 acts × 7 emotions
_N_EMOT   = 7


def _load_hmm():
    if not _HMM_PATH.exists():
        logger.warning(f"CDM HMM not found at {_HMM_PATH} — using uniform priors.")
        unif_s = np.ones(N_CDM_STATES, dtype=np.float64) / N_CDM_STATES
        unif_A = np.ones((N_CDM_STATES, N_CDM_STATES), dtype=np.float64) / N_CDM_STATES
        unif_B = np.ones((N_CDM_STATES, _N_OBS),        dtype=np.float64) / _N_OBS
        return unif_s, unif_A, unif_B
    try:
        with open(_HMM_PATH, "rb") as f:
            d = pickle.load(f)
        sp = np.array(d["start_prob"],    dtype=np.float64)
        A  = np.array(d["transmat"],      dtype=np.float64)
        B  = np.array(d["emissionprob"],  dtype=np.float64)
        sp /= sp.sum()
        A  /= A.sum(axis=1, keepdims=True)
        B  /= B.sum(axis=1, keepdims=True)
        logger.info(f"CDM HMM loaded: {N_CDM_STATES} intent states, {_N_OBS} obs.")
        return sp, A, B
    except Exception as e:
        logger.warning(f"CDM HMM load error: {e} — uniform priors.")
        unif = np.ones(N_CDM_STATES, dtype=np.float64) / N_CDM_STATES
        return unif.copy(), np.tile(unif, (N_CDM_STATES, 1)), np.ones((N_CDM_STATES, _N_OBS)) / _N_OBS


_STARTPROB, _TRANSMAT, _EMISSIONPROB = _load_hmm()


def reload_hmm() -> None:
    """Reload HMM parameters from disk — call after train_cdm_hmm.ensure_model() completes."""
    global _STARTPROB, _TRANSMAT, _EMISSIONPROB
    _STARTPROB, _TRANSMAT, _EMISSIONPROB = _load_hmm()


# ── Observation encoding ──────────────────────────────────────────────────────

# GoEmotions label → DailyDialog 7-class emotion index
_GOEMO_TO_IDX: Dict[str, int] = {
    "neutral": 0, "realization": 0, "confusion": 0,
    "anger": 1, "annoyance": 1, "disapproval": 1,
    "disgust": 2,
    "fear": 3, "nervousness": 3,
    "joy": 4, "admiration": 4, "amusement": 4, "approval": 4,
    "caring": 4, "desire": 4, "excitement": 4, "gratitude": 4,
    "love": 4, "optimism": 4, "pride": 4, "relief": 4,
    "sadness": 5, "disappointment": 5, "embarrassment": 5,
    "grief": 5, "remorse": 5,
    "surprise": 6, "curiosity": 6,
}

try:
    _nlp = spacy.load("en_core_web_sm", disable=["ner"])
except Exception as e:
    logger.warning(f"Failed to load spacy model: {e}. Falling back to rule-based dialog acts.")
    _nlp = None

def _dialog_act_hint(text: str) -> int:
    """SpaCy-based dialog-act classifier. Returns 0=inform 1=question 2=directive 3=commissive."""
    t = text.strip()
    if not t:
        return 0

    if not _nlp:
        t_low = t.lower()
        if t_low.endswith("?"):
            return 1
        if t_low.startswith(("i will", "i'll", "we will", "we'll")):
            return 3
        if t_low.startswith(("please", "could you", "can you", "do ")):
            return 2
        return 0

    doc = _nlp(t)
    
    if t.endswith("?"):
        return 1
    for token in doc:
        if token.tag_ in ("WP", "WRB") and token.dep_ in ("nsubj", "attr", "advmod"):
            return 1

    for token in doc:
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            nsubj = [child for child in token.children if child.dep_ == "nsubj"]
            if nsubj and nsubj[0].text.lower() in ("i", "we"):
                aux = [child for child in token.children if child.dep_ == "aux"]
                if (aux and aux[0].text.lower() in ("will", "'ll", "shall", "would")) or token.lemma_.lower() in ("promise", "guarantee", "swear"):
                    return 3

    for token in doc:
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            if token.tag_ == "VB":
                nsubj = [child for child in token.children if child.dep_ == "nsubj"]
                if not nsubj:
                    return 2
            aux = [child for child in token.children if child.dep_ == "aux"]
            if aux and aux[0].lemma_.lower() in ("must", "should", "need"):
                return 2

    return 0


def _emotion_to_idx(emotion: str) -> int:
    return _GOEMO_TO_IDX.get(emotion.lower().strip(), 0)


# ── NLP sensitivity priors (one dict per intent state) ────────────────────────
_STATE_PRIORS: List[Dict[str, float]] = [
    # 0  NEUTRAL
    {"sarcasm": 1.0, "negative": 1.0, "positive": 1.0, "conflict": 1.0, "complexity": 1.0},
    # 1  WARMTH
    {"sarcasm": 0.9, "negative": 1.3, "positive": 0.8, "conflict": 1.3, "complexity": 0.9},
    # 2  PRAISE
    {"sarcasm": 1.2, "negative": 1.4, "positive": 0.7, "conflict": 1.4, "complexity": 0.9},
    # 3  HELP_REQUEST
    {"sarcasm": 0.9, "negative": 1.1, "positive": 0.9, "conflict": 1.1, "complexity": 1.3},
    # 4  HUMOR
    {"sarcasm": 1.7, "negative": 1.2, "positive": 0.8, "conflict": 1.2, "complexity": 0.8},
    # 5  TENSION
    {"sarcasm": 1.3, "negative": 0.8, "positive": 1.3, "conflict": 0.7, "complexity": 1.1},
    # 6  CONFLICT
    {"sarcasm": 1.2, "negative": 0.7, "positive": 1.4, "conflict": 0.7, "complexity": 1.0},
    # 7  ARGUMENT
    {"sarcasm": 1.3, "negative": 0.75, "positive": 1.3, "conflict": 0.7, "complexity": 1.2},
    # 8  WITHDRAWAL
    {"sarcasm": 1.1, "negative": 0.9, "positive": 1.2, "conflict": 1.0, "complexity": 0.9},
    # 9  RECONCILIATION
    {"sarcasm": 1.1, "negative": 1.2, "positive": 0.9, "conflict": 1.2, "complexity": 1.0},
    # 10 CURIOSITY
    {"sarcasm": 1.0, "negative": 1.1, "positive": 0.9, "conflict": 1.1, "complexity": 1.3},
    # 11 ASSERTIVENESS
    {"sarcasm": 1.1, "negative": 1.1, "positive": 1.0, "conflict": 1.0, "complexity": 1.1},
    # 12 EMPATHY
    {"sarcasm": 0.9, "negative": 1.3, "positive": 0.8, "conflict": 1.3, "complexity": 0.9},
    # 13 FRUSTRATION
    {"sarcasm": 1.4, "negative": 0.8, "positive": 1.3, "conflict": 0.8, "complexity": 1.1},
    # 14 AGREEMENT
    {"sarcasm": 1.0, "negative": 1.3, "positive": 0.8, "conflict": 1.3, "complexity": 0.9},
]


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class StateDiagnostics:
    current_state:          int
    current_state_name:     str
    previous_state:         int
    previous_state_name:    str
    transition_probability: float
    state_momentum:         float
    catalyst_signals:       List[str]  = field(default_factory=list)
    one_hot:                np.ndarray = field(default_factory=lambda: np.zeros(N_CDM_STATES, dtype=np.float32))
    # HMM diagnostics
    hmm_alpha_next:         np.ndarray = field(default_factory=lambda: np.ones(N_CDM_STATES, dtype=np.float64) / N_CDM_STATES)
    hmm_confidence:         float = 0.0
    hmm_state_entropy:      float = 0.0
    hmm_emission_logprob:   float = 0.0
    hmm_top3_next:          List[float] = field(default_factory=lambda: [1/3, 1/3, 1/3])


# ── Intent State Machine ──────────────────────────────────────────────────────

class IntentStateMachine:
    """
    HMM-based 15-state intent machine.

    Uses Bayesian forward filtering (α-pass):
        alpha_pred = alpha_prev @ A          (predict)
        alpha_t    = alpha_pred * B[:, obs]  (update)
        alpha_t   /= sum(alpha_t)            (normalise)

    A and B are learned from DailyDialog by train_cdm_hmm.py.
    API is backward-compatible with the v3 ConversationDynamicsMachine.
    """

    def transition(
        self,
        current_state:   int,
        velocity:        float,
        entropy:         float,
        speaker_delta:   float,
        topic_coherence: float,
        hmm_alpha:       Optional[np.ndarray] = None,
        top_emotion:     str = "neutral",
        message_text:    str = "",
    ) -> StateDiagnostics:
        """
        Compute next intent state via HMM forward filtering.

        Args:
            current_state:   previous state index (used as fallback and for logging)
            velocity / entropy / speaker_delta / topic_coherence: kept for API compat,
                not used internally (HMM replaces hand-tuned signal weighting)
            hmm_alpha:       prior forward variable α_{t-1} [15]. None → _STARTPROB.
            top_emotion:     GoEmotions argmax from previous turn
            message_text:    current message (for dialog-act heuristic)
        """
        current_state = int(current_state) % N_CDM_STATES

        # ── Encode observation ─────────────────────────────────────────────
        act_idx  = _dialog_act_hint(message_text)
        emot_idx = _emotion_to_idx(top_emotion)
        obs      = act_idx * _N_EMOT + emot_idx  # 0-27

        # ── Forward filtering ──────────────────────────────────────────────
        if hmm_alpha is None or len(hmm_alpha) != N_CDM_STATES:
            alpha_prev = _STARTPROB.copy()
        else:
            alpha_prev = np.asarray(hmm_alpha, dtype=np.float64)

        alpha_pred = alpha_prev @ _TRANSMAT                 # predict  [15]
        alpha_t    = alpha_pred * _EMISSIONPROB[:, obs]     # update   [15]
        norm       = alpha_t.sum()
        alpha_t    = alpha_t / norm if norm > 1e-12 else _STARTPROB.copy()

        next_state    = int(np.argmax(alpha_t))
        trans_prob    = float(alpha_t[next_state])
        prev_momentum = float(alpha_t[current_state])

        # ── HMM diagnostics ────────────────────────────────────────────────
        hmm_entropy = float(-np.sum(alpha_t * np.log(alpha_t + 1e-12)))
        top3_next   = sorted(_TRANSMAT[next_state].tolist(), reverse=True)[:3]

        raw_lp    = float(np.log(_EMISSIONPROB[next_state, obs] + 1e-12))
        emit_norm = float(np.clip((raw_lp - np.log(1e-12)) / (0.0 - np.log(1e-12)), 0.0, 1.0))

        # ── Catalysts ──────────────────────────────────────────────────────
        act_names = ["inform", "question", "directive", "commissive"]
        catalysts = [f"act={act_names[act_idx]}, emotion={top_emotion}"]
        if next_state != current_state:
            catalysts.append(f"Intent: {_STATE_NAMES[current_state]} → {_STATE_NAMES[next_state]}")
        if trans_prob > 0.7:
            catalysts.append(f"High confidence ({trans_prob:.2f})")

        oh = np.zeros(N_CDM_STATES, dtype=np.float32)
        oh[next_state] = 1.0

        return StateDiagnostics(
            current_state          = next_state,
            current_state_name     = _STATE_NAMES[next_state],
            previous_state         = current_state,
            previous_state_name    = _STATE_NAMES[current_state],
            transition_probability = trans_prob,
            state_momentum         = prev_momentum,
            catalyst_signals       = catalysts,
            one_hot                = oh,
            hmm_alpha_next         = alpha_t,
            hmm_confidence         = trans_prob,
            hmm_state_entropy      = hmm_entropy,
            hmm_emission_logprob   = emit_norm,
            hmm_top3_next          = top3_next,
        )

    @staticmethod
    def one_hot(state_idx: int) -> np.ndarray:
        v = np.zeros(N_CDM_STATES, dtype=np.float32)
        v[int(state_idx) % N_CDM_STATES] = 1.0
        return v

    @staticmethod
    def entry_abruptness(state_prev: int, state_next: int) -> float:
        return abs(state_next - state_prev) / float(N_CDM_STATES - 1)

    @staticmethod
    def get_state_prior(state_idx: int) -> Dict[str, float]:
        return dict(_STATE_PRIORS[int(state_idx) % N_CDM_STATES])

    @staticmethod
    def get_next_probs(state_idx: int) -> List[float]:
        """Top-3 P(next | state_idx) from the learned transition matrix."""
        return sorted(_TRANSMAT[int(state_idx) % N_CDM_STATES].tolist(), reverse=True)[:3]


CDM = IntentStateMachine()
