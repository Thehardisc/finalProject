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

N_CDM_STATES = 15

_STATE_NAMES: List[str] = [
    "NEUTRAL", "WARMTH", "PRAISE", "HELP_REQUEST", "HUMOR",
    "TENSION", "CONFLICT", "ARGUMENT", "WITHDRAWAL", "RECONCILIATION",
    "CURIOSITY", "ASSERTIVENESS", "EMPATHY", "FRUSTRATION", "AGREEMENT",
]

_HMM_PATH = Path(os.environ.get("MODEL_DIR", "/app/models")) / "cdm_hmm.pkl"
_N_OBS    = 28
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
    global _STARTPROB, _TRANSMAT, _EMISSIONPROB
    _STARTPROB, _TRANSMAT, _EMISSIONPROB = _load_hmm()


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
    hmm_alpha_next:         np.ndarray = field(default_factory=lambda: np.ones(N_CDM_STATES, dtype=np.float64) / N_CDM_STATES)
    hmm_confidence:         float = 0.0
    hmm_state_entropy:      float = 0.0
    hmm_emission_logprob:   float = 0.0
    hmm_top3_next:          List[float] = field(default_factory=lambda: [1/3, 1/3, 1/3])


class IntentStateMachine:
    def transition(
        self,
        current_state:   int,
        velocity:        float,
        entropy:         float,
        topic_coherence: float,
        hmm_alpha:       Optional[np.ndarray] = None,
        top_emotion:     str = "neutral",
        message_text:    str = "",
    ) -> StateDiagnostics:
        current_state = int(current_state) % N_CDM_STATES

        act_idx  = _dialog_act_hint(message_text)
        emot_idx = _emotion_to_idx(top_emotion)
        obs      = act_idx * _N_EMOT + emot_idx

        if hmm_alpha is None or len(hmm_alpha) != N_CDM_STATES:
            alpha_prev = _STARTPROB.copy()
        else:
            alpha_prev = np.asarray(hmm_alpha, dtype=np.float64)

        alpha_pred = alpha_prev @ _TRANSMAT
        alpha_t    = alpha_pred * _EMISSIONPROB[:, obs]
        norm       = alpha_t.sum()
        alpha_t    = alpha_t / norm if norm > 1e-12 else _STARTPROB.copy()

        next_state    = int(np.argmax(alpha_t))
        trans_prob    = float(alpha_t[next_state])
        prev_momentum = float(alpha_t[current_state])

        hmm_entropy = float(-np.sum(alpha_t * np.log(alpha_t + 1e-12)))
        top3_next   = sorted(_TRANSMAT[next_state].tolist(), reverse=True)[:3]

        raw_lp    = float(np.log(_EMISSIONPROB[next_state, obs] + 1e-12))
        emit_norm = float(np.clip((raw_lp - np.log(1e-12)) / (0.0 - np.log(1e-12)), 0.0, 1.0))

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
    def get_next_probs(state_idx: int) -> List[float]:
        return sorted(_TRANSMAT[int(state_idx) % N_CDM_STATES].tolist(), reverse=True)[:3]


CDM = IntentStateMachine()
