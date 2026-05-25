"""
context_engine_service/cdm.py — Conversation Dynamics Machine (CDM)

Classifies each conversation turn into one of N_CDM_STATES latent states
based on 4 real-time features derived entirely from Redis (no ML model calls).

Two backends, chosen at load time:
  - GMM (sklearn GaussianMixture) if models/cdm.pkl exists
  - Rule-based fallback (always available, no training required)

State indices match shared.constants.CDM_STATES.
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Tuple

logger = logging.getLogger("context_engine")

# ── State indices (must match shared/constants.py CDM_STATES order) ───────────
NEUTRAL_EXPLORE    = 0
ASCENDING_POSITIVE = 1
CONFLICT_RISE      = 2
TOPIC_ANCHOR       = 3
CONVERGENCE        = 4
EMOTIONAL_PEAK     = 5
DIVERGING          = 6
N_STATES           = 7

_CDM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "cdm.pkl")


class ConversationDynamicsMachine:
    """
    Classifies the current conversational state from 4 lightweight features.

    Features (in order):
      velocity        — Δvalence (v_t − v_{t-1})
      entropy         — Shannon entropy of the GoEmotions distribution
      speaker_delta   — std of per-speaker valences in this conversation
      topic_coherence — cosine similarity of current vs previous embedding
    """

    def __init__(self):
        self._gmm    = None
        self._scaler = None
        self._load_gmm()

    def _load_gmm(self):
        try:
            if os.path.exists(_CDM_MODEL_PATH):
                data = pickle.load(open(_CDM_MODEL_PATH, "rb"))
                self._gmm    = data["gmm"]
                self._scaler = data["scaler"]
                logger.info(f"CDM: loaded GMM ({self._gmm.n_components} states) from {_CDM_MODEL_PATH}")
            else:
                logger.info("CDM: no cdm.pkl found — using rule-based fallback")
        except Exception as e:
            logger.warning(f"CDM: failed to load GMM ({e}) — using rule-based fallback")
            self._gmm = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify(
        self,
        velocity:        float,
        entropy:         float,
        speaker_delta:   float,
        topic_coherence: float,
    ) -> np.ndarray:
        """
        Returns a probability vector of shape (N_STATES,) over conversation states.
        Uses GMM if available, rule-based otherwise.
        """
        if self._gmm is not None:
            return self._gmm_classify(velocity, entropy, speaker_delta, topic_coherence)
        return self._rule_classify(velocity, entropy, speaker_delta, topic_coherence)

    @staticmethod
    def entry_abruptness(features_now: List[float], features_prev: List[float]) -> float:
        """L2 distance between consecutive feature points, clamped to [0, 1]."""
        delta = np.array(features_now) - np.array(features_prev)
        return float(min(np.linalg.norm(delta) / 4.0, 1.0))  # /4 normalises typical range

    # ── GMM backend ────────────────────────────────────────────────────────────

    def _gmm_classify(self, velocity, entropy, speaker_delta, topic_coherence) -> np.ndarray:
        try:
            x = self._scaler.transform([[velocity, entropy, speaker_delta, topic_coherence]])
            probs = self._gmm.predict_proba(x)[0]
            # Pad or trim to exactly N_STATES
            out = np.zeros(N_STATES, dtype=np.float32)
            n   = min(len(probs), N_STATES)
            out[:n] = probs[:n]
            return out
        except Exception as e:
            logger.warning(f"CDM GMM inference error: {e} — falling back to rules")
            return self._rule_classify(velocity, entropy, speaker_delta, topic_coherence)

    # ── Rule-based backend ─────────────────────────────────────────────────────

    def _rule_classify(self, velocity, entropy, speaker_delta, topic_coherence) -> np.ndarray:
        """
        Deterministic state assignment from feature thresholds.
        Returns a one-hot-like vector (winning state = 0.85, runner-up = 0.15).
        Priority order matters — first matching rule wins.
        """
        if entropy > 2.5:
            winner = EMOTIONAL_PEAK
        elif speaker_delta > 0.40:
            winner = DIVERGING
        elif topic_coherence > 0.75 and entropy < 1.5:
            winner = TOPIC_ANCHOR
        elif velocity > 0.15:
            winner = ASCENDING_POSITIVE
        elif velocity < -0.15:
            winner = CONFLICT_RISE
        elif speaker_delta < 0.12 and abs(velocity) < 0.06:
            winner = CONVERGENCE
        else:
            winner = NEUTRAL_EXPLORE

        probs = np.full(N_STATES, 0.15 / (N_STATES - 1), dtype=np.float32)
        probs[winner] = 0.85
        return probs


# Module-level singleton — instantiated once when the service starts
CDM = ConversationDynamicsMachine()
