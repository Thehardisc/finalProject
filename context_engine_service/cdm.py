"""
context_engine_service/cdm.py — Conversation Dynamics Machine (CDM)

Deterministic Finite State Machine over 7 conversation states.
Transitions fire only when a feature threshold is crossed; otherwise the
machine stays in its current state.  Output is always a one-hot vector
(no probability distribution, no GMM).

State indices match shared.constants.CDM_STATES order.
"""

import numpy as np
import logging

logger = logging.getLogger("context_engine")

# ── State indices (must match shared/constants.py CDM_STATES order) ──────────
NEUTRAL_EXPLORE    = 0
ASCENDING_POSITIVE = 1
CONFLICT_RISE      = 2
TOPIC_ANCHOR       = 3
CONVERGENCE        = 4
EMOTIONAL_PEAK     = 5
DIVERGING          = 6
N_STATES           = 7

# Transition guard table: (from_state, to_state) pairs that are forbidden.
# Prevents teleporting across emotionally incompatible states in one step.
_BLOCKED = {
    (CONVERGENCE,        EMOTIONAL_PEAK),
    (EMOTIONAL_PEAK,     CONVERGENCE),
    (ASCENDING_POSITIVE, CONFLICT_RISE),
    (CONFLICT_RISE,      ASCENDING_POSITIVE),
}


class ConversationDynamicsMachine:
    """
    DFSM over 7 conversation states.

    Call `transition(current_state, velocity, entropy, speaker_delta,
    topic_coherence)` each turn.  Returns the next state index.  Caller is
    responsible for persisting `current_state` between calls.

    `one_hot(state_idx)` returns a 7-dim float32 array for the context vector.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def transition(
        self,
        current_state:   int,
        velocity:        float,
        entropy:         float,
        speaker_delta:   float,
        topic_coherence: float,
    ) -> int:
        """
        Deterministic transition function.
        Returns next state index (may equal current_state if no rule fires).
        Priority order: emotional extremes first, then directional, then stable.
        """
        candidate = self._rules(velocity, entropy, speaker_delta, topic_coherence, current_state)
        if (current_state, candidate) in _BLOCKED:
            return current_state
        return candidate

    @staticmethod
    def one_hot(state_idx: int) -> np.ndarray:
        """Returns a one-hot float32 vector of length N_STATES."""
        v = np.zeros(N_STATES, dtype=np.float32)
        v[state_idx] = 1.0
        return v

    @staticmethod
    def entry_abruptness(state_prev: int, state_next: int) -> float:
        """Normalised L1 distance between consecutive states, clamped to [0, 1]."""
        return abs(state_next - state_prev) / float(N_STATES - 1)

    # ── Transition rules ──────────────────────────────────────────────────────

    @staticmethod
    def _rules(
        velocity:        float,
        entropy:         float,
        speaker_delta:   float,
        topic_coherence: float,
        current_state:   int,
    ) -> int:
        # Tier 1 — extreme signal; overrides any current state
        if entropy > 2.5:
            return EMOTIONAL_PEAK
        if speaker_delta > 0.40:
            return DIVERGING

        # Tier 2 — directional momentum
        if velocity > 0.15:
            return ASCENDING_POSITIVE
        if velocity < -0.15:
            return CONFLICT_RISE

        # Tier 3 — stable patterns
        if topic_coherence > 0.75 and entropy < 1.5:
            return TOPIC_ANCHOR
        if speaker_delta < 0.12 and abs(velocity) < 0.06:
            return CONVERGENCE

        # No rule fires — stay put (key DFSM property: no implicit drift to NEUTRAL)
        return current_state


# Module-level singleton
CDM = ConversationDynamicsMachine()
