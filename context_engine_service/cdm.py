"""
context_engine_service/cdm.py — Conversation Dynamics Machine (CDM) v3.0
Parallel Belief State Machine (PBSM).

The machine holds a *probability distribution* (belief) over all 28 GoEmotions
states rather than a single discrete state. It can therefore be in several
states simultaneously — the "active" states are the top-3 of the belief vector.

Belief update is a Bayesian HMM step:

    B_pred = (1 - μ)·(Tᵀ · B_t) + μ·B_t      # temporal dynamics + inertia
    B_obs  = B_pred · (obs + ε)               # condition on GoEmotions evidence
    B_new  = B_obs / B_obs.sum()              # normalise

where:
    B_t  — previous belief over 28 emotions (sums to 1)
    obs  — normalised last-turn GoEmotions distribution (the observation)
    T    — 28×28 transition matrix, initialised from Russell's (1980)
           valence-arousal circumplex; every entry > 0 (fully connected).
    μ(r) — inertia weight that grows with how long the top-1 state has held,
           preventing the belief from collapsing to the raw observation each
           turn (emotional inertia — EmotionIC, 2023).

State indices and names follow shared.constants.EMOTION_LABELS order.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import logging

from shared.constants import EMOTION_LABELS, CDM_N_STATES

logger = logging.getLogger("context_engine")

N_STATES = CDM_N_STATES  # 28

# ── Russell (1980) circumplex coordinates per emotion: (valence ∈ [-1,1], arousal ∈ [0,1]) ──
# Order MUST match shared.constants.EMOTION_LABELS.
_VA = {
    'admiration':    (0.60, 0.50), 'amusement':     (0.70, 0.60),
    'anger':         (-0.70, 0.85), 'annoyance':     (-0.40, 0.50),
    'approval':      (0.50, 0.35), 'caring':        (0.60, 0.40),
    'confusion':     (-0.15, 0.45), 'curiosity':     (0.30, 0.55),
    'desire':        (0.40, 0.60), 'disappointment': (-0.50, 0.35),
    'disapproval':   (-0.45, 0.45), 'disgust':       (-0.70, 0.60),
    'embarrassment': (-0.35, 0.55), 'excitement':    (0.70, 0.85),
    'fear':          (-0.65, 0.80), 'gratitude':     (0.70, 0.40),
    'grief':         (-0.85, 0.50), 'joy':           (0.85, 0.70),
    'love':          (0.85, 0.60), 'nervousness':   (-0.40, 0.65),
    'optimism':      (0.60, 0.50), 'pride':         (0.65, 0.60),
    'realization':   (0.10, 0.45), 'relief':        (0.50, 0.30),
    'remorse':       (-0.55, 0.40), 'sadness':       (-0.70, 0.35),
    'surprise':      (0.15, 0.75), 'neutral':        (0.00, 0.10),
}


@dataclass
class StateDiagnostics:
    """Rich output of a single PBSM transition step."""
    belief:                 List[float]   # 28-dim distribution → ctx[0:28]
    top3_states:            List[int]     # 3 indices, descending probability
    top3_probs:             List[float]
    top1_name:              str
    top2_name:              str
    top3_name:              str
    prev_top1_idx:          int
    transition_probability: float         # P[top1] after the update
    state_momentum:         float          # μ(r) at decision time
    entry_abruptness:       float          # 1 - cosine_sim(B_old, B_new)
    catalyst_state:         str            # emotion that gained the most belief mass
    catalyst_obs:           str            # most salient observed emotion this turn


class ParallelBeliefStateMachine:
    """
    28-state Bayesian belief machine over the GoEmotions vocabulary.

    Usage per turn:
        diag = CDM.transition(prev_belief, observation, residency)
        ctx[0:28] = diag.belief
        # persist diag.belief + top1 + residency_count to Redis

    The caller is responsible for persisting the belief between turns.
    """

    def __init__(
        self,
        mu0: float = 0.40,
        mu1: float = 0.40,
        r_cap: float = 0.80,
        sigma: float = 0.50,
        eps: float = 0.01,
    ):
        self.mu0   = mu0
        self.mu1   = mu1
        self.r_cap = r_cap
        self.eps   = eps
        self._T    = self._build_transition_matrix(sigma)

    # ── Initialisation ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_transition_matrix(sigma: float) -> np.ndarray:
        """
        Row-normalised 28×28 affinity matrix from valence-arousal proximity.
        T[i, j] = exp(-||VA_i - VA_j||² / σ²) + ε   (every entry > 0 — fully connected)
        Row i is the distribution of "what state i tends to flow toward".
        """
        coords = np.array([_VA[e] for e in EMOTION_LABELS], dtype=np.float64)  # [28, 2]
        diff   = coords[:, None, :] - coords[None, :, :]                        # [28, 28, 2]
        d2     = (diff ** 2).sum(axis=-1)                                       # [28, 28]
        T      = np.exp(-d2 / (sigma ** 2)) + 1e-3
        T      = T / T.sum(axis=1, keepdims=True)                               # row-normalise
        return T

    def uniform_belief(self) -> np.ndarray:
        """Cold-start belief — uniform over all 28 states."""
        return np.ones(N_STATES, dtype=np.float64) / N_STATES

    # ── Public API ──────────────────────────────────────────────────────────────

    def transition(
        self,
        current_belief: np.ndarray,
        observation:    np.ndarray,
        residency:      float,
    ) -> StateDiagnostics:
        """
        One Bayesian HMM belief-update step.

        Args:
            current_belief: [28] prior belief (will be sanitised / re-normalised)
            observation:    [28] normalised last-turn GoEmotions distribution
            residency:      float in [0, 1] — how long the top-1 state has held

        Returns:
            StateDiagnostics with the new belief, top-3 active states, and
            interpretability fields (catalyst, momentum, abruptness).
        """
        B_t = self._sanitise(current_belief)
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape[0] != N_STATES or not np.isfinite(obs).all():
            obs = np.zeros(N_STATES, dtype=np.float64)

        prev_top1 = int(np.argmax(B_t))

        # Step 1 — temporal prediction with inertia
        mu     = self.mu0 + self.mu1 * float(np.clip(residency, 0.0, self.r_cap))
        B_pred = (1.0 - mu) * (self._T.T @ B_t) + mu * B_t

        # Step 2 — Bayesian observation update (Laplace-smoothed)
        B_obs = B_pred * (obs + self.eps)

        # Step 3 — normalise (guard against degenerate all-zero product)
        total = float(B_obs.sum())
        B_new = (B_obs / total) if total > 1e-9 else self.uniform_belief()

        # Top-3 simultaneously active states
        order      = np.argsort(B_new)[::-1]
        top3       = [int(i) for i in order[:3]]
        top3_probs = [float(B_new[i]) for i in top3]

        # Interpretability
        shift          = B_new - B_t
        catalyst_state = EMOTION_LABELS[int(np.argmax(shift))]
        catalyst_obs   = EMOTION_LABELS[int(np.argmax(obs))] if obs.sum() > 0 else "neutral"

        cos = float(np.dot(B_t, B_new) / (np.linalg.norm(B_t) * np.linalg.norm(B_new) + 1e-9))
        abruptness = float(np.clip(1.0 - cos, 0.0, 1.0))

        return StateDiagnostics(
            belief=B_new.tolist(),
            top3_states=top3,
            top3_probs=top3_probs,
            top1_name=EMOTION_LABELS[top3[0]],
            top2_name=EMOTION_LABELS[top3[1]],
            top3_name=EMOTION_LABELS[top3[2]],
            prev_top1_idx=prev_top1,
            transition_probability=top3_probs[0],
            state_momentum=float(mu),
            entry_abruptness=abruptness,
            catalyst_state=catalyst_state,
            catalyst_obs=catalyst_obs,
        )

    def get_state_prior(self, top3_states: List[int]) -> np.ndarray:
        """
        Per-active-state predictive prior: the transition-matrix rows for the
        top-3 states — i.e. what each currently-active state expects next.
        Shape [k, 28]. Consumed downstream as a T-1 → T sensitivity bias.
        """
        rows = [self._T[s] for s in top3_states if 0 <= s < N_STATES]
        return np.stack(rows) if rows else self._T[EMOTION_LABELS.index("neutral")][None, :]

    @staticmethod
    def one_hot(state_idx: int) -> np.ndarray:
        """Backward-compat helper — one-hot float32 vector of length N_STATES."""
        v = np.zeros(N_STATES, dtype=np.float32)
        if 0 <= state_idx < N_STATES:
            v[state_idx] = 1.0
        return v

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sanitise(self, belief) -> np.ndarray:
        """Coerce any input into a valid 28-dim probability vector."""
        b = np.asarray(belief, dtype=np.float64)
        if b.shape != (N_STATES,) or not np.isfinite(b).all() or b.sum() <= 0:
            return self.uniform_belief()
        return b / b.sum()


def observation_from_emotions(emotions: dict) -> np.ndarray:
    """
    Convert a GoEmotions {label: score} dict into a normalised 28-dim
    observation vector in EMOTION_LABELS order (the machine's state order).
    Returns all-zeros if the dict is empty (→ neutral Bayesian update).
    """
    v = np.array([float(emotions.get(e, 0.0)) for e in EMOTION_LABELS], dtype=np.float64)
    s = v.sum()
    return v / s if s > 0 else np.zeros(N_STATES, dtype=np.float64)


# Module-level singleton (name preserved for caller compatibility)
CDM = ParallelBeliefStateMachine()
