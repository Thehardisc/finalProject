"""
context_engine_service/train_cdm_hmm.py

Trains a CategoricalHMM on DailyDialog to learn intent-state transition dynamics.

Saves a dict of numpy arrays to models/cdm_hmm.pkl:
  {
    "start_prob":    np.ndarray [N_STATES]        — initial state distribution
    "transmat":      np.ndarray [N_STATES, N_STATES] — P(next | current)
    "emissionprob":  np.ndarray [N_STATES, N_OBS]   — P(obs | state)
  }

N_STATES = 15  (intent states 0-14)
N_OBS    = 28  (4 dialog acts × 7 emotions)

Observation encoding:  obs = act_idx * 7 + emotion_idx
  act_idx:     0=inform  1=question  2=directive  3=commissive  (DailyDialog acts -1)
  emotion_idx: 0=neutral 1=anger 2=disgust 3=fear 4=happiness 5=sadness 6=surprise

Call ensure_model() at service startup — it is a no-op if the pkl already exists.
"""

import os
import logging
import pickle
import numpy as np
from pathlib import Path

logger = logging.getLogger("train_cdm_hmm")

N_STATES = 15
N_ACTS   = 4
N_EMOT   = 7
N_OBS    = N_ACTS * N_EMOT  # 28

_DEFAULT_MODEL_PATH = Path(os.environ.get("MODEL_DIR", "/app/models")) / "cdm_hmm.pkl"

# ── Heuristic (act, emotion) → intent state ───────────────────────────────────
# DailyDialog emotion: 0=neutral 1=anger 2=disgust 3=fear 4=happiness 5=sadness 6=surprise
# DailyDialog act (0-indexed): 0=inform 1=question 2=directive 3=commissive
#
#            neutral  anger  disgust  fear  happy  sadness  surprise
_HEURISTIC_MAP = [
    [0,       5,     5,      0,    2,     0,      4   ],  # act=inform
    [10,      7,    13,      3,   10,    10,     10   ],  # act=question
    [11,      6,     6,      3,   11,    11,     11   ],  # act=directive
    [14,      9,     9,      8,    1,    12,      1   ],  # act=commissive
]


def _obs(act: int, emot: int) -> int:
    return act * N_EMOT + emot


def _build_emission_prior() -> np.ndarray:
    """Build initial emission matrix from _HEURISTIC_MAP with Laplace smoothing."""
    E = np.ones((N_STATES, N_OBS), dtype=np.float64)
    for act in range(N_ACTS):
        for emot in range(N_EMOT):
            E[_HEURISTIC_MAP[act][emot], _obs(act, emot)] += 10.0
    E /= E.sum(axis=1, keepdims=True)
    return E


def _build_fallback_transmat() -> np.ndarray:
    """
    Topology-derived 15×15 transition matrix for when DailyDialog is unavailable.
    Semantically adjacent states share higher P(next|current).
    """
    A = np.ones((N_STATES, N_STATES), dtype=np.float64) * 0.5

    # NEUTRAL is a hub — reachable from everywhere
    A[:, 0] += 1.0

    # Positive collaborative cluster: WARMTH PRAISE EMPATHY AGREEMENT RECONCILIATION
    for i in [1, 2, 9, 12, 14]:
        for j in [1, 2, 9, 12, 14]:
            A[i, j] += 2.0

    # Exploratory cluster: CURIOSITY HELP_REQUEST HUMOR ASSERTIVENESS
    for i in [3, 4, 10, 11]:
        for j in [3, 4, 10, 11]:
            A[i, j] += 2.0

    # Conflict cluster: TENSION CONFLICT ARGUMENT FRUSTRATION WITHDRAWAL
    for i in [5, 6, 7, 8, 13]:
        for j in [5, 6, 7, 8, 13]:
            A[i, j] += 2.0

    # Cross-cluster repair & escalation
    A[6,  9] += 3.0   # CONFLICT → RECONCILIATION
    A[7,  9] += 3.0   # ARGUMENT → RECONCILIATION
    A[8,  9] += 2.0   # WITHDRAWAL → RECONCILIATION
    A[9,  1] += 3.0   # RECONCILIATION → WARMTH
    A[9, 14] += 2.0   # RECONCILIATION → AGREEMENT
    A[5,  6] += 3.0   # TENSION → CONFLICT
    A[13, 6] += 2.0   # FRUSTRATION → CONFLICT
    A[13, 8] += 2.0   # FRUSTRATION → WITHDRAWAL
    A[10, 3] += 3.0   # CURIOSITY → HELP_REQUEST
    A[11, 6] += 1.5   # ASSERTIVENESS → CONFLICT
    A[11,14] += 2.0   # ASSERTIVENESS → AGREEMENT

    # Self-loops (state inertia)
    for i in range(N_STATES):
        A[i, i] += 3.0

    A /= A.sum(axis=1, keepdims=True)
    return A


def _label_sequence(acts_raw: list, emotions: list) -> list:
    """Map a DailyDialog dialog's (act, emotion) pairs to intent state indices."""
    states       = []
    neg_run      = 0
    for act_raw, emot_raw in zip(acts_raw, emotions):
        act  = max(0, int(act_raw) - 1) % N_ACTS   # DailyDialog uses 1-indexed acts
        emot = int(emot_raw) % N_EMOT
        # Promote commissive-after-anger to RECONCILIATION(9) if there was a conflict run
        if act == 3 and emot in (1, 2) and neg_run >= 2:
            state = 9
        else:
            state = _HEURISTIC_MAP[act][emot]
        states.append(state)
        neg_run = (neg_run + 1) if emot in (1, 2) else 0
    return states


def _save(path: Path, start: np.ndarray, trans: np.ndarray, emit: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"start_prob": start, "transmat": trans, "emissionprob": emit}, f)
    logger.info(f"CDM HMM saved → {path}")


def _train_on_dailydialog(model_path: Path) -> bool:
    """Download DailyDialog, train CategoricalHMM, save pkl. Returns True on success."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        logger.warning(f"Missing dependency ({e}) — using fallback model.")
        return False

    try:
        logger.info("Downloading DailyDialog from HuggingFace…")
        # Try the canonical name first (Parquet version if available), then the org-scoped name.
        # trust_remote_code is intentionally omitted — loading scripts are no longer supported.
        dd = None
        for ds_name in ("daily_dialog", "li2017dailydialog/daily_dialog"):
            try:
                dd = load_dataset(ds_name)
                break
            except Exception:
                continue
        if dd is None:
            raise RuntimeError("All DailyDialog dataset names failed.")
        logger.info(f"DailyDialog: {len(dd['train'])} train / {len(dd['validation'])} val dialogs")
    except Exception as e:
        logger.warning(f"DailyDialog download failed: {e} — using fallback model.")
        return False

    obs_seqs   = []
    state_seqs = []
    for split in ("train", "validation"):
        for dialog in dd[split]:
            try:
                da = dialog["dialog_act"]
                acts     = da["act_type"] if isinstance(da, dict) else dialog.get("act", [])
                emotions = dialog["emotion"]
            except (KeyError, TypeError):
                continue
            if not acts or len(acts) != len(emotions) or len(acts) < 2:
                continue
            obs_seq   = [_obs(max(0, int(a) - 1) % N_ACTS, int(e) % N_EMOT)
                         for a, e in zip(acts, emotions)]
            state_seq = _label_sequence(acts, emotions)
            obs_seqs.append(np.array(obs_seq, dtype=int))
            state_seqs.append(state_seq)

    if not obs_seqs:
        logger.warning("No valid DailyDialog sequences — using fallback.")
        return False

    logger.info(f"Building transition counts from {len(state_seqs)} sequences…")
    A_counts     = np.ones((N_STATES, N_STATES), dtype=np.float64)
    start_counts = np.ones(N_STATES,             dtype=np.float64)
    for seq in state_seqs:
        start_counts[seq[0]] += 1.0
        for i in range(len(seq) - 1):
            A_counts[seq[i], seq[i + 1]] += 1.0

    transmat_init  = A_counts     / A_counts.sum(axis=1, keepdims=True)
    startprob_init = start_counts / start_counts.sum()

    logger.info(f"Calculating supervised emission counts from {sum(len(s) for s in obs_seqs)} utterances…")
    E_counts = np.ones((N_STATES, N_OBS), dtype=np.float64) # Laplace smoothing
    for seq, obs in zip(state_seqs, obs_seqs):
        for s, o in zip(seq, obs):
            E_counts[s, o] += 1.0
    emissionprob = E_counts / E_counts.sum(axis=1, keepdims=True)

    _save(model_path, startprob_init, transmat_init, emissionprob)
    logger.info("Saved supervised CategoricalHMM parameters without running EM.")
    return True


def ensure_model(model_path: Path = None) -> None:
    """
    Called at service startup.  No-op if pkl already exists.
    Falls back to a topology-derived transition matrix if DailyDialog is unavailable.
    """
    if model_path is None:
        model_path = _DEFAULT_MODEL_PATH

    if model_path.exists():
        logger.info(f"CDM HMM model found at {model_path} — skipping training.")
        return

    logger.info("CDM HMM model not found — starting training…")
    if not _train_on_dailydialog(model_path):
        logger.info("Saving topology-derived fallback HMM…")
        _save(
            model_path,
            start=np.ones(N_STATES, dtype=np.float64) / N_STATES,
            trans=_build_fallback_transmat(),
            emit=_build_emission_prior(),
        )
    logger.info("CDM HMM ready.")
