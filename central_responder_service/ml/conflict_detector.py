"""
ml/conflict_detector.py — Heuristic sarcasm and emotional conflict detection.

Analyses the raw feature vector for signature patterns that indicate
sarcasm, passive-aggression, or semantic/visual dissonance.
"""
import numpy as np
from typing import Tuple, Optional

from shared.utils.logger import get_logger

logger = get_logger("meta_learner")


def detect_emotional_conflicts(vec: np.ndarray) -> Tuple[float, Optional[str]]:
    """
    Heuristic layer to detect 'Sentiment Flipping' (Sarcasm / Slang).

    Feature vector offsets (must match shared/constants.py):
      VADER: 0-3 | BERT: 4-10 | GoE: 11-38 | Emoji: 39-66 | Context: 67-95

    Returns:
        (sarcasm_score [0.0-1.0], conflict_description [str or None])
    """
    try:
        # Flatten to 1D so indexing works whether caller passes (103,) or (1,103)
        if vec.ndim > 1:
            vec = vec[0]

        v_pos = vec[2]
        v_neg = vec[0]
        v_cmp = vec[3]

        bert_joy   = vec[7]
        bert_anger = vec[4]

        # Emoji block — annoyance: 3, disapproval: 10, disgust: 11
        emo_annoyance   = vec[39 + 3]
        emo_disapproval = vec[39 + 10]
        emo_disgust     = vec[39 + 11]

        neg_emo_signal  = max(emo_annoyance, emo_disapproval, emo_disgust)
        pos_text_signal = (v_pos + bert_joy) / 2

        sarcasm_score = 0.0
        conflict_desc = None

        # Pattern 1: Positive text + eye-roll / negative emoji
        if pos_text_signal > 0.6 and neg_emo_signal > 0.4:
            sarcasm_score = min(min(pos_text_signal, neg_emo_signal) * 1.2, 1.0)
            conflict_desc = ("Cognitive Dissonance: High-fidelity positive text "
                             "paired with dismissive visual cues.")

        # Pattern 2: Extreme positive compound + emoji flip
        elif v_cmp > 0.8 and neg_emo_signal > 0.2:
            sarcasm_score = min(0.5 + neg_emo_signal, 1.0)
            conflict_desc = "Sarcasm detected: Semantic praise contradicts visual frustration."

        # Pattern 3: Passive-aggressive (formal neutral text + emoji tension)
        elif vec[8] > 0.7 and (emo_annoyance > 0.1 or emo_disapproval > 0.1):
            sarcasm_score = 0.4
            conflict_desc = ("Passive-aggression suspected: Formal 'Neutral' text "
                             "with underlying emoji tension.")

        return min(sarcasm_score, 1.0), conflict_desc

    except Exception:
        return 0.0, None
