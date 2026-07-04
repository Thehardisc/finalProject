"""ml/conflict_detector.py — Heuristic sarcasm and emotional conflict detection."""
import numpy as np
from typing import Tuple, Optional

from shared.utils.logger import get_logger

logger = get_logger("meta_learner")


def detect_emotional_conflicts(vec: np.ndarray) -> Tuple[float, Optional[str]]:
    """Heuristic layer to detect 'Sentiment Flipping' (Sarcasm / Slang)."""
    try:
        if vec.ndim > 1:
            vec = vec[0]

        v_pos = vec[2]
        v_neg = vec[0]
        v_cmp = vec[3]

        bert_joy     = vec[7]
        bert_anger   = vec[4]
        bert_neutral = vec[8]

        pos_text_signal = (v_pos + bert_joy) / 2

        sarcasm_score = 0.0
        conflict_desc = None

        if pos_text_signal > 0.6 and v_neg > 0.4:
            sarcasm_score = min(min(pos_text_signal, v_neg) * 1.2, 1.0)
            conflict_desc = ("Cognitive Dissonance: Positive surface text "
                             "paired with high negativity signal.")

        elif v_cmp > 0.8 and bert_anger > 0.3:
            sarcasm_score = min(0.5 + bert_anger, 1.0)
            conflict_desc = "Sarcasm detected: Semantic praise contradicts underlying anger signal."

        elif bert_neutral > 0.7 and v_neg > 0.25:
            sarcasm_score = 0.4
            conflict_desc = ("Passive-aggression suspected: Formal neutral tone "
                             "with underlying negativity.")

        return min(sarcasm_score, 1.0), conflict_desc

    except Exception:
        return 0.0, None
