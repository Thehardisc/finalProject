"""
ml/impact_calculator.py — Per-block feature importance calculation.

Computes the relative contribution of each model block (VADER, BERT, GoEmotions,
EmojiNet, Context) to the final prediction for use in the frontend Logic Map.

Supports both LogisticRegression (coef_) and RandomForest (feature_importances_).
"""
import numpy as np
from typing import Optional

from shared.utils.logger import get_logger

logger = get_logger("meta_learner")

# Feature block offsets — must match build_feature_vector() in predictor.py
# VADER(0-3), BERT(4-10), GoE(11-38), Emoji(39-66), CDM Context(67-110), Derived(111-117)
BLOCK_SLICES = {
    "VADER":      slice(0,   4),
    "BERT":       slice(4,   11),
    "GoEmotions": slice(11,  39),
    "EmojiNet":   slice(39,  67),
    "Context":    slice(67,  111),
}


def calculate_feature_impacts(model, feature_vector: np.ndarray,
                               predicted_emotion: str) -> dict:
    """
    Calculate the contribution of each model block to the final prediction.

    Returns a dict of {block_name: relative_importance} normalised to [0, 1].
    Returns {} on any error.
    """
    try:
        scaler = model.named_steps['scaler']
        clf    = model.named_steps['clf']

        X_scaled = scaler.transform(feature_vector)[0]

        # ── Weight source: LR uses per-class coef_, RF uses global importances ──
        if hasattr(clf, 'feature_importances_'):
            weights = clf.feature_importances_
        else:
            classes = list(clf.classes_)
            if predicted_emotion not in classes:
                return {}
            class_idx = classes.index(predicted_emotion)
            weights = clf.coef_[class_idx]

        contributions = X_scaled * weights

        impacts = {
            name: float(np.sum(contributions[sl]))
            for name, sl in BLOCK_SLICES.items()
        }

        # Normalise to relative importance
        total = sum(abs(v) for v in impacts.values())
        if total > 0:
            return {k: round(v / total, 4) for k, v in impacts.items()}
        return impacts

    except Exception as e:
        logger.warning(f"Failed to calculate impacts: {e}")
        return {}
