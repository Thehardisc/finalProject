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
# VADER(0-3), BERT(4-10), GoE(11-38), Emoji(39-66), Context(67-95), Derived(96-102)
BLOCK_SLICES = {
    "VADER":      slice(0,  4),
    "BERT":       slice(4,  11),
    "GoEmotions": slice(11, 39),
    "EmojiNet":   slice(39, 67),
    "Context":    slice(67, 96),
    "Derived":    slice(96, 103),
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

        # The trainer ships a VotingClassifier (LR + HGB + RF) which exposes neither
        # coef_ nor feature_importances_ directly — reach into a sub-estimator. Prefer
        # the LR member's per-class coef_ (linear, attributable); fall back to any
        # estimator that does expose importances; else attribute the bare clf.
        sub = getattr(clf, 'named_estimators_', {}) or {}
        est = sub.get('lr') or sub.get('rf') or clf

        if hasattr(est, 'coef_'):
            classes = list(est.classes_)
            if predicted_emotion not in classes:
                return {}
            weights = est.coef_[classes.index(predicted_emotion)]
        elif hasattr(est, 'feature_importances_'):
            weights = est.feature_importances_      # global (not per-class)
        else:
            return {}

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
