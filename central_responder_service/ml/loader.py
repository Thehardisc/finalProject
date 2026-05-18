"""
ml/loader.py — Load and validate the trained meta-learner model from disk.
"""
import os
import json
import pickle
import numpy as np
from typing import Optional

from shared.utils.logger import get_logger
from shared.constants import FEATURE_DIM

logger = get_logger("meta_learner")

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl")
DEFAULT_META_PATH  = DEFAULT_MODEL_PATH.replace(".pkl", "_meta.json")


def load_meta_learner(model_path: str = DEFAULT_MODEL_PATH) -> Optional[object]:
    """
    Attempt to load the trained sklearn Pipeline from disk.
    Returns the model or None — never raises.
    """
    try:
        if not os.path.exists(model_path):
            logger.warning(f"No model file at '{model_path}'. Fallback mode will be used.")
            return None

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        if not hasattr(model, 'predict_proba'):
            logger.warning("Loaded object is not a valid sklearn Pipeline. Fallback mode.")
            return None

        # Dimension sanity check
        try:
            model.predict(np.zeros((1, FEATURE_DIM)))
        except Exception as e:
            logger.warning(f"Model incompatible with {FEATURE_DIM}D features: {e}. Fallback mode.")
            return None

        _log_metadata()
        logger.info(f"Meta-learner loaded successfully from '{model_path}'.")
        return model

    except (pickle.UnpicklingError, EOFError, AttributeError) as e:
        logger.warning(f"Failed to load model (corrupt?): {e}. Fallback mode.")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error loading model: {e}. Fallback mode.")
        return None


def _log_metadata():
    """Log training metadata if the meta JSON file exists."""
    try:
        if os.path.exists(DEFAULT_META_PATH):
            with open(DEFAULT_META_PATH, 'r') as f:
                meta = json.load(f)
            logger.info(f"   Trained at      : {meta.get('trained_at', 'unknown')}")
            logger.info(f"   Training samples: {meta.get('training_samples', '?')}")
            logger.info(f"   Val accuracy    : {meta.get('validation_accuracy', '?')}")
            logger.info(f"   Test accuracy   : {meta.get('test_accuracy', '?')}")
    except Exception:
        pass
