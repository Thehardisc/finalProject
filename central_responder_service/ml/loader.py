import os
import json
import pickle
import hashlib
import numpy as np
from typing import Optional

from shared.utils.logger import get_logger
from shared.constants import FEATURE_DIM

logger = get_logger("meta_learner")

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl")
DEFAULT_META_PATH  = DEFAULT_MODEL_PATH.replace(".pkl", "_meta.json")


def load_meta_learner(model_path: str = DEFAULT_MODEL_PATH) -> Optional[object]:
    try:
        if not os.path.exists(model_path):
            logger.warning(f"No model file at '{model_path}'. Fallback mode will be used.")
            return None

        if not _verify_checksum(model_path):
            logger.warning("Model file failed SHA-256 integrity check. Fallback mode.")
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


def _verify_checksum(model_path: str) -> bool:
    """
    Check that a .sha256 sidecar file exists and matches the model file.
    If no sidecar exists, the check passes (first-ever load or pre-existing model).
    Returns True if OK, False if tampered.
    """
    checksum_path = model_path + ".sha256"
    if not os.path.exists(checksum_path):
        # No sidecar yet — first load, write one now so future loads are checked.
        try:
            digest = _sha256(model_path)
            with open(checksum_path, "w") as f:
                f.write(digest)
            logger.info(f"SHA-256 sidecar written for '{model_path}'.")
        except Exception as e:
            logger.warning(f"Could not write checksum sidecar: {e}")
        return True  # trust the first load

    try:
        expected = open(checksum_path).read().strip()
        actual   = _sha256(model_path)
        if actual != expected:
            logger.error(
                f"INTEGRITY VIOLATION: model hash mismatch!\n"
                f"  Expected: {expected}\n"
                f"  Actual:   {actual}"
            )
            return False
        return True
    except Exception as e:
        logger.warning(f"Checksum verification failed: {e}. Allowing load.")
        return True


def _sha256(path: str) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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
