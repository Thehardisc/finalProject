import json
import os
import sys
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text

from shared.constants import EMOTION_LABELS, FEATURE_DIM, CDM_CTX_DIM, PRIOR_DIM
from shared.utils.logger import get_logger
from meta_learner import build_feature_vector
from trainer.utils import _vader, _run
from trainer.data.synthetic import build_synthetic_context_vector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'goemotions_service'))
try:
    from vad_lexicon import compute_vad as _compute_vad
except ImportError:
    def _compute_vad(text):  # noqa: F811
        return {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}

logger = get_logger("trainer")

DB_USER     = os.getenv("POSTGRES_USER",     "user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_NAME     = os.getenv("POSTGRES_DB",       "emotion_db")
DB_HOST     = os.getenv("DB_HOST",           "db")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"

MAX_EMPATHETIC_SAMPLES = int(os.environ.get("MAX_EMPATHETIC_SAMPLES", 25_000))

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/models/meta_weights.pkl"))

RELABELED_DATA_PATH = Path(os.environ.get(
    "RELABELED_DATA_PATH",
    "/app/training_data/conversations_relabeled.jsonl",
))


def fetch_live_data(vader, bert, goe) -> tuple:
    X, y = [], []
    try:
        engine = create_engine(DATABASE_URL)
        with engine.begin() as conn:
            try:
                conn.execute(text(
                    "ALTER TABLE emotion_analysis "
                    "ADD COLUMN IF NOT EXISTS ground_truth_emotion VARCHAR(50);"
                ))
            except Exception as e:
                logger.debug(f"  [SQL] Schema repair: {e}")

            rows = conn.execute(text(f"""
                WITH ranked AS (
                    SELECT m.message_id, m.text, a.ground_truth_emotion,
                           m.conversation_id, m.timestamp
                    FROM emotion_analysis a
                    JOIN messages m ON a.message_id = m.message_id
                    WHERE a.is_verified = TRUE
                )
                SELECT text, ground_truth_emotion
                FROM ranked
                ORDER BY timestamp DESC
                LIMIT {MAX_EMPATHETIC_SAMPLES}
            """)).fetchall()

        if not rows:
            return [], []

        _valid_labels = set(EMOTION_LABELS)
        # Cap neutral to 3× the expected average class count to guard against
        # contamination from pipeline periods where "neutral" was the default output.
        _avg_expected = max(len(rows) // len(EMOTION_LABELS), 10)
        _neutral_cap  = _avg_expected * 3
        _neutral_seen = 0
        skipped = 0
        logger.info(f"  [SQL] Found {len(rows)} verified live samples. neutral_cap={_neutral_cap}.")
        for text_content, label in rows:
            if label not in _valid_labels:
                skipped += 1
                continue
            if label == "neutral":
                if _neutral_seen >= _neutral_cap:
                    skipped += 1
                    continue
                _neutral_seen += 1
            vs  = {f"vader_{k}": v for k, v in _vader(vader, text_content).items()}
            bs  = _run(bert, text_content)
            gs  = _run(goe,  text_content)
            ctx = build_synthetic_context_vector(mode="cold")
            fv  = build_feature_vector(
                {"vader": vs, "basic_bert": bs, "go_emotions": gs},
                context_vector=ctx[:CDM_CTX_DIM],
                trajectory_prior=ctx[CDM_CTX_DIM:CDM_CTX_DIM + PRIOR_DIM],
                vad_scores=_compute_vad(text_content),
            )
            X.append(fv.flatten())
            y.append(label)
        if skipped:
            logger.warning(f"  [SQL] Skipped {skipped} rows (invalid label or neutral cap).")

        return X, y
    except Exception as e:
        logger.error(f"  [SQL] Failed to fetch live data: {e}")
        return [], []


def load_relabeled_conversations() -> tuple:
    if not RELABELED_DATA_PATH.exists():
        logger.info(f"  [Relabeled] {RELABELED_DATA_PATH} not found — skipping.")
        return np.empty((0, FEATURE_DIM), dtype=np.float32), []

    try:
        conversations = [
            json.loads(line)
            for line in RELABELED_DATA_PATH.read_text().splitlines()
            if line.strip()
        ]
    except Exception as e:
        logger.warning(f"  [Relabeled] Failed to read file: {e} — skipping.")
        return np.empty((0, FEATURE_DIM), dtype=np.float32), []

    features, labels = [], []

    for conv in conversations:
        chunks   = conv.get("relabeled_chunks", [])
        messages = conv.get("messages", [])

        for chunk in chunks:
            emotions_dict = chunk.get("emotions", {})
            if not emotions_dict:
                continue

            valid = {k: v for k, v in emotions_dict.items() if k in EMOTION_LABELS}
            if not valid:
                continue
            chunk_label = max(valid, key=valid.get)

            for idx in chunk.get("message_indices", []):
                if idx >= len(messages):
                    continue
                stages = messages[idx].get("pipeline", {}).get("stages", {})
                if not stages:
                    continue

                model_outputs = {
                    "vader":       stages.get("vader",       {}),
                    "basic_bert":  stages.get("bert",        {}),
                    "go_emotions": stages.get("goemotions",  {}),
                }

                ctx = build_synthetic_context_vector(label=chunk_label, mode="train")
                text_content = messages[idx].get("text", "")
                fv  = build_feature_vector(
                    model_outputs,
                    context_vector=ctx[:CDM_CTX_DIM],
                    trajectory_prior=ctx[CDM_CTX_DIM:CDM_CTX_DIM + PRIOR_DIM],
                    vad_scores=_compute_vad(text_content),
                )
                features.append(fv.flatten())
                labels.append(chunk_label)

    logger.info(
        f"  [Relabeled] Loaded {len(features)} samples "
        f"from {len(conversations)} conversations."
    )
    return (
        np.array(features, dtype=np.float32) if features
        else np.empty((0, FEATURE_DIM), dtype=np.float32),
        labels,
    )
