"""
trainer/data/db.py — Live database data fetching and relabeled conversation loading.
"""

import json
import os
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text

from shared.constants import EMOTION_LABELS, FEATURE_DIM, CDM_CTX_DIM
from shared.utils.logger import get_logger
from meta_learner import build_feature_vector
from trainer.utils import _vader, _run
from trainer.data.synthetic import build_synthetic_context_vector

logger = get_logger("trainer")

# ── Database ────────────────────────────────────────────────────────────────────
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
    """
    Fetch verified samples from PostgreSQL.
    Context is set to zeros (cold) — no conversation history available for SQL rows.
    """
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

        logger.info(f"  [SQL] Found {len(rows)} verified live samples.")
        for text_content, label in rows:
            vs  = {f"vader_{k}": v for k, v in _vader(vader, text_content).items()}
            bs  = _run(bert, text_content)
            gs  = _run(goe,  text_content)
            ctx = build_synthetic_context_vector(mode="cold")
            fv  = build_feature_vector(
                {"vader": vs, "basic_bert": bs, "go_emotions": gs},
                context_vector=ctx[:CDM_CTX_DIM],
                trajectory_prior=ctx[CDM_CTX_DIM:],
            )
            X.append(fv.flatten())
            y.append(label)

        return X, y
    except Exception as e:
        logger.error(f"  [SQL] Failed to fetch live data: {e}")
        return [], []


def load_relabeled_conversations() -> tuple:
    """
    Load re-labeled conversations produced by relabel.py.

    Key difference from GoEmotions training data:
      - Labels come from Claude's implicit emotion recognition (not GoEmotions predictions)
      - NLP features are the real pipeline outputs stored in conversations.jsonl
      - GoEmotions features in the vector may DISAGREE with the label →
        the model learns that GoEmotions can be wrong and context matters

    CDM context is synthetic (correlated with the Claude-assigned label, same
    augmentation as GoEmotions training) — real CDM vectors are not stored in
    the collected data at sufficient resolution to reconstruct the full 40-dim block.
    """
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

            # Dominant emotion = argmax of Claude's 28-dim scores
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

                # Map stored stage keys → build_feature_vector format
                model_outputs = {
                    "vader":       stages.get("vader",       {}),
                    "basic_bert":  stages.get("bert",        {}),
                    "go_emotions": stages.get("goemotions",  {}),
                }

                ctx = build_synthetic_context_vector(label=chunk_label, mode="train")
                fv  = build_feature_vector(
                    model_outputs,
                    context_vector=ctx[:CDM_CTX_DIM],
                    trajectory_prior=ctx[CDM_CTX_DIM:],
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
