"""
trainer/data_fetcher.py — Fetch verified live training samples from PostgreSQL.

Uses a LAG() window query to capture sequential conversation context alongside
each verified emotion label, giving the trainer real context-aware training data.
"""
from sqlalchemy import create_engine, text

from shared.utils.logger import get_logger
from .config import DATABASE_URL, MAX_SAMPLES
from .analyzers import _vader, _run
from .preprocessor import build_fv

logger = get_logger("trainer")


def fetch_live_data(vader, bert, goe, emoji_scorer):
    """
    Fetch verified samples from PostgreSQL (emotion_analysis joined with messages).
    Returns (X, y) lists of feature vectors and labels.

    emoji_scorer fills the EmojiNet block from each row's text — passing {}
    here would teach the meta-learner that the emoji features are zero for
    live data (which then gets 3× weight), corrupting the block at runtime.
    """
    X, y = [], []
    try:
        engine = create_engine(DATABASE_URL)
        with engine.begin() as conn:
            # Ensure the ground_truth column exists
            try:
                conn.execute(text(
                    "ALTER TABLE emotion_analysis ADD COLUMN IF NOT EXISTS "
                    "ground_truth_emotion VARCHAR(50);"
                ))
            except Exception as e:
                logger.debug(f"  [SQL] Schema repair concurrent lock: {e}")

            query = text(f"""
                WITH ranked_messages AS (
                    SELECT m.message_id, m.text, a.ground_truth_emotion,
                           m.conversation_id, m.timestamp,
                           LAG(a.ground_truth_emotion)
                               OVER (PARTITION BY m.conversation_id ORDER BY m.timestamp)
                               AS prev_emo
                    FROM emotion_analysis a
                    JOIN messages m ON a.message_id = m.message_id
                    WHERE a.is_verified = TRUE
                )
                SELECT text, ground_truth_emotion, prev_emo
                FROM ranked_messages
                ORDER BY timestamp DESC
                LIMIT {MAX_SAMPLES}
            """)
            rows = conn.execute(query).fetchall()

            if not rows:
                return [], []

            logger.info(f"  [SQL] Found {len(rows)} verified live samples with context tracking.")

            ema = 0.0
            alpha = 0.35
            for text_content, label, prev_emo in rows:
                vs = {f"vader_{k}": v for k, v in _vader(vader, text_content).items()}
                bs = _run(bert, text_content)
                gs = _run(goe,  text_content)
                es = emoji_scorer.analyze(text_content)
                context = {"avg_valence": ema, "prev_emotion": prev_emo or "neutral"}
                fv = build_fv(vs, bs, gs, es, context=context)
                X.append(fv)
                y.append(label)
                # Update EMA for next sample (matches inference-time aggregation_service)
                ema = alpha * vs.get("vader_compound", 0.0) + (1.0 - alpha) * ema

        return X, y
    except Exception as e:
        logger.error(f"  [SQL] Failed to fetch live data: {e}")
        return [], []
