"""
trainer/analyzers.py — Transient AI model loaders and per-model inference helpers.

These are loaded into RAM once per training cycle, used to build feature vectors,
then explicitly deleted and garbage-collected to free memory.
"""
from shared.utils.logger import get_logger

logger = get_logger("trainer")


def _get_analyzers(device):
    """Load VADER, BERT, and GoEmotions pipelines into RAM."""
    logger.info("Loading analyzers transiently into RAM...")
    import torch
    torch.set_num_threads(1)

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from transformers import pipeline as hf_pipeline

    logger.info("  [1/3] Loading VADER Sentiment...")
    vader = SentimentIntensityAnalyzer()

    logger.info("  [2/3] Loading BERT (j-hartmann/emotion-english-distilroberta-base)...")
    logger.info("        (This may take 1-2 minutes on first run to download ~330MB)")
    bert = hf_pipeline("text-classification",
                        model="j-hartmann/emotion-english-distilroberta-base",
                        return_all_scores=True,
                        device=device)

    logger.info("  [3/3] Loading GoEmotions (SamLowe/roberta-base-go_emotions)...")
    logger.info("        (This may take 1-2 minutes on first run to download ~500MB)")
    goe = hf_pipeline("text-classification",
                       model="SamLowe/roberta-base-go_emotions",
                       return_all_scores=True,
                       device=device)

    logger.info("✅ All AI Analyzers fully loaded into memory.")
    return vader, bert, goe


def _vader(v, text: str) -> dict:
    s = v.polarity_scores(text)
    return {k: s[k] for k in ['neg', 'neu', 'pos', 'compound']}


def _run(model, text: str) -> dict:
    try:
        return {r['label']: r['score'] for r in model(text[:512])[0]}
    except Exception:
        return {}


def _emojinet(text: str) -> dict:
    try:
        import emoji as emoji_lib
        from shared.constants import EMOJI_EMOTION_DB
        found = emoji_lib.distinct_emoji_list(text)
        scores, count = {}, 0
        for ch in found:
            entry = EMOJI_EMOTION_DB.get(ch) or EMOJI_EMOTION_DB.get(ch.replace('\ufe0f', ''))
            if entry:
                for e, sc in entry.get("emotions", {}).items():
                    scores[e] = scores.get(e, 0.0) + sc
                count += 1
        return {k: v / count for k, v in scores.items()} if count else {}
    except Exception:
        return {}
