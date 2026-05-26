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


def _emojinet(goe_model, text: str) -> dict:
    try:
        from collections import Counter
        import emoji as emoji_lib
        from shared.constants import EMOTION_LABELS
        occurrences = emoji_lib.emoji_list(text)
        if not occurrences:
            return {}
        counts = Counter(e["emoji"] for e in occurrences)
        total = sum(counts.values())
        weighted = {}
        for ch, cnt in counts.items():
            desc = emoji_lib.demojize(ch).strip(":").replace("_", " ")
            if not desc:
                continue
            result = _run(goe_model, desc)
            w = cnt / total
            for emo, sc in result.items():
                weighted[emo] = weighted.get(emo, 0.0) + sc * w
        return weighted
    except Exception:
        return {}
