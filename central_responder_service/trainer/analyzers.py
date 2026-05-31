"""
trainer/analyzers.py — Transient AI model loaders and per-model inference helpers.

These are loaded into RAM once per training cycle, used to build feature vectors,
then explicitly deleted and garbage-collected to free memory.
"""
from shared.utils.logger import get_logger
from shared.emoji_scoring import EmojiSemanticScorer

logger = get_logger("trainer")


def _get_analyzers(device):
    """
    Load VADER, BERT, GoEmotions, and the emoji scorer into RAM.

    Returns (vader, bert, goe, emoji_scorer). The emoji scorer reuses the
    GoEmotions pipeline's encoder, so this adds only the 28 anchor-phrase
    pre-compute (a few hundred ms) on top of model loading.

    On failure, logs a structured `analyzer_load_failed` event naming the stage
    that failed and re-raises — the trainer loop (runner._loop) catches it, marks
    the cycle errored, and retries next interval (the daemon stays alive).
    """
    logger.info("Loading analyzers transiently into RAM...")
    stage = "torch"
    try:
        import torch
        torch.set_num_threads(1)

        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        from transformers import pipeline as hf_pipeline

        stage = "vader"
        logger.info("  [1/3] Loading VADER Sentiment...")
        vader = SentimentIntensityAnalyzer()

        stage = "bert"
        logger.info("  [2/3] Loading BERT (j-hartmann/emotion-english-distilroberta-base)...")
        logger.info("        (This may take 1-2 minutes on first run to download ~330MB)")
        bert = hf_pipeline("text-classification",
                            model="j-hartmann/emotion-english-distilroberta-base",
                            return_all_scores=True,
                            device=device)

        stage = "goemotions"
        logger.info("  [3/3] Loading GoEmotions (SamLowe/roberta-base-go_emotions)...")
        logger.info("        (This may take 1-2 minutes on first run to download ~500MB)")
        goe = hf_pipeline("text-classification",
                           model="SamLowe/roberta-base-go_emotions",
                           return_all_scores=True,
                           device=device)

        stage = "emoji_scorer"
        logger.info("  Pre-computing emoji anchor embeddings...")
        emoji_scorer = EmojiSemanticScorer(goe)

        logger.info("✅ All AI Analyzers fully loaded into memory.")
        return vader, bert, goe, emoji_scorer

    except Exception as e:
        logger.error(
            f"analyzer_load_failed stage={stage} error={type(e).__name__}",
            extra={"extra_data": {
                "event":       "analyzer_load_failed",
                "stage":       stage,
                "error_class": type(e).__name__,
                "error":       str(e),
            }},
        )
        raise


def _vader(v, text: str) -> dict:
    s = v.polarity_scores(text)
    return {k: s[k] for k in ['neg', 'neu', 'pos', 'compound']}


def _run(model, text: str) -> dict:
    try:
        return {r['label']: r['score'] for r in model(text[:512])[0]}
    except Exception:
        return {}
