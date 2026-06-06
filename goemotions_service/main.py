import asyncio
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import pipeline
from shared.nlp_worker import run_nlp_worker
from shared.utils.logger import get_logger

logger = get_logger("goemotions_service")


class GoEmotionsAnalyzer:
    def __init__(self):
        logger.info("Loading GoEmotions BERT model...")
        device = -1
        if torch.cuda.is_available():
            device = 0
            logger.info("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Metal (MPS) GPU")
        else:
            logger.info("Using CPU")
        self.classifier = pipeline(
            "text-classification",
            model="bhadresh-savani/bert-base-go-emotion",
            top_k=None,
            device=device,
        )
        logger.info("GoEmotions BERT model loaded.")

    def analyze(self, text: str) -> dict:
        if len(text) > 512:
            text = text[:512]
        return {r["label"]: r["score"] for r in self.classifier(text)[0]}


def _format_stats(data, scores, elapsed):
    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return {
        "Message ID": data.get("message_id", "N/A"),
        "Top Emotions": ", ".join(f"{e} ({s:.2%})" for e, s in top3),
        "Inference Time": f"{elapsed:.2f}ms",
    }


logger.info("Initializing GoEmotions...")
_analyzer = GoEmotionsAnalyzer()

if __name__ == "__main__":
    asyncio.run(run_nlp_worker(
        model_name="go_emotions",
        group_name="goemotions_group",
        consumer_name="goemotions_worker_1",
        analyzer=_analyzer,
        get_text=lambda d: d.get("processed_text_demojized", "") or d.get("processed_text", ""),
        format_stats=_format_stats,
        logger=logger,
    ))
