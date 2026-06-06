import asyncio
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.nlp_worker import run_nlp_worker
from shared.utils.logger import get_logger
from transformers import pipeline

logger = get_logger("bert_service")


class BasicBertAnalyzer:
    def __init__(self):
        import torch
        device = -1
        if torch.cuda.is_available():
            device = 0
            logger.info("BERT using CUDA GPU")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("BERT using Apple Metal (MPS) GPU")
        else:
            logger.info("BERT using CPU")
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=device,
        )

    def analyze(self, text: str) -> dict:
        if len(text) > 512:
            text = text[:512]
        return {r["label"]: r["score"] for r in self.classifier(text)[0]}


def _format_stats(data, scores, elapsed):
    top = max(scores.items(), key=lambda x: x[1])
    return {
        "Message ID": data.get("message_id", "N/A"),
        "Dominant": f"{top[0]} ({top[1]:.2%})",
        "Latency": f"{elapsed:.2f}ms",
    }


logger.info("Loading BERT model...")
_analyzer = BasicBertAnalyzer()
logger.info("BERT model loaded.")

if __name__ == "__main__":
    asyncio.run(run_nlp_worker(
        model_name="basic_bert",
        group_name="bert_group",
        consumer_name="bert_worker_1",
        analyzer=_analyzer,
        get_text=lambda d: d.get("processed_text_demojized", "") or d.get("processed_text", ""),
        format_stats=_format_stats,
        logger=logger,
    ))
