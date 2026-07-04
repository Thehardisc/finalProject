import asyncio
import sys
import os
import concurrent.futures

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.nlp_worker import run_nlp_worker
from shared.utils.logger import get_logger
from transformers import pipeline

_INFERENCE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="bert_inf")

logger = get_logger("bert_service")


class BasicBertAnalyzer:
    _SEC_MAP = {
        "anger": "anger", "fear": "fear", "joy": "joy",
        "love": "joy", "sadness": "sadness", "surprise": "surprise",
    }

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
        logger.info("Loading primary BERT model (j-hartmann/emotion-english-distilroberta-base)...")
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=device,
        )
        logger.info("Loading secondary BERT model (bhadresh-savani/distilbert-base-uncased-emotion)...")
        self.clf_secondary = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            top_k=None,
            device=device,
        )
        logger.info("Both BERT models loaded — ensemble mode active.")

    def analyze(self, text: str) -> dict:
        if len(text) > 512:
            text = text[:512]
        f_primary   = _INFERENCE_POOL.submit(lambda: {r["label"]: r["score"] for r in self.classifier(text)[0]})
        f_secondary = _INFERENCE_POOL.submit(lambda: {r["label"]: r["score"] for r in self.clf_secondary(text)[0]})
        primary       = f_primary.result()
        secondary_raw = f_secondary.result()
        sec = {lbl: 0.0 for lbl in primary}
        for sec_lbl, score in secondary_raw.items():
            target = self._SEC_MAP.get(sec_lbl)
            if target and target in sec:
                sec[target] += score
        sec_total = sum(sec.values())
        if sec_total > 1e-6:
            sec = {k: v / sec_total for k, v in sec.items()}
        blended = {lbl: 0.70 * primary.get(lbl, 0.0) + 0.30 * sec.get(lbl, 0.0) for lbl in primary}
        total = sum(blended.values())
        if total > 1e-6:
            blended = {k: v / total for k, v in blended.items()}
        return blended


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
