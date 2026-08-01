import asyncio
import sys
import os
import concurrent.futures

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from transformers import pipeline
from shared.nlp_worker import run_nlp_worker
from shared.utils.logger import get_logger
from shared.constants import EMOTION_LABELS
from vad_lexicon import compute_vad

_INFERENCE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="goe_inf")

logger = get_logger("goemotions_service")

_PROTO_SENTENCES = [
    "I truly admire and respect this person, it is deeply impressive",
    "This is hilarious and funny, I cannot stop laughing",
    "I am furious and extremely angry, this enrages me",
    "This is so annoying and irritating to deal with",
    "I approve of this decision and think it is the right choice",
    "I care deeply about this and want to help and support",
    "I am confused and do not understand what is happening here",
    "I am curious and want to explore and learn more about this",
    "I want this so badly and desire it deeply",
    "I am deeply disappointed and let down by what happened",
    "I strongly disapprove of this and think it is wrong",
    "This disgusts me and makes me feel revolted and sick",
    "I feel embarrassed and ashamed about what occurred",
    "I am incredibly excited and thrilled, this is amazing",
    "I am terrified and scared, this fills me with fear and dread",
    "I am very grateful and thankful for this kindness",
    "I am overwhelmed with grief and deep sorrow and loss",
    "I feel joyful and happy, this fills me with pure happiness",
    "I feel deep love and affection for this person",
    "I feel nervous and anxious and worried about what might happen",
    "I feel optimistic and hopeful that things will work out well",
    "I feel proud of this great achievement and accomplishment",
    "I just realized something important that I had not noticed before",
    "I feel such relief that this difficult situation resolved well",
    "I feel deep remorse and regret for what I said or did",
    "I feel very sad and melancholy, my heart is heavy with sorrow",
    "I am completely surprised and shocked by this unexpected event",
    "This is a normal everyday situation with no particular emotional response",
]


class GoEmotionsAnalyzer:
    def __init__(self):
        device = -1
        if torch.cuda.is_available():
            device = 0
            logger.info("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Metal (MPS) GPU")
        else:
            logger.info("Using CPU")

        _finetuned_dir = "/app/models/samlowe_finetuned"
        _primary_model = _finetuned_dir if os.path.isdir(_finetuned_dir) else "SamLowe/roberta-base-go_emotions"
        logger.info(f"Loading GoEmotions primary model ({_primary_model})...")
        self.clf_primary = pipeline(
            "text-classification",
            model=_primary_model,
            top_k=None,
            device=device,
        )
        logger.info("Loading GoEmotions secondary model (bhadresh-savani/bert-base-go-emotion)...")
        self.clf_secondary = pipeline(
            "text-classification",
            model="bhadresh-savani/bert-base-go-emotion",
            top_k=None,
            device=device,
        )
        logger.info("Loading sentence transformer for semantic NLI (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer
        self._sent_model = SentenceTransformer("all-MiniLM-L6-v2")
        proto_embs = self._sent_model.encode(_PROTO_SENTENCES, convert_to_numpy=True,
                                             show_progress_bar=False)
        norms = np.linalg.norm(proto_embs, axis=1, keepdims=True) + 1e-8
        self._proto_norm = proto_embs / norms
        logger.info("All GoEmotions models + semantic NLI loaded — 3-model blend active.")

    def _semantic_scores(self, text: str) -> dict:
        emb = self._sent_model.encode([text], convert_to_numpy=True,
                                      show_progress_bar=False)[0]
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        sims = self._proto_norm @ emb_norm
        exp_s = np.exp((sims - sims.max()) * 5.0)
        probs = exp_s / exp_s.sum()
        return {e: float(probs[i]) for i, e in enumerate(EMOTION_LABELS)}

    def analyze(self, text: str) -> dict:
        if len(text) > 512:
            text = text[:512]
        f_primary   = _INFERENCE_POOL.submit(lambda: {r["label"]: r["score"] for r in self.clf_primary(text)[0]})
        f_secondary = _INFERENCE_POOL.submit(lambda: {r["label"]: r["score"] for r in self.clf_secondary(text)[0]})
        f_semantic  = _INFERENCE_POOL.submit(self._semantic_scores, text)
        primary   = f_primary.result()
        secondary = f_secondary.result()
        semantic  = f_semantic.result()
        all_labels = set(primary) | set(secondary) | set(semantic)
        scores = {
            lbl: 0.55 * primary.get(lbl, 0.0) + 0.25 * secondary.get(lbl, 0.0) + 0.20 * semantic.get(lbl, 0.0)
            for lbl in all_labels
        }
        vad = compute_vad(text)
        scores["vad_valence"]   = vad["valence"]
        scores["vad_arousal"]   = vad["arousal"]
        scores["vad_dominance"] = vad["dominance"]
        scores["goe_confidence"] = max(
            scores.get(lbl, 0.0) for lbl in EMOTION_LABELS
        )
        return scores


def _format_stats(data, scores, elapsed):
    top3 = sorted(((k, v) for k, v in scores.items() if k in EMOTION_LABELS), key=lambda x: x[1], reverse=True)[:3]
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
