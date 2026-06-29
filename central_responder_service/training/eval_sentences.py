"""
eval_sentences.py — Offline sentence-battery evaluation for the meta-learner.

Runs a fixed set of sentences through the real NLP analyzers + the trained
GatingNetworkWrapper pkl, printing per sentence:
  - GoEmotions top label + confidence + normalized entropy
  - BERT top label + confidence
  - raw meta-learner argmax (before NLP guards)
  - final guarded output (predict_with_meta_learner)
  - gate α [vader, bert, goe, ctx]
  - PASS/FAIL against an expected label set (cluster-level)

Usage (from project root):
    python central_responder_service/training/eval_sentences.py
    python central_responder_service/training/eval_sentences.py \
        --goe-model bhadresh-savani/bert-base-go-emotion --json baseline.json

The --goe-model flag exists because production (goemotions_service) and the
trainer historically used different GoEmotions checkpoints. Default is the
trainer's checkpoint (SamLowe), which is also the production model after the
2026-06 alignment fix.
"""
import argparse
import json
import math
import os
import sys

_HERE         = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SVC_ROOT     = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_PROJECT_ROOT, _SVC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from shared.constants import EMOTION_LABELS, BERT_LABELS
from meta_learner import (
    load_meta_learner, build_feature_vector, predict_with_meta_learner,
    GatingNetworkWrapper,
)

DEFAULT_MODEL_PATH = os.path.join(_SVC_ROOT, "models", "meta_weights.pkl")
DEFAULT_GOE_MODEL  = "SamLowe/roberta-base-go_emotions"
DEFAULT_BERT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# (sentence, set of acceptable final labels)
SENTENCE_BATTERY = [
    ("I'm absolutely furious right now",     {"anger", "annoyance"}),
    ("I am completely devastated",           {"sadness", "grief", "disappointment"}),
    ("I totally love this!",                 {"love", "joy", "excitement", "admiration"}),
    ("I hate this so much",                  {"anger", "annoyance", "disgust"}),
    ("This makes me so angry",               {"anger", "annoyance"}),
    ("I'm really happy today",               {"joy", "excitement"}),
    ("This is absolutely amazing",           {"admiration", "excitement", "approval", "joy"}),
    ("I'm utterly terrified",                {"fear", "nervousness"}),
    ("I'm completely disgusted by this",     {"disgust", "disapproval"}),
    ("What time is the meeting tomorrow?",   {"neutral", "curiosity"}),
    ("I cannot believe you did that",        {"surprise", "anger", "disapproval", "confusion"}),
    ("Thank you so much for your help",      {"gratitude", "joy", "admiration"}),
]


def _entropy_norm(scores: dict, keys: list) -> float:
    eps   = 1e-9
    probs = [max(scores.get(k, 0.0), 0.0) for k in keys]
    total = sum(probs) + eps
    probs = [p / total for p in probs]
    h = -sum(p * math.log(p + eps) for p in probs)
    return h / math.log(len(keys))


def _load_analyzers(bert_model: str, goe_model: str, device):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from transformers import pipeline as hf_pipeline
    vader = SentimentIntensityAnalyzer()
    bert  = hf_pipeline("text-classification", model=bert_model, top_k=None, device=device)
    goe   = hf_pipeline("text-classification", model=goe_model,  top_k=None, device=device)
    return vader, bert, goe


def evaluate(model, vader, bert, goe) -> list:
    results = []
    for text, expected in SENTENCE_BATTERY:
        v = vader.polarity_scores(text)
        vader_scores = {f"vader_{k}": v[k] for k in ("neg", "neu", "pos", "compound")}
        bert_scores  = {r["label"]: r["score"] for r in bert(text[:512])[0]}
        goe_scores   = {r["label"]: r["score"] for r in goe(text[:512])[0]}

        fv = build_feature_vector({
            "vader":       vader_scores,
            "basic_bert":  bert_scores,
            "go_emotions": goe_scores,
        })

        goe_top  = max(goe_scores,  key=goe_scores.get)
        bert_top = max(bert_scores, key=bert_scores.get)

        raw_pred, raw_conf = None, None
        gate = None
        if model is not None:
            proba    = model.predict_proba(fv)[0]
            raw_idx  = int(np.argmax(proba))
            raw_pred = str(model.classes_[raw_idx])
            raw_conf = float(proba[raw_idx])
            if isinstance(model, GatingNetworkWrapper):
                gate = [round(float(a), 4) for a in model.get_gate_weights(fv)[0]]

        final_pred, final_conf, _, _, _, _ = predict_with_meta_learner(model, fv)

        ok = final_pred in expected
        results.append({
            "text":        text,
            "expected":    sorted(expected),
            "goe_top":     goe_top,
            "goe_max":     round(goe_scores[goe_top], 4),
            "goe_entropy": round(_entropy_norm(goe_scores, EMOTION_LABELS), 4),
            "bert_top":    bert_top,
            "bert_max":    round(bert_scores[bert_top], 4),
            "raw_meta":    raw_pred,
            "raw_conf":    round(raw_conf, 4) if raw_conf is not None else None,
            "final":       final_pred,
            "final_conf":  round(float(final_conf), 4),
            "guard_fired": (raw_pred is not None and final_pred != raw_pred),
            "gate_alpha":  gate,
            "pass":        ok,
        })
    return results


def print_table(results: list) -> None:
    hdr = (f"{'PASS':<5} {'sentence':<38} {'GoE top':<18} {'BERT top':<14} "
           f"{'raw meta':<14} {'final':<16} {'gate α [v,b,g,c]':<30}")
    print(hdr)
    print("─" * len(hdr))
    for r in results:
        mark = "✓" if r["pass"] else "✗ FAIL"
        goe  = f"{r['goe_top']} {r['goe_max']:.2f}"
        brt  = f"{r['bert_top']} {r['bert_max']:.2f}"
        raw  = f"{r['raw_meta']} {r['raw_conf']:.2f}" if r["raw_meta"] else "(fallback)"
        fin  = f"{r['final']} {r['final_conf']:.2f}" + ("*" if r["guard_fired"] else "")
        gate = str(r["gate_alpha"]) if r["gate_alpha"] else "—"
        print(f"{mark:<5} {r['text'][:37]:<38} {goe:<18} {brt:<14} {raw:<14} {fin:<16} {gate:<30}")
    n_pass = sum(r["pass"] for r in results)
    print("─" * len(hdr))
    print(f"{n_pass}/{len(results)} passed   (* = NLP guard changed the raw prediction)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline meta-learner sentence evaluation")
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    ap.add_argument("--goe-model",  default=DEFAULT_GOE_MODEL)
    ap.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)
    ap.add_argument("--device",     default=-1, help="-1=CPU, 'mps', or CUDA index")
    ap.add_argument("--json",       default=None, help="write results JSON to this path")
    args = ap.parse_args()

    try:
        device = int(args.device)
    except (TypeError, ValueError):
        device = args.device

    print(f"Model pkl : {args.model_path}")
    print(f"GoE model : {args.goe_model}")
    print(f"BERT model: {args.bert_model}")

    model = load_meta_learner(args.model_path)
    print(f"Meta-learner: {type(model).__name__ if model is not None else 'None (rule-based fallback)'}\n")

    vader, bert, goe = _load_analyzers(args.bert_model, args.goe_model, device)
    results = evaluate(model, vader, bert, goe)
    print_table(results)

    if args.json:
        payload = {
            "model_path": args.model_path,
            "goe_model":  args.goe_model,
            "bert_model": args.bert_model,
            "results":    results,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults written to {args.json}")

    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
