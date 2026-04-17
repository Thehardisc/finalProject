"""
Offline training script for the Central Responder meta-learner.

Run this once manually (NOT at Docker boot) to produce:
  central_responder_service/models/meta_weights.pkl
  central_responder_service/models/meta_weights_meta.json

Usage:
  pip install -r training/requirements.txt
  python training/train_meta_learner.py

The script:
  1. Loads the GoEmotions dataset (58k Reddit comments, 28-class labels)
  2. Runs each sample through all 4 local model analyzers
  3. Builds a feature vector from the combined model outputs
  4. Trains a Logistic Regression meta-learner on the feature vectors vs gold labels
  5. Evaluates on validation and test splits
  6. Saves the trained model and metadata
"""

import os
import sys
import json
import pickle
import datetime
import numpy as np
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if os.environ.get("MODEL_PATH"):
    PKL_PATH = Path(os.environ["MODEL_PATH"])
    OUTPUT_DIR = PKL_PATH.parent
elif os.path.exists("/app"):
    # If we are running inside the container, write to /app/models
    OUTPUT_DIR = Path("/app/models")
    PKL_PATH = OUTPUT_DIR / "meta_weights.pkl"
else:
    # Local fallback
    OUTPUT_DIR = ROOT / "models"
    PKL_PATH = OUTPUT_DIR / "meta_weights.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = OUTPUT_DIR / (PKL_PATH.stem + "_meta.json")


# ── Emotion label space (must match central_responder_service/main.py) ────────
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
LABEL_TO_IDX = {label: i for i, label in enumerate(EMOTION_LABELS)}

# ── VADER labels (fixed order for consistent feature vector) ──────────────────
VADER_KEYS = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']

# ── BERT (7 Ekman) labels ─────────────────────────────────────────────────────
BERT_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# Feature vector dimensions:
#   VADER:       4
#   BERT:        7
#   GoEmotions: 28
#   EmojiNet:   28  (mapped to GoEmotions label space, zeros if not applicable)
# Total:        67
FEATURE_DIM = len(VADER_KEYS) + len(BERT_LABELS) + len(EMOTION_LABELS) + len(EMOTION_LABELS)


def build_feature_vector(vader_scores: dict, bert_scores: dict,
                          goemotions_scores: dict, emojinet_scores: dict) -> np.ndarray:
    """Assemble a consistent fixed-length feature vector from all 4 model outputs."""
    vec = []

    # VADER block (4 dims)
    for k in VADER_KEYS:
        vec.append(vader_scores.get(k, 0.0))

    # BERT block (7 dims, fixed order)
    for k in BERT_LABELS:
        vec.append(bert_scores.get(k, 0.0))

    # GoEmotions block (28 dims, fixed order)
    for k in EMOTION_LABELS:
        vec.append(goemotions_scores.get(k, 0.0))

    # EmojiNet block (28 dims mapped to GoEmotions label space, zeros if no emojis)
    for k in EMOTION_LABELS:
        vec.append(emojinet_scores.get(k, 0.0))

    return np.array(vec, dtype=np.float32)


def init_models():
    """Load all 4 model analyzers. Heavy imports here so they only run once."""
    print("Loading VADER...")
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()

    import torch
    device = 0 if torch.cuda.is_available() else -1

    print("Loading BERT (j-hartmann/emotion-english-distilroberta-base)...")
    from transformers import pipeline as hf_pipeline
    bert = hf_pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        device=device
    )

    print("Loading GoEmotions (SamLowe/roberta-base-go_emotions)...")
    goemotions = hf_pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        return_all_scores=True,
        device=device
    )

    print("All models loaded.\n")
    return vader, bert, goemotions


def run_vader(vader_model, text: str) -> dict:
    scores = vader_model.polarity_scores(text)
    return {
        "vader_neg": scores["neg"],
        "vader_neu": scores["neu"],
        "vader_pos": scores["pos"],
        "vader_compound": scores["compound"]
    }


def run_bert(bert_model, text: str) -> dict:
    try:
        results = bert_model(text[:512])  # transformer token limit
        return {r['label']: r['score'] for r in results[0]}
    except Exception as e:
        print(f"  BERT error: {e}")
        return {}


def run_goemotions(goemotions_model, text: str) -> dict:
    try:
        results = goemotions_model(text[:512])
        return {r['label']: r['score'] for r in results[0]}
    except Exception as e:
        print(f"  GoEmotions error: {e}")
        return {}


def run_emojinet(text: str) -> dict:
    """Simplified inline EmojiNet — same DB as emojinet_service/main.py."""
    import emoji as emoji_lib
    EMOJINET_DB = {
        "😂": {"emotions": {"joy": 0.9, "amusement": 0.95}},
        "😭": {"emotions": {"sadness": 0.8, "grief": 0.6, "joy": 0.2}},
        "😍": {"emotions": {"love": 0.95, "admiration": 0.9, "joy": 0.8}},
        "🔥": {"emotions": {"excitement": 0.9, "admiration": 0.8, "joy": 0.7}},
        "💀": {"emotions": {"amusement": 0.9, "joy": 0.7, "fear": 0.1}},
        "🙃": {"emotions": {"amusement": 0.4, "annoyance": 0.6, "confusion": 0.3}},
        "🤔": {"emotions": {"curiosity": 0.8, "confusion": 0.4, "disapproval": 0.2}},
        "🙄": {"emotions": {"annoyance": 0.9, "disapproval": 0.8, "disgust": 0.5}},
        "💩": {"emotions": {"disgust": 0.8, "amusement": 0.5, "annoyance": 0.4}},
        "❤️": {"emotions": {"love": 1.0, "caring": 0.9, "joy": 0.8}},
        "✨": {"emotions": {"excitement": 0.7, "admiration": 0.6, "joy": 0.5}},
    }
    found = emoji_lib.distinct_emoji_list(text)
    if not found:
        return {}
    scores = {}
    count = 0
    for char in found:
        entry = EMOJINET_DB.get(char) or EMOJINET_DB.get(char.replace('\ufe0f', ''))
        if entry:
            for emo, score in entry["emotions"].items():
                scores[emo] = scores.get(emo, 0.0) + score
            count += 1
    if count == 0:
        return {}
    return {k: v / count for k, v in scores.items()}


def process_split(split_data, vader, bert, goemotions, split_name: str):
    """Run all 4 analyzers on a dataset split, return (X, y)."""
    X, y = [], []
    total = len(split_data)
    print(f"\nProcessing {split_name} split ({total} samples)...")

    for i, sample in enumerate(split_data):
        if i % 500 == 0:
            print(f"  [{split_name}] {i}/{total}")

        text = sample["text"]

        # Run analyzers
        vader_scores = run_vader(vader, text)
        bert_scores = run_bert(bert, text)
        goemotions_scores = run_goemotions(goemotions, text)
        emojinet_scores = run_emojinet(text)

        fv = build_feature_vector(vader_scores, bert_scores, goemotions_scores, emojinet_scores)

        # GoEmotions uses multi-label — pick label with highest id value, or first positive
        label_ids = sample.get("labels", [])
        if not label_ids:
            continue  # skip unlabeled samples

        # Use the first label (or max confidence from goemotions_scores as proxy)
        # Map GoEmotions label index (0-27) to our EMOTION_LABELS
        # GoEmotions label indices match our EMOTION_LABELS ordering 
        gold_label_idx = label_ids[0]  # take first label for single-label training
        if gold_label_idx >= len(EMOTION_LABELS):
            continue

        gold_label = EMOTION_LABELS[gold_label_idx]

        X.append(fv)
        y.append(gold_label)

    print(f"  Done. {len(X)} usable samples from {split_name}.")
    return np.array(X), np.array(y)


def train(max_samples_per_split: int = 10000):
    """Full training pipeline."""
    print("=" * 60)
    print("Meta-Learner Training Script")
    print("=" * 60)

    # ── Load GoEmotions dataset ───────────────────────────────────────────────
    print("\nLoading GoEmotions dataset from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    except Exception as e:
        print(f"ERROR: Could not load GoEmotions dataset: {e}")
        print("Make sure 'datasets' is installed: pip install datasets")
        sys.exit(1)

    train_data = list(dataset["train"])[:max_samples_per_split]
    val_data   = list(dataset["validation"])[:max_samples_per_split // 5]
    test_data  = list(dataset["test"])[:max_samples_per_split // 5]

    print(f"Dataset sizes — Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # ── Load analyzers ────────────────────────────────────────────────────────
    vader, bert, goemotions = init_models()

    # ── Process splits ────────────────────────────────────────────────────────
    X_train, y_train = process_split(train_data, vader, bert, goemotions, "train")
    X_val,   y_val   = process_split(val_data,   vader, bert, goemotions, "val")
    X_test,  y_test  = process_split(test_data,  vader, bert, goemotions, "test")

    # ── Train Logistic Regression ─────────────────────────────────────────────
    print("\nTraining Logistic Regression meta-learner...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, classification_report

    meta_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            multi_class='multinomial',
            class_weight='balanced',  # handles class imbalance in GoEmotions
            random_state=42
        ))
    ])

    meta_pipeline.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_acc  = accuracy_score(y_val,  meta_pipeline.predict(X_val))
    test_acc = accuracy_score(y_test, meta_pipeline.predict(X_test))

    print(f"\n── Results ──────────────────────────────────────")
    print(f"  Validation accuracy : {val_acc:.4f}")
    print(f"  Test accuracy       : {test_acc:.4f}")
    print(f"\n── Classification Report (Test) ─────────────────")
    print(classification_report(y_test, meta_pipeline.predict(X_test),
                                 labels=list(set(y_test)), zero_division=0))

    # ── Save model ────────────────────────────────────────────────────────────
    print(f"\nSaving model to {PKL_PATH}...")
    with open(PKL_PATH, 'wb') as f:
        pickle.dump(meta_pipeline, f)

    metadata = {
        "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
        "training_samples": len(X_train),
        "validation_accuracy": round(val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "feature_dim": FEATURE_DIM,
        "emotion_labels": EMOTION_LABELS,
        "models_used": ["vader", "basic_bert", "go_emotions", "emojinet"],
        "dataset": "google-research-datasets/go_emotions (simplified)",
        "algorithm": "LogisticRegression (sklearn Pipeline + StandardScaler)"
    }

    with open(META_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {META_PATH}")
    print("\n✅ Training complete!")
    print(f"   PKL:  {PKL_PATH}")
    print(f"   Meta: {META_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the meta-learner for the Central Responder")
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Max training samples from GoEmotions (default: 10000)")
    args = parser.parse_args()
    train(max_samples_per_split=args.max_samples)
