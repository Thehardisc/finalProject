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
import time
import numpy as np
from pathlib import Path
from collections import Counter
from shared.utils.logger import get_logger

logger = get_logger("train_meta_learner")

# paths
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


from shared.constants import EMOTION_LABELS, VADER_KEYS, BERT_LABELS, FEATURE_DIM
LABEL_TO_IDX = {label: i for i, label in enumerate(EMOTION_LABELS)}


def build_feature_vector(vader_scores: dict, bert_scores: dict,
                          goemotions_scores: dict, emojinet_scores: dict,
                          prev_valence: float = 0.0, prev_mood: str = "neutral") -> np.ndarray:
    """
    Assemble a consistent 96D fixed-length feature vector.
    Must match meta_learner.py exact ordering.
    """
    vec = []

    # Block 1: VADER (4 dims)
    for k in VADER_KEYS:
        vec.append(vader_scores.get(k, 0.0))

    # Block 2: BERT (7 dims)
    for k in BERT_LABELS:
        vec.append(bert_scores.get(k, 0.0))

    # Block 3: GoEmotions (28 dims)
    for k in EMOTION_LABELS:
        vec.append(goemotions_scores.get(k, 0.0))

    # Block 4: EmojiNet (28 dims)
    for k in EMOTION_LABELS:
        vec.append(emojinet_scores.get(k, 0.0))
        
    # Block 5: Contextual Valence (1 dim)
    vec.append(prev_valence)
    
    # Block 6: Previous Mood One-Hot (28 dims)
    mood_vec = [0.0] * len(EMOTION_LABELS)
    if prev_mood in LABEL_TO_IDX:
        mood_vec[LABEL_TO_IDX[prev_mood]] = 1.0
    vec.extend(mood_vec)

    return np.array(vec, dtype=np.float32)


def init_models():
    """Load all 4 model analyzers. Heavy imports here so they only run once."""
    logger.info("Loading VADER...")
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()

    import torch
    device = 0 if torch.cuda.is_available() else -1

    logger.info("Loading BERT (j-hartmann/emotion-english-distilroberta-base)...")
    from transformers import pipeline as hf_pipeline
    bert = hf_pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        device=device
    )

    logger.info("Loading GoEmotions (SamLowe/roberta-base-go_emotions)...")
    goemotions = hf_pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        return_all_scores=True,
        device=device
    )

    logger.info("All models loaded.\n")
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
        logger.warning(f"  BERT error: {e}")
        return {}


def run_goemotions(goemotions_model, text: str) -> dict:
    try:
        results = goemotions_model(text[:512])
        return {r['label']: r['score'] for r in results[0]}
    except Exception as e:
        logger.warning(f"  GoEmotions error: {e}")
        return {}


def run_emojinet(text: str) -> dict:
    """EmojiNet feature extractor — uses shared EMOJI_EMOTION_DB (single source of truth)."""
    try:
        import emoji as emoji_lib
        from shared.constants import EMOJI_EMOTION_DB
        found = emoji_lib.distinct_emoji_list(text)
        if not found:
            return {}
        scores, count = {}, 0
        for char in found:
            entry = EMOJI_EMOTION_DB.get(char) or EMOJI_EMOTION_DB.get(char.replace('\ufe0f', ''))
            if entry:
                for emo, score in entry["emotions"].items():
                    scores[emo] = scores.get(emo, 0.0) + score
                count += 1
        return {k: v / count for k, v in scores.items()} if count else {}
    except Exception as e:
        logger.warning(f"  EmojiNet error: {e}")
        return {}


def process_split(split_data, vader, bert, goemotions, split_name: str):
    """Run all 4 analyzers on a dataset split, return (X, y)."""
    import random
    rng = random.Random(42)  # Seeded for reproducibility

    X, y = [], []
    total = len(split_data)
    logger.info(f"Processing {split_name} split ({total} samples)...")
    t0 = time.time()

    for i, sample in enumerate(split_data):
        if i % 500 == 0:
            logger.info(f"  [{split_name}] {i}/{total}")

        text = sample["text"]

        # Run analyzers
        vader_scores      = run_vader(vader, text)
        bert_scores       = run_bert(bert, text)
        goemotions_scores = run_goemotions(goemotions, text)
        emojinet_scores   = run_emojinet(text)

        # synthetic context injection
        # avg_valence uniformly sampled from [-1, 1]; prev_mood randomly sampled
        # from EMOTION_LABELS. This ensures the model learns from context features
        # instead of treating them as always-zero noise.
        synthetic_valence = rng.uniform(-1.0, 1.0)
        synthetic_mood    = rng.choice(EMOTION_LABELS)

        fv = build_feature_vector(
            vader_scores, bert_scores, goemotions_scores, emojinet_scores,
            prev_valence=synthetic_valence, prev_mood=synthetic_mood
        )

        # GoEmotions uses multi-label — take first (highest confidence) label
        label_ids = sample.get("labels", [])
        if not label_ids:
            continue

        gold_label_idx = label_ids[0]
        if gold_label_idx >= len(EMOTION_LABELS):
            continue

        gold_label = EMOTION_LABELS[gold_label_idx]

        X.append(fv)
        y.append(gold_label)

    dur = time.time() - t0
    logger.info(f"  Done. {len(X)} usable samples from {split_name}. Took {dur:.1f}s (avg {dur*1000/max(len(X),1):.2f}ms/smp)")

    dist = Counter(y).most_common(5)
    logger.log_stats(f"{split_name} distribution", dict(dist))
    return np.array(X), np.array(y)


def log_feature_importance(pipeline, feature_names):
    """Log the most influential features per model block."""
    try:
        clf = pipeline.named_steps['clf']
        # For multinomial LogisticRegression, coef_ is (n_classes, n_features)
        # We'll take the mean absolute coefficient across classes
        importances = np.mean(np.abs(clf.coef_), axis=0)
        
        # Group by blocks
        # Block 1: VADER (0-3)
        # Block 2: BERT (4-10)
        # Block 3: GoEmotions (11-38)
        # Block 4: EmojiNet (39-66)
        # Block 5: Valence (67)
        # Block 6: Mood History (68-95)
        
        vader_imp = np.mean(importances[0:4])
        bert_imp  = np.mean(importances[4:11])
        goemo_imp = np.mean(importances[11:39])
        emoji_imp = np.mean(importances[39:67])
        ctx_imp   = np.mean(importances[67:96])
        
        sums = vader_imp + bert_imp + goemo_imp + emoji_imp + ctx_imp
        
        importance_report = {
            "BERT Semantic Layer": f"{(bert_imp/sums):.2%}",
            "GoEmotions Depth":   f"{(goemo_imp/sums):.2%}",
            "Emoji Visual Signal": f"{(emoji_imp/sums):.2%}",
            "VADER Lexicon":       f"{(vader_imp/sums):.2%}",
            "Contextual Memory":   f"{(ctx_imp/sums):.2%}"
        }
        
        logger.log_stats("Knowledge Distillation (Influence)", importance_report)
    except Exception as e:
        logger.warning(f"Could not calculate feature importance: {e}")


def train(max_samples_per_split: int = 10000):
    """Full training pipeline."""
    logger.info("=" * 60)
    logger.info("Meta-Learner Training Script")
    logger.info("=" * 60)

    # load dataset
    logger.info("Loading GoEmotions dataset from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    except Exception as e:
        logger.error(f"ERROR: Could not load GoEmotions dataset: {e}")
        logger.error("Make sure 'datasets' is installed: pip install datasets")
        sys.exit(1)

    train_data = list(dataset["train"])[:max_samples_per_split]
    val_data   = list(dataset["validation"])[:max_samples_per_split // 5]
    test_data  = list(dataset["test"])[:max_samples_per_split // 5]

    logger.info(f"Dataset sizes — Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # load models
    vader, bert, goemotions = init_models()

    # generate features
    X_train, y_train = process_split(train_data, vader, bert, goemotions, "train")
    X_val,   y_val   = process_split(val_data,   vader, bert, goemotions, "val")
    X_test,  y_test  = process_split(test_data,  vader, bert, goemotions, "test")

    # train model
    logger.info("Training Logistic Regression meta-learner...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, classification_report, f1_score

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
    
    # check feature influence
    log_feature_importance(meta_pipeline, [])

    # test performance
    y_val_pred  = meta_pipeline.predict(X_val)
    y_test_pred = meta_pipeline.predict(X_test)

    val_acc   = accuracy_score(y_val,  y_val_pred)
    test_acc  = accuracy_score(y_test, y_test_pred)
    val_f1    = f1_score(y_val,  y_val_pred,  average='macro', zero_division=0)
    test_f1   = f1_score(y_test, y_test_pred, average='macro', zero_division=0)

    stats = {
        "Validation Accuracy":  f"{val_acc:.4f}",
        "Validation F1 (macro)": f"{val_f1:.4f}",
        "Test Accuracy":        f"{test_acc:.4f}",
        "Test F1 (macro)":      f"{test_f1:.4f}",
        "Train Samples":        len(X_train),
        "Val Samples":          len(X_val),
        "Test Samples":         len(X_test)
    }
    logger.log_stats("Final Results", stats)
    
    logger.info("Classification Report (Test):")
    report = classification_report(y_test, y_test_pred, labels=list(set(y_test)), zero_division=0)
    logger.info("\n" + report)

    # export model
    logger.info(f"Saving model to {PKL_PATH}...")
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

    logger.info(f"Saved metadata to {META_PATH}")
    logger.info("[OK] Training complete!")
    logger.info(f"   PKL:  {PKL_PATH}")
    logger.info(f"   Meta: {META_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the meta-learner for the Central Responder")
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Max training samples from GoEmotions (default: 10000)")
    args = parser.parse_args()
    train(max_samples_per_split=args.max_samples)
