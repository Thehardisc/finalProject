"""Feature extraction from enriched conversation JSONL."""

import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import List

from features.schema import (
    EMOTION_LABELS_28,
    BERT_LABELS_7,
    VADER_KEYS_4,
    MSG_DIM,
    WINDOW_SIZE,
    WINDOW_DIM,
    N_EMOTIONS,
    CDM_CTX_DIM,
)

DEFAULT_JSONL  = Path(__file__).parent.parent / "training_data" / "conversations.jsonl"
DEFAULT_OUT    = Path(__file__).parent.parent / "features"



def msg_to_vec(pipeline: dict) -> np.ndarray:
    """Convert a single message's pipeline dict to a 79-dim feature vector."""
    if not pipeline:
        return np.zeros(MSG_DIM, dtype=np.float32)

    stages = pipeline.get("stages", {})

    go    = stages.get("goemotions", {})
    bert  = stages.get("bert", {})
    vader = stages.get("vader", {})
    ce    = stages.get("context_engine", {})

    go_vec    = np.array([go.get(e, 0.0)    for e in EMOTION_LABELS_28], dtype=np.float32)
    bert_vec  = np.array([bert.get(e, 0.0)  for e in BERT_LABELS_7],     dtype=np.float32)
    vader_vec = np.array([vader.get(k, 0.0) for k in VADER_KEYS_4],      dtype=np.float32)

    cv_raw = ce.get("context_vector")
    if cv_raw and isinstance(cv_raw, list) and len(cv_raw) == CDM_CTX_DIM:
        ce_vec = np.array(cv_raw, dtype=np.float32)
    elif cv_raw and isinstance(cv_raw, list) and len(cv_raw) == 38:
        ce_vec = np.zeros(CDM_CTX_DIM, dtype=np.float32)
        ce_vec[:38] = cv_raw
    else:
        ce_vec = np.zeros(CDM_CTX_DIM, dtype=np.float32)

    return np.concatenate([go_vec, bert_vec, vader_vec, ce_vec])



def window_to_derived(window_msgs: List[dict]) -> np.ndarray:
    """Compute 9 derived features from a list of WINDOW_SIZE message dicts."""
    valences   = []
    agreements = []
    entropies  = []
    dominants  = []

    for msg in window_msgs:
        pipeline = msg.get("pipeline") or {}
        stages   = pipeline.get("stages", {})
        decision = pipeline.get("decision", {})

        vader    = stages.get("vader", {})
        go       = stages.get("goemotions", {})
        bert     = stages.get("bert", {})

        valences.append(float(vader.get("vader_compound", 0.0)))

        go_top   = max(go,   key=go.get)   if go   else "neutral"
        bert_top = max(bert, key=bert.get) if bert else "neutral"
        agreements.append(1.0 if go_top == bert_top else 0.0)

        probs   = np.array([go.get(e, 0.0) for e in EMOTION_LABELS_28], dtype=np.float32)
        s       = probs.sum()
        probs   = probs / s if s > 0 else probs
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        entropies.append(entropy)

        dominants.append(decision.get("dominant") or "neutral")

    while len(valences) < WINDOW_SIZE:
        valences.append(0.0)
        agreements.append(0.0)
        entropies.append(0.0)
        dominants.append("neutral")

    x     = np.arange(WINDOW_SIZE, dtype=np.float32)
    slope = float(np.polyfit(x, valences[:WINDOW_SIZE], 1)[0])
    var   = float(np.var(valences[:WINDOW_SIZE]))

    shifts = sum(1 for i in range(1, len(dominants)) if dominants[i] != dominants[i - 1])

    return np.array(
        [slope, var]
        + agreements[:WINDOW_SIZE]
        + entropies[:WINDOW_SIZE]
        + [float(shifts)],
        dtype=np.float32,
    )



def extract(jsonl_path: Path) -> dict:
    """Read conversations.jsonl and produce both windowed and sequence datasets."""
    conversations = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    conversations.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(conversations)} conversations from {jsonl_path.name}")

    sequences:       List[np.ndarray] = []
    seq_targets:     List[np.ndarray] = []
    seq_trajectories: List[str]       = []

    X_win:   List[np.ndarray] = []
    y_dist:  List[np.ndarray] = []
    y_label: List[str]        = []
    meta:    List[dict]       = []
    traj_win: List[str]       = []

    skipped = 0

    for conv in conversations:
        t_type  = conv.get("trajectory_type", "unknown")
        conv_id = conv.get("conversation_id", "")
        msgs    = conv.get("messages", [])

        valid = [
            m for m in msgs
            if m.get("pipeline") and m["pipeline"].get("stages")
        ]

        if len(valid) < WINDOW_SIZE + 1:
            print(f"  SKIP {conv_id[:12]} ({t_type}): only {len(valid)} valid msgs")
            skipped += 1
            continue

        vecs = np.stack([msg_to_vec(m["pipeline"]) for m in valid])

        tgt_rows = []
        for m in valid[1:]:
            go   = m["pipeline"].get("stages", {}).get("goemotions", {})
            trow = np.array([go.get(e, 0.0) for e in EMOTION_LABELS_28], dtype=np.float32)
            tgt_rows.append(trow)
        tgt = np.stack(tgt_rows)

        sequences.append(vecs)
        seq_targets.append(tgt)
        seq_trajectories.append(t_type)

        for i in range(len(valid) - WINDOW_SIZE):
            window     = valid[i : i + WINDOW_SIZE]
            target_msg = valid[i + WINDOW_SIZE]

            base    = np.concatenate([msg_to_vec(m["pipeline"]) for m in window])
            derived = window_to_derived(window)
            x       = np.concatenate([base, derived])

            go_next  = target_msg["pipeline"].get("stages", {}).get("goemotions", {})
            dist_tgt = np.array([go_next.get(e, 0.0) for e in EMOTION_LABELS_28], dtype=np.float32)
            dom_tgt  = target_msg["pipeline"].get("decision", {}).get("dominant") or "neutral"

            X_win.append(x)
            y_dist.append(dist_tgt)
            y_label.append(dom_tgt)
            traj_win.append(t_type)
            meta.append({"conversation_id": conv_id, "turn": i + WINDOW_SIZE, "trajectory": t_type})

    n_seq  = len(sequences)
    n_win  = len(X_win)

    print(f"\nExtraction summary:")
    print(f"  Valid conversations : {n_seq}  (skipped {skipped})")
    print(f"  Windowed samples    : {n_win}  (each window = {WINDOW_DIM} dims)")
    print(f"  Sequence samples    : {n_seq}  (variable length, 79 dims/step)")

    if n_win > 0:
        unique_labels, counts = np.unique(y_label, return_counts=True)
        print(f"\n  Label distribution (top 8):")
        for lbl, cnt in sorted(zip(unique_labels, counts), key=lambda x: -x[1])[:8]:
            print(f"    {lbl:<20} {cnt:>4} ({100*cnt/n_win:.1f}%)")

    return {
        "sequences":          sequences,
        "seq_targets":        seq_targets,
        "seq_trajectories":   seq_trajectories,
        "X_windows":          np.array(X_win,  dtype=np.float32) if X_win  else np.zeros((0, WINDOW_DIM), dtype=np.float32),
        "y_dist":             np.array(y_dist, dtype=np.float32) if y_dist else np.zeros((0, N_EMOTIONS), dtype=np.float32),
        "y_label":            y_label,
        "trajectory_types":   traj_win,
        "metadata":           meta,
        "n_conversations":    len(conversations),
        "n_valid":            n_seq,
        "n_windows":          n_win,
    }



def split_dataset(data: dict, train: float = 0.70, val: float = 0.15, seed: int = 42) -> dict:
    """Stratified split by trajectory type."""


    rng = np.random.default_rng(seed)
    traj_types = np.array(data["trajectory_types"])

    train_idx, val_idx, test_idx = [], [], []

    for ttype in np.unique(traj_types):
        idx = np.where(traj_types == ttype)[0]
        rng.shuffle(idx)
        n_train = int(len(idx) * train)
        n_val   = int(len(idx) * val)
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())

    splits = {}
    for name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        idxs = np.array(idxs)
        splits[name] = {
            "X_windows":       data["X_windows"][idxs],
            "y_dist":          data["y_dist"][idxs],
            "y_label":         [data["y_label"][i]       for i in idxs],
            "trajectory_types":[data["trajectory_types"][i] for i in idxs],
            "metadata":        [data["metadata"][i]      for i in idxs],
        }
        print(f"  {name:<6}: {len(idxs):>4} windows")

    return splits



def fit_scaler(X_train: np.ndarray) -> object:
    """Fit a StandardScaler on the training split. Returns the fitted scaler."""
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler



def save_dataset(data: dict, splits: dict, scaler, out_dir: Path):
    """Save all artefacts to out_dir/."""
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "dataset.npz",
        X_train=scaler.transform(splits["train"]["X_windows"]),
        X_val=scaler.transform(splits["val"]["X_windows"]),
        X_test=scaler.transform(splits["test"]["X_windows"]),
        y_dist_train=splits["train"]["y_dist"],
        y_dist_val=splits["val"]["y_dist"],
        y_dist_test=splits["test"]["y_dist"],
    )

    with open(out_dir / "sequences.pkl", "wb") as fh:
        pickle.dump({
            "sequences":        data["sequences"],
            "seq_targets":      data["seq_targets"],
            "seq_trajectories": data["seq_trajectories"],
        }, fh)

    with open(out_dir / "labels.json", "w") as fh:
        json.dump({
            "train": {"y_label": splits["train"]["y_label"], "trajectory": splits["train"]["trajectory_types"]},
            "val":   {"y_label": splits["val"]["y_label"],   "trajectory": splits["val"]["trajectory_types"]},
            "test":  {"y_label": splits["test"]["y_label"],  "trajectory": splits["test"]["trajectory_types"]},
            "emotion_labels": EMOTION_LABELS_28,
        }, fh, indent=2)

    with open(out_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)

    stats = {
        "n_conversations":  data["n_conversations"],
        "n_valid":          data["n_valid"],
        "n_windows":        data["n_windows"],
        "window_dim":       WINDOW_DIM,
        "msg_dim":          MSG_DIM,
        "n_emotions":       N_EMOTIONS,
        "window_size":      WINDOW_SIZE,
        "split": {
            "train": len(splits["train"]["y_label"]),
            "val":   len(splits["val"]["y_label"]),
            "test":  len(splits["test"]["y_label"]),
        },
    }
    with open(out_dir / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"\nSaved to {out_dir}/")
    print(f"  dataset.npz      — windowed X/y arrays (normalised)")
    print(f"  sequences.pkl    — LSTM sequence data")
    print(f"  labels.json      — label lists + emotion label schema")
    print(f"  scaler.pkl       — fitted StandardScaler")
    print(f"  stats.json       — dataset statistics")



def main():
    parser = argparse.ArgumentParser(description="Extract features from conversations.jsonl")
    parser.add_argument("--input",  default=str(DEFAULT_JSONL), help="Path to conversations.jsonl")
    parser.add_argument("--output", default=str(DEFAULT_OUT),   help="Output directory for artefacts")
    args = parser.parse_args()

    jsonl_path = Path(args.input)
    out_dir    = Path(args.output)

    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found. Run collect.py first.")
        return

    print("=== STEP 2: FEATURE EXTRACTION ===\n")
    data = extract(jsonl_path)

    if data["n_windows"] == 0:
        print("\nNo windows extracted — not enough valid messages. Check pipeline results.")
        return

    print("\nSplitting dataset...")
    splits = split_dataset(data)

    print("\nFitting scaler on training split...")
    scaler = fit_scaler(splits["train"]["X_windows"])

    save_dataset(data, splits, scaler, out_dir)
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
