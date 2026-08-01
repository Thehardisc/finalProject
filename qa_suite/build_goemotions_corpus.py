import argparse
import json
import os
import random

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "data", "goemotions_sample.jsonl")
DATASET = "google-research-datasets/go_emotions"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    import datasets
    ds = datasets.load_dataset(DATASET, "simplified", split=args.split)
    names = ds.features["labels"].feature.names

    rows = []
    for r in ds:
        text = (r.get("text") or "").strip()
        labs = [names[i] for i in r.get("labels", [])]
        if text and labs:
            rows.append({"text": text, "labels": labs, "id": r.get("id")})

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    sample = rows[: args.n]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    from collections import Counter
    hist = Counter(l for row in sample for l in row["labels"])
    print(f"wrote {len(sample)} messages → {args.out}")
    print("label coverage:", len(hist), "/ 28 labels present")
    print("most common:", hist.most_common(6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
