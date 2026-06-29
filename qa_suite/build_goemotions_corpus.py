import argparse                                                # CLI args
import json                                                    # write JSONL
import os                                                      # paths
import random                                                  # deterministic sampling

HERE = os.path.dirname(__file__)                               # qa_suite/ dir
OUT = os.path.join(HERE, "data", "goemotions_sample.jsonl")    # output path
DATASET = "google-research-datasets/go_emotions"               # HF dataset id


def main() -> int:                                             # entry point
    ap = argparse.ArgumentParser()                             # parser
    ap.add_argument("--n", type=int, default=2000)             # how many messages to sample
    ap.add_argument("--seed", type=int, default=7)             # sampling seed (reproducible)
    ap.add_argument("--split", default="test")                 # which split to draw from
    ap.add_argument("--out", default=OUT)                      # output file
    args = ap.parse_args()                                     # parse

    import datasets                                            # HF datasets (only needed to rebuild)
    ds = datasets.load_dataset(DATASET, "simplified", split=args.split)  # load the split
    names = ds.features["labels"].feature.names                # label index -> label name

    rows = []                                                  # cleaned rows
    for r in ds:                                               # each dataset row
        text = (r.get("text") or "").strip()                  # the message
        labs = [names[i] for i in r.get("labels", [])]        # gold label names
        if text and labs:                                     # keep non-empty, labeled rows
            rows.append({"text": text, "labels": labs, "id": r.get("id")})

    rng = random.Random(args.seed)                             # seeded RNG
    rng.shuffle(rows)                                          # deterministic shuffle
    sample = rows[: args.n]                                    # take the first n

    os.makedirs(os.path.dirname(args.out), exist_ok=True)      # ensure data/ exists
    with open(args.out, "w") as f:                             # write JSONL
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")  # one JSON object per line

    from collections import Counter                            # quick sanity histogram
    hist = Counter(l for row in sample for l in row["labels"])  # label frequencies
    print(f"wrote {len(sample)} messages → {args.out}")        # report
    print("label coverage:", len(hist), "/ 28 labels present")  # coverage
    print("most common:", hist.most_common(6))                 # top labels
    return 0                                                   # success


if __name__ == "__main__":                                    # run as a script
    raise SystemExit(main())
