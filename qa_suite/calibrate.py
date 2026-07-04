import argparse
import json
import os
import sys

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_SVC = os.path.join(_ROOT, "central_responder_service")
for _p in (_ROOT, _SVC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.eval_sentences import (
    evaluate, _load_analyzers, DEFAULT_GOE_MODEL, DEFAULT_BERT_MODEL, DEFAULT_MODEL_PATH,
)
from meta_learner import load_meta_learner


def main() -> int:
    ap = argparse.ArgumentParser(description="Derive QA-suite thresholds from the live battery")
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    ap.add_argument("--goe-model", default=DEFAULT_GOE_MODEL)
    ap.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)
    ap.add_argument("--json", default=None, help="write the raw battery results here")
    args = ap.parse_args()

    model = load_meta_learner(args.model_path)
    mode = type(model).__name__ if model is not None else "None (rule-based fallback)"
    print(f"Meta-learner: {mode}")

    vader, bert, goe = _load_analyzers(args.bert_model, args.goe_model, -1)
    results = evaluate(model, vader, bert, goe)

    passing = [r for r in results if r["pass"]]
    clear_confs = [r["final_conf"] for r in passing]
    entropies = [r["goe_entropy"] for r in results]

    print("\n=== Suggested qa_suite/thresholds.py values ===")
    if model is not None and clear_confs:
        floor = max(0.0, min(clear_confs) - 0.05)
        print(f"CLEAR_CONF_FLOOR   = {floor:.2f}   "
              f"# min passing final_conf={min(clear_confs):.3f} (margin 0.05)")
    else:
        print("CLEAR_CONF_FLOOR   = (skip — rule-based fallback, no calibrated model)")
    if entropies:
        print(f"VAGUE_ENTROPY_MIN  ~ {sum(entropies)/len(entropies):.2f}   "
              f"# clear-input entropy range [{min(entropies):.2f}, {max(entropies):.2f}]; "
              f"set vague threshold ABOVE the clear max")
    print(f"\n{sum(r['pass'] for r in results)}/{len(results)} battery sentences pass.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"model": mode, "results": results}, f, indent=2)
        print(f"Raw results → {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
