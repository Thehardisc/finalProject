import argparse                                                # CLI args
import json                                                    # optional results dump
import os                                                      # paths + env
import sys                                                     # sys.path bootstrap + exit code

_HERE = os.path.dirname(__file__)                              # qa_suite/ dir
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))             # repo root
_SVC = os.path.join(_ROOT, "central_responder_service")        # service dir
for _p in (_ROOT, _SVC):                                       # make pipeline helpers importable
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.eval_sentences import (                          # reuse the offline battery
    evaluate, _load_analyzers, DEFAULT_GOE_MODEL, DEFAULT_BERT_MODEL, DEFAULT_MODEL_PATH,
)
from meta_learner import load_meta_learner                     # load the trained model (or None)


def main() -> int:                                             # entry point
    ap = argparse.ArgumentParser(description="Derive QA-suite thresholds from the live battery")  # parser
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))  # meta pkl
    ap.add_argument("--goe-model", default=DEFAULT_GOE_MODEL)   # GoEmotions checkpoint
    ap.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)  # BERT checkpoint
    ap.add_argument("--json", default=None, help="write the raw battery results here")  # optional dump
    args = ap.parse_args()                                     # parse

    model = load_meta_learner(args.model_path)                 # load model (None => fallback)
    mode = type(model).__name__ if model is not None else "None (rule-based fallback)"  # describe mode
    print(f"Meta-learner: {mode}")                            # report mode

    vader, bert, goe = _load_analyzers(args.bert_model, args.goe_model, -1)  # load analyzers (CPU)
    results = evaluate(model, vader, bert, goe)                # run the battery

    passing = [r for r in results if r["pass"]]               # passing sentences
    clear_confs = [r["final_conf"] for r in passing]          # their confidences
    entropies = [r["goe_entropy"] for r in results]           # all entropies

    print("\n=== Suggested qa_suite/thresholds.py values ===")  # header
    if model is not None and clear_confs:                     # confidence floor needs a real model
        floor = max(0.0, min(clear_confs) - 0.05)            # min passing conf minus margin
        print(f"CLEAR_CONF_FLOOR   = {floor:.2f}   "
              f"# min passing final_conf={min(clear_confs):.3f} (margin 0.05)")
    else:                                                     # fallback: skip
        print("CLEAR_CONF_FLOOR   = (skip — rule-based fallback, no calibrated model)")
    if entropies:                                             # suggest a vague entropy threshold
        print(f"VAGUE_ENTROPY_MIN  ~ {sum(entropies)/len(entropies):.2f}   "
              f"# clear-input entropy range [{min(entropies):.2f}, {max(entropies):.2f}]; "
              f"set vague threshold ABOVE the clear max")
    print(f"\n{sum(r['pass'] for r in results)}/{len(results)} battery sentences pass.")  # summary

    if args.json:                                             # optional raw dump
        with open(args.json, "w") as f:
            json.dump({"model": mode, "results": results}, f, indent=2)
        print(f"Raw results → {args.json}")
    return 0                                                  # success


if __name__ == "__main__":                                    # run as a script
    sys.exit(main())
