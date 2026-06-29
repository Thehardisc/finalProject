import json                                                    # read the JSONL sample
import os                                                      # path + existence check

import pytest                                                  # test framework

from corpora import FAMILY, LABEL_FAMILY                       # family mapping
from shared.constants import EMOTION_LABELS                    # the 28 valid labels
import thresholds as T                                         # the real-data gate

pytestmark = pytest.mark.slow                                  # heavy (2000 messages) -> opt-in

_DATA = os.path.join(os.path.dirname(__file__), "data", "goemotions_sample.jsonl")  # real-data sample


def _load():                                                   # load the JSONL rows
    if not os.path.exists(_DATA):                              # built by build_goemotions_corpus.py
        pytest.skip(f"{_DATA} missing — run `python qa_suite/build_goemotions_corpus.py`")
    with open(_DATA, encoding="utf-8") as f:                   # open it
        return [json.loads(line) for line in f if line.strip()]  # one dict per line


def test_goemotions_benchmark(analyze):                        # benchmark vs human gold labels
    rows = _load()                                             # 2000 real messages
    top1 = fam = 0                                             # exact + family hit counters
    invalid = set()                                            # any non-canonical predictions
    for r in rows:                                             # each message
        pred = analyze(r["text"])["final_label"]              # predicted emotion
        if pred not in EMOTION_LABELS:                        # validity guard
            invalid.add(pred)
        gold = set(r["labels"])                               # human gold labels (multi-label)
        gold_fams = {LABEL_FAMILY[g] for g in gold if g in LABEL_FAMILY}  # their families
        top1 += pred in gold                                  # exact match against any gold
        fam += LABEL_FAMILY.get(pred) in gold_fams           # family match against any gold
    n = len(rows)                                             # total
    print(f"\n[goemotions] n={n}  top1-exact={top1/n:.3f}  family-accuracy={fam/n:.3f}  "  # report
          f"(gate {T.GOEMOTIONS_FAMILY_GATE})")
    assert not invalid, f"non-canonical labels emitted: {invalid}"  # all predictions valid
    assert fam / n >= T.GOEMOTIONS_FAMILY_GATE, (             # honest real-data gate
        f"real-data family accuracy {fam/n:.3f} < gate {T.GOEMOTIONS_FAMILY_GATE}")
