import json
import os

import pytest

from corpora import LABEL_FAMILY
from shared.constants import EMOTION_LABELS
import thresholds as T

pytestmark = pytest.mark.slow

_DATA = os.path.join(os.path.dirname(__file__), "data", "goemotions_sample.jsonl")


def _load():
    if not os.path.exists(_DATA):
        pytest.skip(f"{_DATA} missing — run `python qa_suite/build_goemotions_corpus.py`")
    with open(_DATA, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_goemotions_benchmark(analyze):
    rows = _load()
    top1 = fam = 0
    invalid = set()
    for r in rows:
        pred = analyze(r["text"])["final_label"]
        if pred not in EMOTION_LABELS:
            invalid.add(pred)
        gold = set(r["labels"])
        gold_fams = {LABEL_FAMILY[g] for g in gold if g in LABEL_FAMILY}
        top1 += pred in gold
        fam += LABEL_FAMILY.get(pred) in gold_fams
    n = len(rows)
    print(f"\n[goemotions] n={n}  top1-exact={top1/n:.3f}  family-accuracy={fam/n:.3f}  "
          f"(gate {T.GOEMOTIONS_FAMILY_GATE})")
    assert not invalid, f"non-canonical labels emitted: {invalid}"
    assert fam / n >= T.GOEMOTIONS_FAMILY_GATE, (
        f"real-data family accuracy {fam/n:.3f} < gate {T.GOEMOTIONS_FAMILY_GATE}")
