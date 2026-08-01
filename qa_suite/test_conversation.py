import json
import os

import pytest

from corpora import FAMILY, LABEL_FAMILY, generate_conversations
import thresholds as T

_DATA = os.path.join(os.path.dirname(__file__), "data", "conversations.json")
with open(_DATA, encoding="utf-8") as _f:
    CONVS = json.load(_f)["conversations"]

GEN_CONVS = generate_conversations(T.CONV_GEN_N)


def _trend_ok(comps, trend, eps=T.CONV_TREND_EPS):
    first, last, lo, hi = comps[0], comps[-1], min(comps), max(comps)
    if trend == "up":
        return last - first >= eps
    if trend == "down":
        return first - last >= eps
    if trend == "down_then_up":
        return lo < first - eps and last - lo >= eps
    if trend == "up_then_down":
        return hi > first + eps and hi - last >= eps
    if trend == "positive":
        return lo >= -0.25 and hi > 0.05
    if trend == "negative":
        return hi <= 0.25 and lo < -0.05
    if trend == "flat":
        return (hi - lo) <= 0.60
    return False


@pytest.fixture(scope="module")
def conv_runs(analyze):
    return {c["id"]: [(t["emotion"], r["final_label"], r["compound"])
                      for t in c["turns"] for r in (analyze(t["text"]),)]
            for c in CONVS}


def test_conversation_family_accuracy(io):
    hits = tot = 0
    misses = []
    for conv in CONVS:
        for t in conv["turns"]:
            r, ok = io(t["text"], family=FAMILY[LABEL_FAMILY[t["emotion"]]])
            hits += ok
            tot += 1
            if not ok:
                misses.append(f'  {conv["id"]:<22} want {t["emotion"]:<13} got {r["final_label"]}')
    acc = hits / tot
    print(f"\n[conversation] per-turn family-accuracy={acc:.3f} ({hits}/{tot})")
    if misses:
        print("misses:\n" + "\n".join(misses))
    assert acc >= T.CONV_FAMILY_GATE, f"per-turn family accuracy {acc:.3f} < gate {T.CONV_FAMILY_GATE}"


@pytest.mark.parametrize("conv", CONVS, ids=[c["id"] for c in CONVS])
def test_conversation_trend(conv_runs, conv):
    comps = [c for _, _, c in conv_runs[conv["id"]]]
    assert _trend_ok(comps, conv["valence_trend"]), (
        f'{conv["id"]} compounds={[round(c, 2) for c in comps]} trend={conv["valence_trend"]}')


@pytest.fixture(scope="module")
def gen_runs(analyze):
    return [[(t["emotion"], analyze(t["text"])) for t in c["turns"]] for c in GEN_CONVS]


@pytest.mark.slow
def test_generated_conversation_family_accuracy(gen_runs):
    hits = tot = 0
    for rows in gen_runs:
        for emo, r in rows:
            hits += r["final_label"] in FAMILY[LABEL_FAMILY[emo]]
            tot += 1
    acc = hits / tot
    print(f"\n[gen-conv] turns={tot}  per-turn family-accuracy={acc:.3f}")
    assert acc >= T.CONV_GEN_FAMILY_GATE, f"generated per-turn accuracy {acc:.3f} < gate {T.CONV_GEN_FAMILY_GATE}"


@pytest.mark.slow
def test_generated_conversation_arc_accuracy(gen_runs):
    from collections import Counter
    ok = 0
    bad = Counter()
    for conv, rows in zip(GEN_CONVS, gen_runs):
        comps = [r["compound"] for _, r in rows]
        if _trend_ok(comps, conv["valence_trend"]):
            ok += 1
        else:
            bad[conv["arc"]] += 1
    frac = ok / len(GEN_CONVS)
    print(f"\n[gen-conv] arc-trajectory match={frac:.3f} ({ok}/{len(GEN_CONVS)})  worst arcs={bad.most_common(4)}")
    assert frac >= T.CONV_ARC_GATE, f"generated arc match {frac:.3f} < gate {T.CONV_ARC_GATE}"
