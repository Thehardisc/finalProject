import json                                                    # load curated conversations
import os                                                      # path to the data file

import pytest                                                  # test framework

from corpora import FAMILY, LABEL_FAMILY, generate_conversations  # mappings + conversation generator
import thresholds as T                                         # conversation gates

_DATA = os.path.join(os.path.dirname(__file__), "data", "conversations.json")  # curated conversations file
with open(_DATA, encoding="utf-8") as _f:                      # open it
    CONVS = json.load(_f)["conversations"]                     # list of curated conversations

GEN_CONVS = generate_conversations(T.CONV_GEN_N)               # 3000 generated full conversations


def _trend_ok(comps, trend, eps=T.CONV_TREND_EPS):             # do per-turn compounds follow the arc?
    first, last, lo, hi = comps[0], comps[-1], min(comps), max(comps)  # key points of the valence curve
    if trend == "up":                                          # ends higher than it started
        return last - first >= eps
    if trend == "down":                                        # ends lower than it started
        return first - last >= eps
    if trend == "down_then_up":                                # dips below the start, then recovers
        return lo < first - eps and last - lo >= eps
    if trend == "up_then_down":                                # rises above the start, then falls
        return hi > first + eps and hi - last >= eps
    if trend == "positive":                                    # stays positive throughout
        return lo >= -0.25 and hi > 0.05
    if trend == "negative":                                    # stays negative throughout
        return hi <= 0.25 and lo < -0.05
    if trend == "flat":                                        # small overall range
        return (hi - lo) <= 0.60
    return False                                               # unknown trend


@pytest.fixture(scope="module")                                # run curated turns once
def conv_runs(analyze):                                        # id -> [(emotion, predicted, compound), ...]
    return {c["id"]: [(t["emotion"], r["final_label"], r["compound"])
                      for t in c["turns"] for r in (analyze(t["text"]),)]
            for c in CONVS}


def test_conversation_family_accuracy(io):                     # curated per-turn emotion accuracy (records each turn)
    hits = tot = 0                                             # counters
    misses = []                                                # miss report
    for conv in CONVS:                                         # each conversation
        for t in conv["turns"]:                               # each turn
            r, ok = io(t["text"], family=FAMILY[LABEL_FAMILY[t["emotion"]]])  # analyze + record expected/verdict
            hits += ok                                         # tally
            tot += 1                                           # tally
            if not ok:                                         # record miss
                misses.append(f'  {conv["id"]:<22} want {t["emotion"]:<13} got {r["final_label"]}')
    acc = hits / tot                                           # accuracy
    print(f"\n[conversation] per-turn family-accuracy={acc:.3f} ({hits}/{tot})")  # report
    if misses:                                                 # list misses
        print("misses:\n" + "\n".join(misses))
    assert acc >= T.CONV_FAMILY_GATE, f"per-turn family accuracy {acc:.3f} < gate {T.CONV_FAMILY_GATE}"  # gate


@pytest.mark.parametrize("conv", CONVS, ids=[c["id"] for c in CONVS])  # one test per curated conversation
def test_conversation_trend(conv_runs, conv):                  # curated arc must match the valence curve
    comps = [c for _, _, c in conv_runs[conv["id"]]]          # per-turn compounds
    assert _trend_ok(comps, conv["valence_trend"]), (          # trajectory check
        f'{conv["id"]} compounds={[round(c, 2) for c in comps]} trend={conv["valence_trend"]}')


@pytest.fixture(scope="module")                                # run all generated turns once
def gen_runs(analyze):                                         # per conversation: [(emotion, analyze-result), ...]
    return [[(t["emotion"], analyze(t["text"])) for t in c["turns"]] for c in GEN_CONVS]


@pytest.mark.slow                                              # heavy: thousands of turns
def test_generated_conversation_family_accuracy(gen_runs):     # generated per-turn emotion accuracy
    hits = tot = 0                                             # counters
    for rows in gen_runs:                                      # each conversation
        for emo, r in rows:                                    # each turn
            hits += r["final_label"] in FAMILY[LABEL_FAMILY[emo]]  # family hit?
            tot += 1                                           # tally
    acc = hits / tot                                           # accuracy
    print(f"\n[gen-conv] turns={tot}  per-turn family-accuracy={acc:.3f}")  # report
    assert acc >= T.CONV_GEN_FAMILY_GATE, f"generated per-turn accuracy {acc:.3f} < gate {T.CONV_GEN_FAMILY_GATE}"  # gate


@pytest.mark.slow                                              # heavy: thousands of conversations
def test_generated_conversation_arc_accuracy(gen_runs):        # fraction of generated convs matching their arc
    from collections import Counter                            # tally worst arcs
    ok = 0                                                     # matched conversations
    bad = Counter()                                            # mismatches per arc
    for conv, rows in zip(GEN_CONVS, gen_runs):                # pair conv with its runs
        comps = [r["compound"] for _, r in rows]              # per-turn compounds
        if _trend_ok(comps, conv["valence_trend"]):           # trajectory match?
            ok += 1                                            # count match
        else:                                                  # else record arc
            bad[conv["arc"]] += 1
    frac = ok / len(GEN_CONVS)                                 # match fraction
    print(f"\n[gen-conv] arc-trajectory match={frac:.3f} ({ok}/{len(GEN_CONVS)})  worst arcs={bad.most_common(4)}")  # report
    assert frac >= T.CONV_ARC_GATE, f"generated arc match {frac:.3f} < gate {T.CONV_ARC_GATE}"  # gate
