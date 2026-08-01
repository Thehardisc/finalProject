import json
import os

import pytest

from corpora import FAMILY, LABEL_FAMILY
import thresholds as T

_DATA = os.path.join(os.path.dirname(__file__), "data", "edge_messages.json")
with open(_DATA, encoding="utf-8") as _f:
    MESSAGES = json.load(_f)["messages"]

STRICT = [m for m in MESSAGES if m.get("strict")]


def _family_of(emotion):
    return FAMILY[LABEL_FAMILY[emotion]]


@pytest.mark.parametrize("msg", STRICT,
                         ids=[f'{m["edge"]}:{m["text"][:18]}' for m in STRICT])
def test_strict_edge_message(io, msg):
    fam = _family_of(msg["emotion"])
    r, ok = io(msg["text"], family=fam)
    assert ok, (
        f'{msg["text"]!r} [{msg["edge"]}] → {r["final_label"]}; expected {msg["emotion"]} (family {sorted(fam)})')


@pytest.mark.slow
def test_edge_aggregate_accuracy(io):
    from collections import defaultdict
    by_edge = defaultdict(lambda: [0, 0])
    hits, misses = 0, []
    for m in MESSAGES:
        r, ok = io(m["text"], family=_family_of(m["emotion"]))
        hits += ok
        by_edge[m["edge"]][0] += ok
        by_edge[m["edge"]][1] += 1
        if not ok:
            misses.append(f'  {m["edge"]:<14} {m["text"][:34]!r:<36} → {r["final_label"]} (want {m["emotion"]})')
    acc = hits / len(MESSAGES)
    print(f"\n[edge] family-accuracy={acc:.3f} ({hits}/{len(MESSAGES)})")
    for edge, (h, t) in sorted(by_edge.items()):
        print(f"  {edge:<16} {h}/{t} = {h/t:.2f}")
    if misses:
        print("misses:\n" + "\n".join(misses))
    assert acc >= T.EDGE_FAMILY_GATE, f"edge family accuracy {acc:.3f} < gate {T.EDGE_FAMILY_GATE}"
