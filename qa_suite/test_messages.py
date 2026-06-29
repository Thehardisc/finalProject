import json                                                    # load the message data file
import os                                                      # path to the data file

import pytest                                                  # test framework

from corpora import FAMILY, LABEL_FAMILY                       # emotion -> family mappings
import thresholds as T                                         # the edge accuracy gate

_DATA = os.path.join(os.path.dirname(__file__), "data", "edge_messages.json")  # authored messages file
with open(_DATA, encoding="utf-8") as _f:                      # open it
    MESSAGES = json.load(_f)["messages"]                       # list of {text, emotion, edge, strict}

STRICT = [m for m in MESSAGES if m.get("strict")]              # only the strict-tagged subset


def _family_of(emotion):                                       # acceptable family set for an emotion
    return FAMILY[LABEL_FAMILY[emotion]]                       # via LABEL_FAMILY then FAMILY


@pytest.mark.parametrize("msg", STRICT,                        # one test per strict message
                         ids=[f'{m["edge"]}:{m["text"][:18]}' for m in STRICT])
def test_strict_edge_message(io, msg):                         # strict -> predicted family must match
    fam = _family_of(msg["emotion"])                           # expected family
    r, ok = io(msg["text"], family=fam)                        # analyze + record expected/verdict
    assert ok, (                                               # must land in the right family
        f'{msg["text"]!r} [{msg["edge"]}] → {r["final_label"]}; expected {msg["emotion"]} (family {sorted(fam)})')


@pytest.mark.slow                                              # heavy: runs every message
def test_edge_aggregate_accuracy(io):                          # aggregate family-accuracy gate over ALL messages
    from collections import defaultdict                        # per-edge tally
    by_edge = defaultdict(lambda: [0, 0])                      # edge -> [hits, total]
    hits, misses = 0, []                                       # global hits + miss report
    for m in MESSAGES:                                         # each authored message
        r, ok = io(m["text"], family=_family_of(m["emotion"]))  # analyze + record expected/verdict
        hits += ok                                             # tally global
        by_edge[m["edge"]][0] += ok                            # tally per-edge hit
        by_edge[m["edge"]][1] += 1                             # tally per-edge total
        if not ok:                                             # record misses
            misses.append(f'  {m["edge"]:<14} {m["text"][:34]!r:<36} → {r["final_label"]} (want {m["emotion"]})')
    acc = hits / len(MESSAGES)                                 # overall family accuracy
    print(f"\n[edge] family-accuracy={acc:.3f} ({hits}/{len(MESSAGES)})")  # report
    for edge, (h, t) in sorted(by_edge.items()):               # per-edge breakdown
        print(f"  {edge:<16} {h}/{t} = {h/t:.2f}")
    if misses:                                                 # list the misses
        print("misses:\n" + "\n".join(misses))
    assert acc >= T.EDGE_FAMILY_GATE, f"edge family accuracy {acc:.3f} < gate {T.EDGE_FAMILY_GATE}"  # gate
