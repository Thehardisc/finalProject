import pytest                                                  # test framework

from corpora import labeled_cases, LABEL_FAMILY, EMOTION_TRIGGERS  # corpus + family mapping
from shared.constants import EMOTION_LABELS                    # the 28 valid labels
import thresholds as T                                         # coverage gates

pytestmark = pytest.mark.slow                                  # whole module is opt-in slow

FAMILIES = ["joy", "anger", "sadness", "fear", "surprise", "neutral"]  # the six families


@pytest.fixture(scope="module")                                # run the 224-sentence corpus once
def corpus_pred(analyze):                                      # -> [(text, target, family, predicted), ...]
    return [(text, target, fam, analyze(text)["final_label"])  # predict each sentence
            for text, target, fam in labeled_cases()]


def test_overall_emotion_accuracy(corpus_pred):                # global family accuracy + coverage report
    hits = sum(pred in fam for _, _, fam, pred in corpus_pred)  # family hits
    acc = hits / len(corpus_pred)                              # accuracy
    exact = {target for _, target, _, pred in corpus_pred if pred == target}  # labels hit exactly
    missing = sorted(set(EMOTION_TRIGGERS) - exact)            # labels never hit exactly
    print(f"\n[coverage] family-accuracy={acc:.3f} ({hits}/{len(corpus_pred)})  "  # report
          f"exact-label coverage={len(exact)}/28  missing-exact={missing}")
    assert acc >= T.CORPUS_ACCURACY_GATE, (                    # gate
        f"family accuracy {acc:.3f} < gate {T.CORPUS_ACCURACY_GATE} "
        f"(emotion classification regressed)")


@pytest.mark.parametrize("family", FAMILIES)                   # one test per family
def test_family_accuracy(corpus_pred, family):                 # per-family floor
    rows = [(pred, fam) for _, target, fam, pred in corpus_pred if LABEL_FAMILY[target] == family]  # this family
    assert rows, f"no corpus rows for family {family}"        # sanity
    acc = sum(pred in fam for pred, fam in rows) / len(rows)   # family accuracy
    assert acc >= T.FAMILY_ACCURACY_FLOOR, (                   # floor
        f"{family} family accuracy {acc:.2f} < floor {T.FAMILY_ACCURACY_FLOOR}")


def test_every_label_returns_valid(corpus_pred):               # all predictions are canonical labels
    bad = {pred for _, _, _, pred in corpus_pred if pred not in EMOTION_LABELS}  # any non-canonical?
    assert not bad, f"non-canonical labels emitted: {bad}"    # must be none
