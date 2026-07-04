import pytest

from corpora import labeled_cases, LABEL_FAMILY, EMOTION_TRIGGERS
from shared.constants import EMOTION_LABELS
import thresholds as T

pytestmark = pytest.mark.slow

FAMILIES = ["joy", "anger", "sadness", "fear", "surprise", "neutral"]


@pytest.fixture(scope="module")
def corpus_pred(analyze):
    return [(text, target, fam, analyze(text)["final_label"])
            for text, target, fam in labeled_cases()]


def test_overall_emotion_accuracy(corpus_pred):
    hits = sum(pred in fam for _, _, fam, pred in corpus_pred)
    acc = hits / len(corpus_pred)
    exact = {target for _, target, _, pred in corpus_pred if pred == target}
    missing = sorted(set(EMOTION_TRIGGERS) - exact)
    print(f"\n[coverage] family-accuracy={acc:.3f} ({hits}/{len(corpus_pred)})  "
          f"exact-label coverage={len(exact)}/28  missing-exact={missing}")
    assert acc >= T.CORPUS_ACCURACY_GATE, (
        f"family accuracy {acc:.3f} < gate {T.CORPUS_ACCURACY_GATE} "
        f"(emotion classification regressed)")


@pytest.mark.parametrize("family", FAMILIES)
def test_family_accuracy(corpus_pred, family):
    rows = [(pred, fam) for _, target, fam, pred in corpus_pred if LABEL_FAMILY[target] == family]
    assert rows, f"no corpus rows for family {family}"
    acc = sum(pred in fam for pred, fam in rows) / len(rows)
    assert acc >= T.FAMILY_ACCURACY_FLOOR, (
        f"{family} family accuracy {acc:.2f} < floor {T.FAMILY_ACCURACY_FLOOR}")


def test_every_label_returns_valid(corpus_pred):
    bad = {pred for _, _, _, pred in corpus_pred if pred not in EMOTION_LABELS}
    assert not bad, f"non-canonical labels emitted: {bad}"
