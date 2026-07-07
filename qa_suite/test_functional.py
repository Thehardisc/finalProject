import pytest

from tools.eval_sentences import SENTENCE_BATTERY
import thresholds as T


@pytest.mark.parametrize("text,expected", SENTENCE_BATTERY,
                         ids=[s[:24] for s, _ in SENTENCE_BATTERY])
def test_clear_emotion(io, meta_model, text, expected):
    r, ok = io(text, accept=expected)
    assert ok, (
        f"{text!r} → {r['final_label']} (conf {r['final_conf']:.2f}); expected one of {sorted(expected)}")
    if meta_model is not None:
        assert r["final_conf"] >= T.CLEAR_CONF_FLOOR, (
            f"clear input {text!r} confidence {r['final_conf']:.2f} < floor {T.CLEAR_CONF_FLOOR}")


MIXED_CASES = [
    ("I'm thrilled about the move but also terrified of starting over",
     {"excitement", "joy", "optimism", "desire"}, {"fear", "nervousness"}),
    ("I'm so happy it's finally over, though I'll really miss everyone",
     {"joy", "relief", "gratitude", "love", "caring"}, {"sadness", "grief", "disappointment"}),
    ("That's hilarious but honestly kind of infuriating too",
     {"amusement", "joy"}, {"anger", "annoyance"}),
]


@pytest.mark.parametrize("text,clusterA,clusterB", MIXED_CASES,
                         ids=[c[0][:24] for c in MIXED_CASES])
def test_mixed_emotions(io, text, clusterA, clusterB):
    r, _ = io(text, note=f">=2 emotions; both {sorted(clusterA)} & {sorted(clusterB)} in top-6")
    top = r["sorted_goe"]
    above = [(lbl, sc) for lbl, sc in top if sc >= T.MIXED_FLOOR]
    assert len(above) >= 2, (
        f"{text!r}: expected >=2 emotions above {T.MIXED_FLOOR}, got {above[:4]}")
    top_labels = {lbl for lbl, _ in top[:6]}
    assert top_labels & clusterA, f"{text!r}: no {clusterA} emotion in top-6 {top[:6]}"
    assert top_labels & clusterB, f"{text!r}: no {clusterB} emotion in top-6 {top[:6]}"


NEUTRAL_FAMILY = {"neutral", "realization", "approval", "curiosity", "confusion"}
VAGUE_CASES = ["ok then", "sure, whatever you say", "I guess that works", "fine."]


@pytest.mark.parametrize("text", VAGUE_CASES)
def test_vague_no_false_positive(io, text):
    r, _ = io(text, note="neutral-family OR high-entropy OR low-confidence")
    ok = (r["final_label"] in NEUTRAL_FAMILY
          or r["entropy"] >= T.VAGUE_ENTROPY_MIN
          or r["final_conf"] <= T.VAGUE_CONF_MAX)
    assert ok, (
        f"{text!r}: vague input forced into a confident emotion "
        f"{r['final_label']!r} (conf {r['final_conf']:.2f}, entropy {r['entropy']:.2f})")


@pytest.mark.parametrize("text", ["furious", "thanks", "no", "love it", "ugh"])
def test_minimal_input(io, text):
    r, ok = io(text, valid=True)
    assert ok, f"{text!r} → invalid label {r['final_label']!r}"


def test_negation_flips_valence(io):
    pos, _ = io("this is good", note="baseline (positive)")
    neg, _ = io("this is not good", note="negated -> valence should drop")
    assert neg["compound"] < pos["compound"], (
        f"negation did not lower valence: good={pos['compound']:.3f}, not-good={neg['compound']:.3f}")


def test_intensifier_increases_magnitude(io):
    base, _ = io("I am happy", note="baseline")
    boost, _ = io("I am very happy", note="intensified -> |valence| should grow")
    assert abs(boost["compound"]) >= abs(base["compound"]) - 1e-6, (
        f"intensifier did not increase magnitude: happy={base['compound']:.3f}, very-happy={boost['compound']:.3f}")


def test_emoji_shifts_valence(io):
    base, _ = io("lunch", note="neutral baseline")
    pos, _ = io("lunch 😀", note="+emoji -> valence should rise")
    neg, _ = io("lunch 😢", note="-emoji -> valence should fall")
    assert pos["compound"] > base["compound"], f"😀 did not raise valence: {base['compound']:.3f} -> {pos['compound']:.3f}"
    assert neg["compound"] < base["compound"], f"😢 did not lower valence: {base['compound']:.3f} -> {neg['compound']:.3f}"


@pytest.mark.parametrize("text", ["im sooo angrryy rn", "thx sm 🙏", "u r the bestttt",
                                  "wtf is going onnnn", "soooo saaaad :("])
def test_robustness_no_crash(io, text):
    r, ok = io(text, valid=True)
    assert ok, f"{text!r} → invalid label {r['final_label']!r}"


@pytest.mark.parametrize("text", ["I'm absolutely furious right now", "Thank you so much"])
def test_determinism(io, text):
    a, _ = io(text, note="run 1")
    b, _ = io(text, note="run 2 -> must match run 1")
    assert a["final_label"] == b["final_label"], f"nondeterministic label for {text!r}"
    assert a["final_conf"] == pytest.approx(b["final_conf"], abs=1e-6), (
        f"nondeterministic confidence for {text!r}")


@pytest.mark.parametrize("text", ["", "    ", "\n\t ", "really " * 300])
def test_boundary_inputs(io, text):
    r, ok = io(text, valid=True)
    assert ok, f"boundary input → invalid label {r['final_label']!r}"
