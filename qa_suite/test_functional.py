import pytest                                                  # test framework

from training.eval_sentences import SENTENCE_BATTERY           # 12 curated clear-emotion sentences
from shared.constants import EMOTION_LABELS                    # the 28 valid labels
import thresholds as T                                         # calibrated bands


@pytest.mark.parametrize("text,expected", SENTENCE_BATTERY,    # one case per battery sentence
                         ids=[s[:24] for s, _ in SENTENCE_BATTERY])
def test_clear_emotion(io, meta_model, text, expected):        # clear emotion -> accepted label
    r, ok = io(text, accept=expected)                          # analyze + record expected/verdict
    assert ok, (                                               # predicted label must be in the accepted set
        f"{text!r} → {r['final_label']} (conf {r['final_conf']:.2f}); expected one of {sorted(expected)}")
    if meta_model is not None:                                 # confidence floor only with a trained model
        assert r["final_conf"] >= T.CLEAR_CONF_FLOOR, (        # clear input should clear the floor
            f"clear input {text!r} confidence {r['final_conf']:.2f} < floor {T.CLEAR_CONF_FLOOR}")


MIXED_CASES = [                                                # (text, cluster A, cluster B) dual-emotion inputs
    ("I'm thrilled about the move but also terrified of starting over",
     {"excitement", "joy", "optimism", "desire"}, {"fear", "nervousness"}),
    ("I'm so happy it's finally over, though I'll really miss everyone",
     {"joy", "relief", "gratitude", "love", "caring"}, {"sadness", "grief", "disappointment"}),
    ("That's hilarious but honestly kind of infuriating too",
     {"amusement", "joy"}, {"anger", "annoyance"}),
]


@pytest.mark.parametrize("text,clusterA,clusterB", MIXED_CASES,  # one case per mixed input
                         ids=[c[0][:24] for c in MIXED_CASES])
def test_mixed_emotions(io, text, clusterA, clusterB):         # mixed -> two emotions present
    r, _ = io(text, note=f">=2 emotions; both {sorted(clusterA)} & {sorted(clusterB)} in top-6")  # record intent
    top = r["sorted_goe"]                                      # GoE labels sorted by score
    above = [(lbl, sc) for lbl, sc in top if sc >= T.MIXED_FLOOR]  # labels above the mixed floor
    assert len(above) >= 2, (                                  # need at least two emotions
        f"{text!r}: expected >=2 emotions above {T.MIXED_FLOOR}, got {above[:4]}")
    top_labels = {lbl for lbl, _ in top[:6]}                   # consider the top-6 labels
    assert top_labels & clusterA, f"{text!r}: no {clusterA} emotion in top-6 {top[:6]}"  # cluster A present
    assert top_labels & clusterB, f"{text!r}: no {clusterB} emotion in top-6 {top[:6]}"  # cluster B present


NEUTRAL_FAMILY = {"neutral", "realization", "approval", "curiosity", "confusion"}  # acceptable "not a strong emotion"
VAGUE_CASES = ["ok then", "sure, whatever you say", "I guess that works", "fine."]  # ambiguous inputs


@pytest.mark.parametrize("text", VAGUE_CASES)                  # one case per vague input
def test_vague_no_false_positive(io, text):                   # vague -> no confident strong emotion
    r, _ = io(text, note="neutral-family OR high-entropy OR low-confidence")  # record intent
    ok = (r["final_label"] in NEUTRAL_FAMILY                   # neutral-family is fine
          or r["entropy"] >= T.VAGUE_ENTROPY_MIN               # OR high uncertainty (entropy)
          or r["final_conf"] <= T.VAGUE_CONF_MAX)              # OR low confidence
    assert ok, (                                               # fail only if forced into a confident emotion
        f"{text!r}: vague input forced into a confident emotion "
        f"{r['final_label']!r} (conf {r['final_conf']:.2f}, entropy {r['entropy']:.2f})")


@pytest.mark.parametrize("text", ["furious", "thanks", "no", "love it", "ugh"])  # 1-2 word inputs
def test_minimal_input(io, text):                             # minimal -> valid label, no crash
    r, ok = io(text, valid=True)                              # analyze + record validity verdict
    assert ok, f"{text!r} → invalid label {r['final_label']!r}"  # must be a canonical label


def test_negation_flips_valence(io):                          # negation lowers VADER valence
    pos, _ = io("this is good", note="baseline (positive)")   # baseline
    neg, _ = io("this is not good", note="negated -> valence should drop")  # negated
    assert neg["compound"] < pos["compound"], (               # must drop
        f"negation did not lower valence: good={pos['compound']:.3f}, not-good={neg['compound']:.3f}")


def test_intensifier_increases_magnitude(io):                 # intensifier raises magnitude
    base, _ = io("I am happy", note="baseline")               # baseline
    boost, _ = io("I am very happy", note="intensified -> |valence| should grow")  # intensified
    assert abs(boost["compound"]) >= abs(base["compound"]) - 1e-6, (  # magnitude must not shrink
        f"intensifier did not increase magnitude: happy={base['compound']:.3f}, very-happy={boost['compound']:.3f}")


def test_emoji_shifts_valence(io):                            # emoji shifts VADER valence directionally
    base, _ = io("lunch", note="neutral baseline")            # baseline
    pos, _ = io("lunch 😀", note="+emoji -> valence should rise")  # positive emoji
    neg, _ = io("lunch 😢", note="-emoji -> valence should fall")  # negative emoji
    assert pos["compound"] > base["compound"], f"😀 did not raise valence: {base['compound']:.3f} -> {pos['compound']:.3f}"
    assert neg["compound"] < base["compound"], f"😢 did not lower valence: {base['compound']:.3f} -> {neg['compound']:.3f}"


@pytest.mark.parametrize("text", ["im sooo angrryy rn", "thx sm 🙏", "u r the bestttt",  # noisy inputs
                                  "wtf is going onnnn", "soooo saaaad :("])
def test_robustness_no_crash(io, text):                       # noisy text -> no crash, valid label
    r, ok = io(text, valid=True)                              # analyze + record validity
    assert ok, f"{text!r} → invalid label {r['final_label']!r}"  # must be a canonical label


@pytest.mark.parametrize("text", ["I'm absolutely furious right now", "Thank you so much"])  # repeatable inputs
def test_determinism(io, text):                              # same input twice -> identical verdict
    a, _ = io(text, note="run 1")                            # first run
    b, _ = io(text, note="run 2 -> must match run 1")        # second run
    assert a["final_label"] == b["final_label"], f"nondeterministic label for {text!r}"  # same label
    assert a["final_conf"] == pytest.approx(b["final_conf"], abs=1e-6), (  # same confidence
        f"nondeterministic confidence for {text!r}")


@pytest.mark.parametrize("text", ["", "    ", "\n\t ", "really " * 300])  # empty / whitespace / very long
def test_boundary_inputs(io, text):                          # boundary -> no crash, valid label
    r, ok = io(text, valid=True)                             # must not raise; record validity
    assert ok, f"boundary input → invalid label {r['final_label']!r}"  # valid label
