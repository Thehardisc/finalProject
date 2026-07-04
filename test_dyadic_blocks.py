"""test_dyadic_blocks.py — ניתוח שיחה משלושה גושי טקסט"""

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline


BLOCK_A = """
I had such a rough day today.
Nothing went right at work and I feel completely drained.
I just needed to talk to someone.
""".strip()

BLOCK_B = """
What do you want me to do about it?
You always complain about work but never change anything.
Fine, tell me, but make it quick.
""".strip()

BLOCK_AB = ""

# ─────────────────────────────── פיצול ──────────────────────────────────────

def split_block(text):
    """פצל גוש טקסט להודעות — לפי שורות, דלג על שורות ריקות."""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def parse_mixed_block(text):
    """פרסר גוש מעורב. תומך בפורמטים:"""
    msgs = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = re.match(r'^([A-Za-z0-9_\[\]]+)[\]:][\s]*(.*)', ln)
        if m:
            raw_spk = m.group(1).strip("[]")
            msgs.append({"speaker": raw_spk, "text": m.group(2).strip()})
        else:
            if msgs:
                msgs[-1]["text"] += " " + ln
    return msgs


def interleave(msgs_a, msgs_b):
    """סרג A ו-B לפי תורות: A0, B0, A1, B1, ..."""
    result = []
    for i in range(max(len(msgs_a), len(msgs_b))):
        if i < len(msgs_a):
            result.append({"speaker": "A", "text": msgs_a[i]})
        if i < len(msgs_b):
            result.append({"speaker": "B", "text": msgs_b[i]})
    return result



print("Loading models...", flush=True)
vader_analyzer = SentimentIntensityAnalyzer()

_device = "mps"
try:
    import torch
    if not torch.backends.mps.is_available():
        _device = "cpu"
except Exception:
    _device = "cpu"

goe_pipeline = pipeline(
    "text-classification",
    model="bhadresh-savani/bert-base-go-emotion",
    top_k=None,
    device=_device,
)
print(f"Models loaded on {_device}\n", flush=True)


GOEMO_VALENCE = {
    'anger': -0.8, 'annoyance': -0.5, 'disapproval': -0.5, 'disgust': -0.7,
    'sadness': -0.7, 'grief': -0.8, 'fear': -0.7, 'remorse': -0.6,
    'nervousness': -0.4, 'embarrassment': -0.4, 'disappointment': -0.6,
    'joy': 0.9, 'love': 0.9, 'excitement': 0.8, 'gratitude': 0.8,
    'caring': 0.7, 'optimism': 0.7, 'approval': 0.6, 'admiration': 0.7,
    'amusement': 0.8, 'relief': 0.6, 'pride': 0.7,
    'curiosity': 0.2, 'realization': 0.1, 'surprise': 0.1,
    'confusion': -0.1, 'desire': 0.4, 'neutral': 0.0,
}

def effective_valence(goe: dict, vader_val: float) -> float:
    """כשVADER ניטרלי אבל GoEmotions ברור — סמוך על GoEmotions."""
    top     = max(goe, key=goe.get)
    goe_val = GOEMO_VALENCE.get(top, 0.0)
    conf    = goe[top]
    if abs(vader_val) < 0.15 and abs(goe_val) > 0.3 and conf > 0.4:
        return goe_val * 0.7
    return vader_val

def analyze(text):
    v   = vader_analyzer.polarity_scores(text)
    raw = goe_pipeline(text)[0]
    g   = {r["label"]: r["score"] for r in raw}
    vader_val = float(v["compound"])
    return g, effective_valence(g, vader_val)



SUPPORT  = {"caring", "love", "gratitude", "relief", "approval", "optimism", "admiration"}
NEGATIVE = {"sadness", "grief", "fear", "anger", "disgust", "disappointment",
            "remorse", "nervousness", "annoyance", "disapproval"}

def classify_dynamic(curr_goe, curr_val, other_goe, other_val,
                     my_prev_vals, other_prev_vals):
    if other_goe is None:
        return "neutral"

    other_top = max(other_goe, key=other_goe.get)
    my_top    = max(curr_goe, key=curr_goe.get)

    my_sign    = 1 if curr_val > 0.15 else (-1 if curr_val < -0.15 else 0)
    other_sign = 1 if other_val > 0.15 else (-1 if other_val < -0.15 else 0)

    if other_top in NEGATIVE and my_top in SUPPORT and my_sign >= 0:
        return "empathy"

    if my_sign != 0 and other_sign != 0 and my_sign != other_sign:
        return "collision"

    my_trending_neg = (len(my_prev_vals) >= 1 and curr_val < my_prev_vals[-1] - 0.1)
    other_neg       = (other_val is not None and other_val < -0.1)
    if my_trending_neg and other_neg:
        return "escalation"

    if my_sign == other_sign and my_sign != 0:
        return "alignment"

    return "neutral"



msgs_a  = split_block(BLOCK_A)
msgs_b  = split_block(BLOCK_B)
combined: list[dict] = (
    parse_mixed_block(BLOCK_AB) if BLOCK_AB.strip()
    else interleave(msgs_a, msgs_b)
)

speakers = list(dict.fromkeys(m["speaker"] for m in combined))
spk_map  = {speakers[0]: "A", speakers[1]: "B"} if len(speakers) >= 2 else {}
for m in combined:
    m["speaker"] = spk_map.get(m["speaker"], m["speaker"])

print("═" * 60)
print(f"Conversation: {len(combined)} turns  |  A={len(msgs_a)} msgs  B={len(msgs_b)} msgs")
print("═" * 60)

histories: dict[str, list[dict]] = {"A": [], "B": []}
dynamic_counts: dict[str, int] = {}

for i, msg in enumerate(combined):
    sp    = msg["speaker"]
    other = "B" if sp == "A" else "A"
    text  = msg["text"]

    goe, val = analyze(text)

    other_last  = histories[other][-1] if histories[other] else None
    other_goe   = other_last["goe"] if other_last else None
    other_val   = other_last["val"] if other_last else None
    my_prev_vals   = [h["val"] for h in histories[sp]]
    other_prev_vals = [h["val"] for h in histories[other]]

    dyn  = classify_dynamic(goe, val, other_goe, other_val, my_prev_vals, other_prev_vals)
    top  = max(goe, key=goe.get)
    conf = goe[top]

    dynamic_counts[dyn] = dynamic_counts.get(dyn, 0) + 1

    print(f"[{sp}] Turn {i:02d} | val={val:+.2f} | {top:<18s} ({conf:.0%}) | ► {dyn}")
    print(f"       \"{text[:70]}{'...' if len(text)>70 else ''}\"")

    histories[sp].append({"goe": goe, "val": val})

print("\n" + "═" * 60)
print("SUMMARY")
print("─" * 40)

for spk in ("A", "B"):
    h = histories[spk]
    if h:
        vals = [x["val"] for x in h]
        avg  = sum(vals) / len(vals)
        trend = "↑" if vals[-1] > vals[0] + 0.1 else ("↓" if vals[-1] < vals[0] - 0.1 else "→")
        print(f"  {spk}: avg_valence={avg:+.2f}  trend={trend}  turns={len(h)}")

print(f"\n  Dynamics distribution:")
for dyn, cnt in sorted(dynamic_counts.items(), key=lambda x: -x[1]):
    bar = "█" * cnt
    print(f"    {dyn:<12s} {bar} ({cnt})")

dominant = max(dynamic_counts, key=dynamic_counts.get)
if dominant == "collision":
    print("\n  ⚠ Dominant pattern: EMOTIONAL COLLISION — misaligned responses")
elif dominant == "empathy":
    print("\n  ✓ Dominant pattern: EMPATHY — supportive dynamic")
elif dominant == "escalation":
    print("\n  ⚠ Dominant pattern: ESCALATION — both parties deteriorating")
elif dominant == "alignment":
    print("\n  ✓ Dominant pattern: ALIGNMENT — shared emotional direction")
