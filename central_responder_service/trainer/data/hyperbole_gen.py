"""One-shot generator: Claude-written hyperbole/internet-slang chat messages with GoEmotions labels.

Usage (host, from repo root, ANTHROPIC_API_KEY in env or .env):
    python3 central_responder_service/trainer/data/hyperbole_gen.py
Writes .cache/hyperbole_samples.csv (text,goemotions_label) which trainer/data/hyperbole.py consumes.
"""
import csv
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_ROOT))

from shared.constants import EMOTION_LABELS

MODEL   = os.environ.get("AI_DEMO_MODEL", "claude-sonnet-5")
OUT     = Path(os.environ.get("HYPERBOLE_OUT", _ROOT / ".cache" / "hyperbole_samples.csv"))
TARGETS = {
    "admiration": 60, "excitement": 60, "amusement": 60, "joy": 50, "love": 40,
    "approval": 30, "desire": 20, "surprise": 20,
    # literal-negative controls so the model keeps real distress intact
    "fear": 20, "sadness": 20, "anger": 20,
}

PROMPT = """Generate {n} realistic casual chat messages that a person would text a friend, ALL expressing the emotion "{label}".

{style}

Rules:
- 5 to 25 words each, texting tone, occasional emoji, no numbering
- Vary topics: food, places, music, pets, plans, work, sports, shows
- Output ONLY a JSON array of strings, nothing else."""

HYPERBOLE_STYLE = (
    "IMPORTANT: at least two thirds must use hyperbolic/dark figurative slang where NEGATIVE "
    "words carry POSITIVE intent — e.g. 'this view is killing me 😍', 'prepare to be personally "
    "victimized by how good this looks', 'I'm dead 😂', 'this cake is insane', 'she ate that', "
    "'it's criminal how good this is', 'I can't handle how cute this is'."
)
LITERAL_STYLE = (
    "IMPORTANT: these must be LITERAL, sincere expressions of the emotion (no sarcasm, no "
    "hyperbole-as-positivity) — real distress/anger/sadness, e.g. 'I'm honestly terrified about "
    "the surgery tomorrow'."
)
NEGATIVE = {"fear", "sadness", "anger"}


def _gen_label(client, label, n):
    style = LITERAL_STYLE if label in NEGATIVE else HYPERBOLE_STYLE
    resp = client.messages.create(
        model=MODEL, max_tokens=4000, thinking={"type": "disabled"},
        messages=[{"role": "user", "content": PROMPT.format(n=n, label=label, style=style)}],
    )
    text = next(b.text for b in resp.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    msgs = json.loads(text)
    return [(m.strip(), label) for m in msgs if isinstance(m, str) and 3 <= len(m.split()) <= 30]


def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import anthropic
    client = anthropic.Anthropic(timeout=180.0)
    rows, failures = [], []
    assert all(l in EMOTION_LABELS for l in TARGETS)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_gen_label, client, label, n): label for label, n in TARGETS.items()}
        for fut in as_completed(futs):
            label = futs[fut]
            try:
                got = fut.result()
                rows += got
                print(f"  {label:12s} +{len(got)}", flush=True)
            except Exception as e:
                failures.append(label)
                print(f"  {label:12s} FAILED: {e}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "goemotions_label"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} samples to {OUT}" + (f" (failed: {failures})" if failures else ""))


if __name__ == "__main__":
    main()
