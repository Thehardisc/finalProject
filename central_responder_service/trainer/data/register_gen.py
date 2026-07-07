"""One-shot generator: Claude-written chat-register training samples with GoEmotions labels.

Each register targets a wording style or coverage gap of the live meta-learner:
  hyperbole — figurative/dark slang where negative words carry positive intent
              ("this view is killing me 😍", "I'm dead 😂")
  banter    — affectionate teasing between close friends where the surface wording
              is negative but the intent is positive ("rude but fair 😂")
  synthetic — first-person situational sentences (EmpatheticDialogues style) for
              the under-covered classes (neutral, relief, amusement, confusion,
              realization)
hyperbole/banter include literal-negative controls so real distress/friction
stays distinct.

Usage (host, from repo root, ANTHROPIC_API_KEY in env or .env):
    python3 central_responder_service/trainer/data/register_gen.py hyperbole|banter|synthetic|all
Writes .cache/<register>_samples.csv (text,goemotions_label), consumed by trainer/data/csv_sets.py.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_ROOT))

from shared.constants import EMOTION_LABELS

MODEL = os.environ.get("AI_DEMO_MODEL", "claude-sonnet-5")

PROMPT = """Generate {n} realistic casual chat messages that a person would text {audience}, ALL expressing the emotion "{label}".

{style}

Rules:
- {words} words each, texting tone, {emoji}, no numbering
- Vary topics: {topics}
- Output ONLY a JSON array of strings, nothing else."""

_HYPERBOLE_STYLE = (
    "IMPORTANT: at least two thirds must use hyperbolic/dark figurative slang where NEGATIVE "
    "words carry POSITIVE intent — e.g. 'this view is killing me 😍', 'prepare to be personally "
    "victimized by how good this looks', 'I'm dead 😂', 'this cake is insane', 'she ate that', "
    "'it's criminal how good this is', 'I can't handle how cute this is'."
)
_HYPERBOLE_LITERAL = (
    "IMPORTANT: these must be LITERAL, sincere expressions of the emotion (no sarcasm, no "
    "hyperbole-as-positivity) — real distress/anger/sadness, e.g. 'I'm honestly terrified about "
    "the surgery tomorrow'."
)
_BANTER_STYLE = (
    "IMPORTANT: at least two thirds must be PLAYFUL TEASING/BANTER where the surface wording "
    "contains negative tokens but the intent is clearly positive-affectionate — e.g. "
    "\"ok that's fair but also mean, joke's on you I already booked it 🤳\", \"rude but fair 😂\", "
    "\"you're insufferable and yet here I am inviting you\", \"don't get sappy on me, we leave at 6\", "
    "\"wow bold words from someone who lost last time\", \"I hate how right you are\". "
    "Mock offense, competitive ribbing, deadpan counter-teasing — but the underlying emotion "
    "must genuinely be \"{label}\"."
)
_BANTER_LITERAL = (
    "IMPORTANT: these must be LITERAL, sincere expressions of the emotion — real friction with "
    "a friend (no affectionate teasing, no irony) — e.g. \"you cancelled again, that actually "
    "sucks\", \"seriously, you never text back and it's getting old\"."
)
_SYNTHETIC_SYSTEM = (
    "You generate first-person situational sentences for emotion classification training data. "
    "Style: EmpatheticDialogues 'situation' field — 1-2 sentences describing a real-life context "
    "where someone feels a given emotion. "
    "Vary the settings: work, relationships, hobbies, discovery, everyday moments. "
    "Output one situation per line. No numbering, no bullets, no preamble."
)

REGISTERS: dict = {
    "hyperbole": {
        "audience": "a friend", "words": "5 to 25", "emoji": "occasional emoji", "max_words": 30,
        "topics": "food, places, music, pets, plans, work, sports, shows",
        "style_pos": _HYPERBOLE_STYLE, "style_neg": _HYPERBOLE_LITERAL,
        "negative": {"fear", "sadness", "anger"},
        "targets": {
            "admiration": 60, "excitement": 60, "amusement": 60, "joy": 50, "love": 40,
            "approval": 30, "desire": 20, "surprise": 20,
            "fear": 20, "sadness": 20, "anger": 20,
        },
    },
    "banter": {
        "audience": "a CLOSE FRIEND", "words": "5 to 30", "emoji": "emoji on roughly half of them",
        "max_words": 35,
        "topics": "travel plans, food, games, gym, shows, pets, group chats, inside jokes",
        "style_pos": _BANTER_STYLE, "style_neg": _BANTER_LITERAL,
        "negative": {"annoyance", "disappointment"},
        "targets": {
            "amusement": 80, "approval": 40, "excitement": 50, "love": 40,
            "pride": 30, "admiration": 30, "curiosity": 20,
            "annoyance": 25, "disappointment": 25,
        },
    },
    # situational sentences: cheaper model, line-based output, batched calls
    "synthetic": {
        "model": "claude-haiku-4-5-20251001",
        "system": _SYNTHETIC_SYSTEM,
        "parse": "lines", "batch_size": 250, "max_tokens": 4096,
        "prompt": "Generate {n} diverse first-person situational sentences expressing the emotion '{label}'.",
        "targets": {
            "neutral": 1069, "relief": 568, "amusement": 568,
            "confusion": 568, "realization": 568,
        },
    },
}


def _build_prompt(reg, label, n):
    if "prompt" in reg:
        return reg["prompt"].format(n=n, label=label)
    style = (reg["style_neg"] if label in reg["negative"] else reg["style_pos"]).format(label=label)
    return PROMPT.format(
        n=n, label=label, style=style, audience=reg["audience"],
        words=reg["words"], emoji=reg["emoji"], topics=reg["topics"],
    )


def _parse(reg, text):
    if reg.get("parse", "json") == "lines":
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return [l for l in lines if len(l) > 10 and not l.startswith("Here are")]
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    msgs = json.loads(text)
    return [m.strip() for m in msgs if isinstance(m, str) and 3 <= len(m.split()) <= reg["max_words"]]


def _gen_label(client, reg, label, target):
    rows, errors = [], 0
    batch_cap = reg.get("batch_size", target)
    while len(rows) < target:
        n = min(batch_cap, target - len(rows))
        kwargs = dict(
            model=reg.get("model", MODEL), max_tokens=reg.get("max_tokens", 4000),
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": _build_prompt(reg, label, n)}],
        )
        if reg.get("system"):
            kwargs["system"] = [{"type": "text", "text": reg["system"], "cache_control": {"type": "ephemeral"}}]
        try:
            resp = client.messages.create(**kwargs)
            text = next(b.text for b in resp.content if b.type == "text").strip()
            got  = _parse(reg, text)
            if not got:
                raise ValueError("batch produced no usable samples")
            rows += [(m, label) for m in got]
            errors = 0
            if len(rows) < target:
                time.sleep(1.5)
        except Exception as e:
            errors += 1
            if errors >= 3:
                raise
            print(f"  {label:14s} batch error ({e}) — retrying in 15s", flush=True)
            time.sleep(15)
    return rows[:target]


def _generate(name: str):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import anthropic
    reg = REGISTERS[name]
    out = Path(os.environ.get(f"{name.upper()}_OUT", _ROOT / ".cache" / f"{name}_samples.csv"))
    client = anthropic.Anthropic(timeout=180.0)
    rows, failures = [], []
    assert all(l in EMOTION_LABELS for l in reg["targets"])
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_gen_label, client, reg, label, n): label for label, n in reg["targets"].items()}
        for fut in as_completed(futs):
            label = futs[fut]
            try:
                got = fut.result()
                rows += got
                print(f"  {label:14s} +{len(got)}", flush=True)
            except Exception as e:
                failures.append(label)
                print(f"  {label:14s} FAILED: {e}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "goemotions_label"])
        w.writerows(rows)
    print(f"\n[{name}] Wrote {len(rows)} samples to {out}" + (f" (failed: {failures})" if failures else ""))


def main():
    choices = list(REGISTERS) + ["all"]
    pick = sys.argv[1] if len(sys.argv) > 1 else None
    if pick not in choices:
        print(f"Usage: python3 {Path(sys.argv[0]).name} {'|'.join(choices)}")
        sys.exit(1)
    for name in (REGISTERS if pick == "all" else [pick]):
        print(f"Generating '{name}' register (model={REGISTERS[name].get('model', MODEL)})...")
        _generate(name)


if __name__ == "__main__":
    main()
