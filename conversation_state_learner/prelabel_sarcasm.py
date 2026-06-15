#!/usr/bin/env python3
"""
prelabel_sarcasm.py — Rule-based sarcasm prelabeler for conversations.jsonl.

Produces training_data/sarcasm_labels.jsonl compatible with
train_sarcasm_classifier.py, using pattern matching instead of Claude API.
"""

import json
import re
import sys
from pathlib import Path

ROOT      = Path(__file__).parent
INPUT     = ROOT / "training_data" / "conversations.jsonl"
OUTPUT    = ROOT / "training_data" / "sarcasm_labels.jsonl"

# ── Patterns ─────────────────────────────────────────────────────────────────

_SARCASM = [
    re.compile(r'\boh (great|wonderful|fantastic|perfect|brilliant|amazing|lovely|nice)\b', re.I),
    re.compile(r'\byeah (right|sure|of course|because that always)\b', re.I),
    re.compile(r'\bwhat could (possibly|ever) go wrong\b', re.I),
    re.compile(r'\bbecause that always works\b', re.I),
    re.compile(r'\bgreat(,| —| )? (just|another|so)\b', re.I),
    re.compile(r'\btotally (makes|makes sense|normal|fine)\b', re.I),
    re.compile(r'\bsure,? (because|that\'?s? always|like|right)\b', re.I),
    re.compile(r'\bwow,? (great|amazing|fantastic|shocking|really)\b', re.I),
    re.compile(r'\bso helpful\b', re.I),
    re.compile(r'\bgood luck with that\b', re.I),
    re.compile(r'\bthanks? for nothing\b', re.I),
    re.compile(r'\bthat went well\b', re.I),
    re.compile(r'\bjust (great|fantastic|wonderful|perfect)\b', re.I),
    re.compile(r'\bof course (you|it|that|he|she)\b', re.I),
    re.compile(r'\bbecause (obviously|clearly) that\b', re.I),
    re.compile(r'\bI\'m sure (that|it|this)\b', re.I),
    re.compile(r'\b(super|very|so|really|totally) (helpful|useful|great|fun|cool)\b', re.I),
    re.compile(r'\bgood(,)? right\b', re.I),
    re.compile(r'\blovely\b', re.I),
]

_PASSIVE_AGG = [
    re.compile(r'^(fine|whatever|sure|okay|ok|alright)\.?\s*$', re.I),
    re.compile(r'\bif (you|that\'?s?) (say so|think so|want|insist)\b', re.I),
    re.compile(r'\bdo whatever (you|u) (want|like|think)\b', re.I),
    re.compile(r'\bnot like (it|I) matter(s?)\b', re.I),
    re.compile(r'\bjust (forget|don\'t|never mind)\b', re.I),
    re.compile(r'\bno(,)? it\'?s? (fine|okay|ok)\b', re.I),
    re.compile(r'\bI\'?m fine\b', re.I),
    re.compile(r'\bI (said|told) (you|them|him|her)\b', re.I),
    re.compile(r'\bdoes(n\'t)? matter\b', re.I),
    re.compile(r'\bwhatever (you|u) (say|think|want|decide)\b', re.I),
    re.compile(r'\bdon\'?t (bother|worry about it)\b', re.I),
    re.compile(r'\bsure(,)? (go ahead|do it|why not)\b', re.I),
]

_IRONY = [
    re.compile(r'\bsurely\b.*\?', re.I),
    re.compile(r'\bobviously\b.*\?', re.I),
    re.compile(r'\bclearly\b.*\?', re.I),
    re.compile(r'\bthe (great|wonderful|amazing|brilliant) (plan|idea|solution)\b', re.I),
    re.compile(r'\bworks perfectly\b', re.I),
    re.compile(r'\bgoes (great|well|smoothly)\b', re.I),
    re.compile(r'\bperfect(ly)? (normal|fine|okay)\b', re.I),
    re.compile(r'\bthat\'?s? (surprising|shocking|unexpected)\b', re.I),
]

# Context-aware boost: short dismissive reply after conflict cue
_CONFLICT_CUES = re.compile(
    r'\b(wrong|stupid|idiot|hate|ridiculous|unbelievable|seriously|why would|stop|don\'t)\b', re.I
)
_SHORT_DISMISSIVE = re.compile(r'^(fine|okay|ok|sure|whatever|great|cool|alright)[\.\!]?\s*$', re.I)


def label_message(text: str, context_texts: list[str]) -> tuple[int, str, str]:
    """Return (is_sarcastic, subtype, reasoning)."""
    t = text.strip()

    for pat in _SARCASM:
        if pat.search(t):
            return 1, "sarcasm", f"matched sarcasm pattern: {pat.pattern}"

    for pat in _IRONY:
        if pat.search(t):
            return 1, "irony", f"matched irony pattern: {pat.pattern}"

    for pat in _PASSIVE_AGG:
        if pat.search(t):
            # Short dismissive after conflict cue is more reliable
            if _SHORT_DISMISSIVE.match(t):
                ctx = " ".join(context_texts[-2:])
                if _CONFLICT_CUES.search(ctx):
                    return 1, "passive_aggression", "short dismissive after conflict cue"
            else:
                return 1, "passive_aggression", f"matched passive_aggression pattern: {pat.pattern}"

    return 0, "none", "no pattern matched"


def _get_speaker(msg: dict) -> str:
    pipe = msg.get("pipeline") or {}
    message = pipe.get("message") or {}
    return message.get("display_name", "Speaker")


def main() -> None:
    conversations = [
        json.loads(l) for l in INPUT.read_text().splitlines() if l.strip()
    ]
    print(f"Conversations: {len(conversations)}")
    total_msgs = sum(len(c.get("messages", [])) for c in conversations)
    print(f"Messages: {total_msgs}")

    n_sarc = 0
    records = []

    for conv in conversations:
        cid  = conv["conversation_id"]
        msgs = conv.get("messages", [])

        for idx, msg in enumerate(msgs):
            text = msg.get("text", "").strip()
            ctx_msgs = msgs[max(0, idx - 3):idx]
            ctx_texts = [m.get("text", "").strip() for m in ctx_msgs]
            context = [
                {"speaker": _get_speaker(m), "text": m.get("text", "").strip()}
                for m in ctx_msgs
            ]

            is_sarc, subtype, reason = label_message(text, ctx_texts)
            n_sarc += is_sarc

            records.append({
                "conversation_id": cid,
                "message_index":   idx,
                "speaker":         _get_speaker(msg),
                "text":            text,
                "context":         context,
                "is_sarcastic":    is_sarc,
                "subtype":         subtype,
                "reasoning":       reason,
                "model":           "rule-based",
            })

    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Sarcastic: {n_sarc}/{total_msgs} ({100*n_sarc/total_msgs:.1f}%)")
    print(f"Written: {OUTPUT}")


if __name__ == "__main__":
    main()
