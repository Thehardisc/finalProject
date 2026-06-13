"""
relabel.py — Re-label conversations via Claude API.

Modes
-----
  (default)  implicit emotion chunking — reads conversations.jsonl,
             writes conversations_relabeled.jsonl

  sarcasm    per-message binary sarcasm/passive-aggression labels —
             reads conversations.jsonl, writes sarcasm_labels.jsonl
             One flat JSONL record per message.

Usage:
  python relabel.py --dry-run                          # default mode dry-run
  python relabel.py --mode sarcasm --dry-run           # sarcasm mode dry-run
  python relabel.py --mode sarcasm --limit 50          # label first 50 conversations
  ANTHROPIC_API_KEY=sk-ant-... python relabel.py --mode sarcasm
  ANTHROPIC_API_KEY=sk-ant-... python relabel.py --mode sarcasm --model claude-haiku-4-5-20251001
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at recognizing implicit emotions and psychological states in human narrative text.

Your task is to analyze a conversation and label its underlying emotional dynamics.

CRITICAL RULE #1: INFER EMOTION FROM SITUATION, NOT FROM EXPLICIT WORDS.
- Ignore explicit emotion words (e.g., "I am sad", "crying", "angry").
- Instead, read the situation and infer what a human would naturally feel.
- Example: "I waited three hours and he never showed up" → Infer disappointment, hurt, and rejection (NOT neutral).
- Example: "I handed in my resignation today" → Read the context. Is it relief? Grief? Pride?
- Example: "She called me by my old name in front of everyone" → Infer embarrassment and hurt.

CRITICAL RULE #2: MARKOV-STYLE EMOTIONAL TRANSITIONS
Emotions do not happen in a vacuum. A conversation is a chain of emotional states where each sentence/message depends on the previous one.
- You must group consecutive messages into emotionally coherent "chunks".
- A new chunk starts when the emotional atmosphere meaningfully shifts.
- For each chunk, you must explain the "Emotional Chain" (Markov transition) — how the previous state led to the current state based on the narrative flow.

HOW TO SEGMENT & LABEL:
1. Divide the conversation into sequential chunks using message indices (0-based).
2. For each chunk, define the overarching "Mood".
3. Provide a score (0.0 to 1.0) for ALL 28 emotion categories based on the underlying situation.
4. Provide the "Transition Logic": How did the emotional state from the previous chunk evolve into this chunk? (For the first chunk, describe the initial state setup).

OUTPUT FORMAT:
You must output ONLY valid JSON. Do not include any markdown formatting, explanations, or text outside the JSON structure.

{
  "conversation_analysis": [
    {
      "chunk_index": 0,
      "message_indices": [0, 1],
      "mood": "<one of the 8 mood labels>",
      "transition_logic": "Initial state: The user establishes a baseline of [X] due to [Situation].",
      "implicit_reasoning": "The situation describes [Event], which implicitly triggers feelings of [Emotions] despite the absence of emotion words.",
      "emotions": {
        "admiration": 0.0,
        "amusement": 0.0,
        "anger": 0.0,
        "annoyance": 0.0,
        "approval": 0.0,
        "caring": 0.0,
        "confusion": 0.0,
        "curiosity": 0.0,
        "desire": 0.0,
        "disappointment": 0.0,
        "disapproval": 0.0,
        "disgust": 0.0,
        "embarrassment": 0.0,
        "excitement": 0.0,
        "fear": 0.0,
        "gratitude": 0.0,
        "grief": 0.0,
        "joy": 0.0,
        "love": 0.0,
        "nervousness": 0.0,
        "optimism": 0.0,
        "pride": 0.0,
        "realization": 0.0,
        "relief": 0.0,
        "remorse": 0.0,
        "sadness": 0.0,
        "surprise": 0.0,
        "neutral": 0.0
      }
    }
  ]
}

MOOD LABELS (Pick exactly one per chunk):
- anxious: tension, worry, anticipation of something bad
- melancholic: sadness, grief, loss, longing
- conflicted: internal struggle, ambivalence, mixed feelings
- warm: support, care, connection, empathy
- resolved: calm after difficulty, acceptance, relief
- joyful: celebration, excitement, gratitude, pride
- hostile: anger, frustration, resentment, passive aggression
- neutral: informational, no strong emotional tone

EMOTION LABELS (All 28 must be present in the JSON keys):
admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity,
desire, disappointment, disapproval, disgust, embarrassment, excitement, fear,
gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief,
remorse, sadness, surprise, neutral."""


# ── Sarcasm mode ─────────────────────────────────────────────────────────────

SARCASM_SYSTEM_PROMPT = """You are a linguist specializing in pragmatics and implied meaning.
You label individual chat messages for sarcasm and passive aggression.

DEFINITIONS
- Sarcasm: the speaker says the opposite of what they mean, often through ironic praise or mock agreement.
  Example: "Oh great, another Monday." → is_sarcastic=1, subtype=sarcasm
- Passive aggression: language that is superficially neutral or polite but masks hostility, contempt, or withdrawal.
  Example: "Fine. Whatever you think is best." (after a conflict) → is_sarcastic=1, subtype=passive_aggression
- Irony: a milder form where the literal and intended meaning diverge without clear hostility.
  Example: "Yeah because that always works out so well." → is_sarcastic=1, subtype=irony

TASK
Given a CONTEXT (up to 3 prior messages) and a TARGET message, decide whether the TARGET is
sarcastic, passively aggressive, or ironic.

CRITICAL RULES
1. Default to 0. Only label 1 when the mismatch between surface meaning and implied meaning is unambiguous.
2. Genuine apologies, sincere agreement, and factual statements are never sarcastic — even if the surrounding conversation is tense.
3. The label applies to the TARGET message only, not the conversation as a whole.
4. A short reply ("Okay.", "Fine.", "Sure.") is passive-aggressive ONLY if the context makes it clear the speaker is suppressing anger or hurt.

OUTPUT FORMAT — valid JSON only, no markdown, no explanation outside the JSON:
{"is_sarcastic": 0, "subtype": "none", "reasoning": "one sentence max"}

subtype must be exactly one of: "none" | "sarcasm" | "passive_aggression" | "irony"
is_sarcastic must be integer 0 or 1."""


def _get_speaker(msg: dict) -> str:
    return (
        msg.get("pipeline", {})
           .get("message", {})
           .get("display_name", "Speaker")
    )


def build_sarcasm_prompt(messages: list, target_index: int) -> str:
    context_start = max(0, target_index - 3)
    context_msgs  = messages[context_start:target_index]
    target_msg    = messages[target_index]

    lines = []
    if context_msgs:
        lines.append("CONTEXT (prior messages, oldest first):")
        for j, m in enumerate(context_msgs):
            lines.append(f"  [{j}] {_get_speaker(m)}: {m.get('text', '').strip()}")
    else:
        lines.append("CONTEXT: (this is the first message — no prior context)")

    lines.append("")
    lines.append("TARGET (label this message):")
    lines.append(f"  {_get_speaker(target_msg)}: {target_msg.get('text', '').strip()}")
    lines.append("")
    lines.append(
        "Output JSON only. is_sarcastic must be 0 or 1. "
        "subtype must be one of: none | sarcasm | passive_aggression | irony."
    )
    return "\n".join(lines)


def parse_sarcasm_response(raw: str, conv_id: str, msg_idx: int) -> Optional[dict]:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as e:
        print(
            f"  [WARN] JSON parse failed for {conv_id}[{msg_idx}]: {e}  "
            f"snippet={raw[:120]}",
            file=sys.stderr,
        )
        return None

    # Validate required fields
    if "is_sarcastic" not in parsed or "subtype" not in parsed:
        print(
            f"  [WARN] Missing fields in response for {conv_id}[{msg_idx}]: {parsed}",
            file=sys.stderr,
        )
        return None

    valid_subtypes = {"none", "sarcasm", "passive_aggression", "irony"}
    if parsed.get("subtype") not in valid_subtypes:
        parsed["subtype"] = "none"

    parsed["is_sarcastic"] = int(bool(parsed["is_sarcastic"]))
    return parsed


def run_sarcasm_mode(args, conversations: list, api_key: Optional[str]) -> None:
    output_path = _HERE / "training_data" / "sarcasm_labels.jsonl"
    output_path.parent.mkdir(exist_ok=True)

    # Build resume set: (conv_id, msg_idx) already in output file
    done: set[tuple[str, int]] = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.add((rec["conversation_id"], rec["message_index"]))
            except Exception:
                pass
    if done:
        print(f"Resuming — {len(done)} messages already labeled, skipping.")

    total_msgs = sum(len(c.get("messages", [])) for c in conversations)
    remaining  = total_msgs - len(done)
    print(f"Conversations: {len(conversations)} | Messages: {total_msgs} | To label: {remaining}")
    print(f"Model: {args.model}")
    print(f"Output: {output_path}")

    # Dry-run: show prompt for message index 2 of first conversation (likely has context)
    if args.dry_run:
        conv  = conversations[0]
        msgs  = conv.get("messages", [])
        idx   = min(2, len(msgs) - 1)
        prompt = build_sarcasm_prompt(msgs, idx)
        sep = "=" * 70
        print(f"\n{sep}")
        print("SYSTEM PROMPT:")
        print(sep)
        print(SARCASM_SYSTEM_PROMPT)
        print(f"\n{sep}")
        print(f"USER MESSAGE (conv={conv['conversation_id']}, msg_index={idx}):")
        print(sep)
        print(prompt)
        print(f"\n{sep}")
        print("No API calls made. Remove --dry-run to process.")
        return

    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    processed = skipped = failed = 0

    with output_path.open("a", encoding="utf-8") as out:
        for conv_i, conv in enumerate(conversations):
            cid  = conv["conversation_id"]
            msgs = conv.get("messages", [])

            for msg_i, msg in enumerate(msgs):
                key = (cid, msg_i)
                if key in done:
                    skipped += 1
                    continue

                label = (
                    f"[conv {conv_i+1}/{len(conversations)}, msg {msg_i}/{len(msgs)-1}] "
                    f"{cid}"
                )
                print(f"{label}...", end=" ", flush=True)

                try:
                    prompt = build_sarcasm_prompt(msgs, msg_i)
                    raw    = call_claude(
                        user_message=prompt,
                        api_key=api_key,
                        model=args.model,
                        system=SARCASM_SYSTEM_PROMPT,
                        max_tokens=128,
                    )
                    result = parse_sarcasm_response(raw, cid, msg_i)

                    if result is None:
                        failed += 1
                        print("FAILED (parse)")
                        continue

                    record = {
                        "conversation_id": cid,
                        "message_index":   msg_i,
                        "speaker":         _get_speaker(msg),
                        "text":            msg.get("text", "").strip(),
                        "context": [
                            {
                                "speaker": _get_speaker(messages[j]),
                                "text":    messages[j].get("text", "").strip(),
                            }
                            for j in range(max(0, msg_i - 3), msg_i)
                            for messages in [msgs]
                        ],
                        "is_sarcastic": result["is_sarcastic"],
                        "subtype":      result["subtype"],
                        "reasoning":    result.get("reasoning", ""),
                        "model":        args.model,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    processed += 1

                    flag = "[SARCASTIC]" if result["is_sarcastic"] else "ok"
                    print(f"{flag}  ({result['subtype']})")

                except Exception as e:
                    failed += 1
                    print(f"FAILED ({e})")

                time.sleep(0.4)

    total_done = processed + skipped
    print(f"\nDone.  labeled={processed}  skipped={skipped}  failed={failed}")
    print(f"Total in file: {total_done}")
    print(f"Output: {output_path}")


# ── Default (implicit emotion) mode ──────────────────────────────────────────

def build_user_message(conversation: dict) -> str:
    lines = [
        f"Trajectory type: {conversation.get('trajectory_type', 'unknown')}",
        f"Conversation ID: {conversation['conversation_id']}",
        "",
        "Messages:",
    ]
    for i, msg in enumerate(conversation['messages']):
        speaker = (
            msg.get('pipeline', {})
               .get('message', {})
               .get('display_name', 'Unknown')
        )
        text = msg.get('text', '')
        lines.append(f"  [{i}] {speaker}: {text}")
    lines.append("")
    lines.append(
        "Analyze the conversation above. "
        "Group messages into emotionally coherent chunks. "
        "Infer emotion from the situation described, not from emotion words."
    )
    return "\n".join(lines)


# ── Claude API call ───────────────────────────────────────────────────────────

def call_claude(
    user_message: str,
    api_key: str,
    model: str,
    system: str = SYSTEM_PROMPT,
    max_tokens: int = 2048,
) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def parse_response(raw: str, conversation_id: str):
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse failed for {conversation_id}: {e}", file=sys.stderr)
        print(f"  Raw response snippet: {raw[:300]}", file=sys.stderr)
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    default="implicit",
                        choices=["implicit", "sarcasm"],
                        help="implicit: emotion chunk relabeling (default); sarcasm: per-message binary labels")
    parser.add_argument("--input",   default="training_data/conversations.jsonl")
    parser.add_argument("--output",  default="training_data/conversations_relabeled.jsonl",
                        help="Only used in implicit mode. Sarcasm mode always writes to training_data/sarcasm_labels.jsonl")
    parser.add_argument("--model",   default="claude-haiku-4-5-20251001",
                        help="claude-haiku-4-5-20251001 (fast/cheap) or claude-sonnet-4-6 (higher quality)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print a sample prompt. No API calls.")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Process only the first N conversations")
    args = parser.parse_args()

    input_path = _HERE / args.input
    conversations = [
        json.loads(line)
        for line in input_path.read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        conversations = conversations[:args.limit]

    print(f"Conversations loaded: {len(conversations)}")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.mode == "sarcasm":
        api_key = os.environ.get("ANTHROPIC_API_KEY") if not args.dry_run else None
        run_sarcasm_mode(args, conversations, api_key)
        return

    # ── Default: implicit emotion mode ────────────────────────────────────────
    output_path = _HERE / args.output
    print(f"Model: {args.model}")

    if args.dry_run:
        conv     = conversations[0]
        user_msg = build_user_message(conv)
        sep = "=" * 70
        print(f"\n{sep}")
        print("SYSTEM PROMPT:")
        print(sep)
        print(SYSTEM_PROMPT)
        print(f"\n{sep}")
        print("USER MESSAGE:")
        print(sep)
        print(user_msg)
        print(f"\n{sep}")
        print("No API calls made. Remove --dry-run to process.")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(exist_ok=True)

    processed = 0
    failed    = 0

    with output_path.open("w", encoding="utf-8") as out:
        for i, conv in enumerate(conversations):
            cid  = conv["conversation_id"]
            traj = conv.get("trajectory_type", "?")
            print(f"[{i+1}/{len(conversations)}] {cid} ({traj})...", end=" ", flush=True)

            try:
                user_msg = build_user_message(conv)
                raw      = call_claude(user_msg, api_key, args.model)
                parsed   = parse_response(raw, cid)

                if parsed is None:
                    failed += 1
                    print("FAILED (parse error)")
                    continue

                enriched = dict(conv)
                enriched["relabeled_chunks"] = parsed.get("conversation_analysis", [])
                enriched["relabeled_model"]  = args.model

                out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                processed += 1
                n_chunks = len(parsed.get("conversation_analysis", []))
                print(f"OK  ({n_chunks} chunks)")

            except Exception as e:
                failed += 1
                print(f"FAILED ({e})")

            time.sleep(0.4)

    print(f"\nDone.  processed={processed}  failed={failed}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
