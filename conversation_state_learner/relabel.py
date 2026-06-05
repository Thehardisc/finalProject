"""
relabel.py — Re-label conversations with implicit emotion recognition via Claude API.

Reads from:  training_data/conversations.jsonl  (original — never modified)
Writes to:   training_data/conversations_relabeled.jsonl  (new file)

Usage:
  python relabel.py --dry-run              # prints full prompt for conv #1, no API call
  python relabel.py --limit 3              # process only first 3 conversations
  ANTHROPIC_API_KEY=sk-ant-... python relabel.py
  ANTHROPIC_API_KEY=sk-ant-... python relabel.py --model claude-sonnet-4-6
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

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

def call_claude(user_message: str, api_key: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
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
    parser.add_argument("--input",   default="training_data/conversations.jsonl")
    parser.add_argument("--output",  default="training_data/conversations_relabeled.jsonl")
    parser.add_argument("--model",   default="claude-haiku-4-5-20251001",
                        help="claude-haiku-4-5-20251001 (fast/cheap) or claude-sonnet-4-6 (higher quality)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the full prompt for the first conversation only. No API calls.")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Process only the first N conversations")
    args = parser.parse_args()

    input_path  = _HERE / args.input
    output_path = _HERE / args.output

    conversations = [
        json.loads(line)
        for line in input_path.read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        conversations = conversations[:args.limit]

    print(f"Conversations loaded: {len(conversations)}")
    print(f"Model: {args.model}")

    # ── Dry-run: show full prompt for first conversation ──────────────────────
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

    # ── Live run ──────────────────────────────────────────────────────────────
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
