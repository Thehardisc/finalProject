"""
collect.py — Main data collection CLI for the conversation state learner.

Phases:
  1. generate  — Use Claude API to write raw conversations per trajectory
  2. run       — Send raw conversations through the live emotion pipeline
  3. full      — Both phases in sequence (default)

Output files:
  raw_conversations/<trajectory_type>_<n>.json     — raw generated text
  training_data/conversations.jsonl                — enriched with pipeline results

Usage examples:
  python collect.py --phase generate --api-key sk-ant-... --count 5
  python collect.py --phase run --input raw_conversations/
  python collect.py --phase full --api-key sk-ant-... --count 15
  python collect.py --phase full --api-key sk-ant-... --trajectories escalating_conflict,celebration

Environment:
  ANTHROPIC_API_KEY — used if --api-key is not provided
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Allow running from the project root as well
sys.path.insert(0, str(Path(__file__).parent))

from data.trajectories import TRAJECTORIES, TRAJECTORY_TYPES, TOTAL_CONVERSATIONS
from data.generator import ConversationGenerator
from data.runner import PipelineRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect")

RAW_DIR      = Path(__file__).parent / "raw_conversations"
TRAINING_DIR = Path(__file__).parent / "training_data"
OUTPUT_FILE  = TRAINING_DIR / "conversations.jsonl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_raw(trajectory_type: str, idx: int, messages: List[str]):
    RAW_DIR.mkdir(exist_ok=True)
    path = RAW_DIR / f"{trajectory_type}_{idx:03d}.json"
    path.write_text(json.dumps({"trajectory_type": trajectory_type, "messages": messages}, indent=2))
    return path


def _load_raw(trajectory_filter: Optional[List[str]] = None) -> List[dict]:
    """Load all raw conversation JSON files from RAW_DIR."""
    records = []
    for f in sorted(RAW_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if trajectory_filter and data.get("trajectory_type") not in trajectory_filter:
                continue
            records.append(data)
        except Exception as e:
            logger.warning(f"Skipping malformed file {f.name}: {e}")
    return records


def _append_training(record: dict):
    TRAINING_DIR.mkdir(exist_ok=True)
    with OUTPUT_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _trajectory_by_type(t_type: str) -> Optional[dict]:
    return next((t for t in TRAJECTORIES if t["type"] == t_type), None)


# ── Phase 1: Generate ─────────────────────────────────────────────────────────

def phase_generate(
    api_key: str,
    trajectories_filter: Optional[List[str]],
    count_override: Optional[int],
    delay: float = 1.0,
):
    logger.info("=== PHASE 1: GENERATE ===")
    gen = ConversationGenerator(api_key=api_key)

    targets = [t for t in TRAJECTORIES
               if trajectories_filter is None or t["type"] in trajectories_filter]
    total = sum(count_override or t["count"] for t in targets)
    logger.info(f"Generating {total} conversations across {len(targets)} trajectories")

    generated = 0
    for trajectory in targets:
        n = count_override or trajectory["count"]
        logger.info(f"\n[{trajectory['type']}] — {n} conversations")

        # Figure out starting index (don't overwrite existing files)
        existing = sorted(RAW_DIR.glob(f"{trajectory['type']}_*.json"))
        start_idx = len(existing)

        batches = gen.generate_batch(trajectory, count=n, delay_between=delay)
        for i, messages in enumerate(batches):
            path = _save_raw(trajectory["type"], start_idx + i, messages)
            logger.info(f"  Saved {len(messages)} messages → {path.name}")
            generated += 1

    logger.info(f"\n✓ Generated {generated} conversation files in {RAW_DIR}/")


# ── Phase 2: Run ──────────────────────────────────────────────────────────────

def phase_run(
    base_url: str,
    trajectories_filter: Optional[List[str]],
    resume: bool = True,
):
    logger.info("=== PHASE 2: RUN PIPELINE ===")
    runner = PipelineRunner(base_url=base_url)

    raw_records = _load_raw(trajectories_filter)
    if not raw_records:
        logger.error(f"No raw conversations found in {RAW_DIR}. Run --phase generate first.")
        sys.exit(1)

    # Track which files have already been enriched (for resume)
    enriched_ids: set[str] = set()
    if resume and OUTPUT_FILE.exists():
        with OUTPUT_FILE.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    # Key: trajectory_type + first message text (unique enough)
                    key = r.get("trajectory_type", "") + "|" + (
                        r.get("messages", [{}])[0].get("text", "")[:50]
                    )
                    enriched_ids.add(key)
                except Exception:
                    pass
        if enriched_ids:
            logger.info(f"Resume mode: skipping {len(enriched_ids)} already-enriched conversations")

    total = len(raw_records)
    done  = 0

    for i, raw in enumerate(raw_records):
        t_type   = raw["trajectory_type"]
        messages = raw["messages"]

        resume_key = t_type + "|" + messages[0][:50]
        if resume and resume_key in enriched_ids:
            logger.info(f"[{i+1}/{total}] SKIP (already enriched): {t_type}")
            done += 1
            continue

        logger.info(f"[{i+1}/{total}] Running: {t_type} ({len(messages)} messages)")

        try:
            record = runner.run_conversation(t_type, messages)
            _append_training(record)
            done += 1
            logger.info(f"  ✓ Saved to {OUTPUT_FILE.name}")
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")

        # Brief pause between conversations to avoid overloading the pipeline
        if i < total - 1:
            time.sleep(3.0)

    logger.info(f"\n✓ Enriched {done}/{total} conversations → {OUTPUT_FILE}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect demo conversation training data for the conversation state learner."
    )
    parser.add_argument(
        "--phase",
        choices=["generate", "run", "full"],
        default="full",
        help="Which phase to execute (default: full = generate + run).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the emotion analysis API (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--trajectories",
        default=None,
        help=(
            "Comma-separated list of trajectory types to process. "
            f"Available: {', '.join(TRAJECTORY_TYPES)}"
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Override number of conversations per trajectory (default: per-trajectory setting).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode (re-run already-processed conversations).",
    )
    parser.add_argument(
        "--list-trajectories",
        action="store_true",
        help="Print available trajectory types and exit.",
    )

    args = parser.parse_args()

    if args.list_trajectories:
        print(f"\nAvailable trajectories ({len(TRAJECTORIES)} total, {TOTAL_CONVERSATIONS} conversations):\n")
        for t in TRAJECTORIES:
            print(f"  {t['type']:<28} {t['count']} convs  arc: {' → '.join(t['arc'][:4])}...")
        print()
        sys.exit(0)

    # Parse trajectory filter
    trajectories_filter = None
    if args.trajectories:
        trajectories_filter = [s.strip() for s in args.trajectories.split(",")]
        invalid = [t for t in trajectories_filter if t not in TRAJECTORY_TYPES]
        if invalid:
            logger.error(f"Unknown trajectory types: {invalid}")
            logger.error(f"Valid types: {TRAJECTORY_TYPES}")
            sys.exit(1)

    # Execute requested phase(s)
    if args.phase in ("generate", "full"):
        if not args.api_key:
            logger.error("--api-key or ANTHROPIC_API_KEY required for generation phase.")
            sys.exit(1)
        phase_generate(
            api_key=args.api_key,
            trajectories_filter=trajectories_filter,
            count_override=args.count,
        )

    if args.phase in ("run", "full"):
        phase_run(
            base_url=args.base_url,
            trajectories_filter=trajectories_filter,
            resume=not args.no_resume,
        )

    logger.info("\n=== COLLECTION COMPLETE ===")
    if OUTPUT_FILE.exists():
        lines = OUTPUT_FILE.read_text().strip().splitlines()
        logger.info(f"Total enriched conversations in training_data/conversations.jsonl: {len(lines)}")


if __name__ == "__main__":
    main()
