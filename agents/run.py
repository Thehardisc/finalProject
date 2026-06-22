#!/usr/bin/env python3
"""
InnerLink Multi-Agent CLI
=========================
Usage:
  python agents/run.py "task description"
  python agents/run.py --agents nlp,meta_learner "check feature vector parity"
  python agents/run.py --status
  python agents/run.py --agent pipeline "debug stream backpressure"

Must be run from the project root:
  cd /path/to/finalProject-main
  python agents/run.py "..."
"""

import argparse
import sys
from pathlib import Path

# Add agents/ to path so imports work when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from head_agent import HeadAgent


def main():
    parser = argparse.ArgumentParser(description="InnerLink multi-agent orchestrator")
    parser.add_argument("task", nargs="?", help="Task description")
    parser.add_argument(
        "--agents", "-a",
        help="Comma-separated list of agents to force-activate (skips routing)",
    )
    parser.add_argument(
        "--agent",
        help="Run a single named agent directly",
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Print current knowledge base summary for all agents",
    )

    args = parser.parse_args()
    head = HeadAgent()

    if args.status:
        print(head.status())
        return

    if not args.task:
        parser.print_help()
        sys.exit(1)

    if args.agent:
        # Single-agent mode
        from sub_agents import ALL_AGENTS
        name = args.agent.strip()
        if name not in ALL_AGENTS:
            print(f"Unknown agent '{name}'. Available: {list(ALL_AGENTS)}")
            sys.exit(1)
        agent = ALL_AGENTS[name]()
        result = agent.run(args.task)
        print("\n=== FINDINGS ===")
        print(result.get("findings", ""))
        print("\n=== ACTIONS ===")
        print(result.get("actions", ""))
        print("\n=== ISSUES ===")
        print(result.get("issues", ""))
        print("\n=== IMPROVEMENTS ===")
        print(result.get("improvements", ""))
        if result.get("needs_help"):
            print("\n=== NEEDS HELP FROM ===")
            for n in result["needs_help"]:
                print(f"  → {n['agent']}: {n['request']}")
        return

    force = [a.strip() for a in args.agents.split(",")] if args.agents else None
    summary = head.run(args.task, force_agents=force)
    print("\n" + "=" * 60)
    print("SYNTHESIS")
    print("=" * 60)
    print(summary)


if __name__ == "__main__":
    main()
