"""Head Agent — orchestrates all InnerLink sub-agents."""

import json
import re
from pathlib import Path
from typing import Optional

import anthropic

from sub_agents import ALL_AGENTS

PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


class HeadAgent:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.agents = {name: cls() for name, cls in ALL_AGENTS.items()}

    # ------------------------------------------------------------------

    def run(self, task: str, force_agents: Optional[list] = None) -> str:
        print(f"\n{'='*60}")
        print(f"[HEAD] Task: {task}")
        print(f"{'='*60}")

        if force_agents:
            active = [a for a in force_agents if a in self.agents]
        else:
            active = self._route(task)

        print(f"[HEAD] Routing to: {active}")

        results: dict[str, dict] = {}
        for name in active:
            print(f"\n  ▶ {name}_agent...")
            results[name] = self.agents[name].run(task)
            _status = results[name].get("findings", "")[:80].replace("\n", " ")
            print(f"  ◀ {name}_agent: {_status}...")

        results = self._resolve_cross(task, results)

        summary = self._synthesise(task, results)
        print(f"\n{'='*60}\n[HEAD] Done.\n{'='*60}\n")
        return summary

    def status(self) -> str:
        """Print a status summary from every agent's knowledge base."""
        lines = []
        for name, agent in self.agents.items():
            kb = agent.get_knowledge()
            lines.append(f"\n{'─'*50}\n## {name.upper()}\n{kb[:600]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def _route(self, task: str) -> list[str]:
        agent_list = "\n".join(
            f"  {name}: {agent.description[:120]}"
            for name, agent in self.agents.items()
        )
        resp = self.client.messages.create(
            model="claude-opus-4-8",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": (
                    f"InnerLink sub-agents available:\n{agent_list}\n\n"
                    f"Task: {task}\n\n"
                    "Return JSON only: "
                    '{"agents": ["name1", "name2"], "reason": "..."}\n'
                    "Include only agents whose domain is directly relevant."
                )
            }],
        )
        text = resp.content[0].text if resp.content else "{}"
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                chosen = [a for a in data.get("agents", []) if a in self.agents]
                if chosen:
                    return chosen
            except json.JSONDecodeError:
                pass
        return list(self.agents.keys())

    def _resolve_cross(self, task: str, results: dict) -> dict:
        """Satisfy NEEDS_HELP requests by running the target agents with cross-context."""
        pending: list[dict] = []
        for from_name, res in results.items():
            for need in res.get("needs_help", []):
                target = need["agent"]
                if target in self.agents:
                    pending.append({"from": from_name, "to": target, "request": need["request"]})

        if not pending:
            return results

        print(f"\n[HEAD] Cross-agent requests: {[(p['from'],'→',p['to']) for p in pending]}")

        for req in pending:
            target = req["to"]
            cross_ctx = (
                f"Request from {req['from']}_agent:\n{req['request']}\n\n"
                f"Their findings:\n{results.get(req['from'], {}).get('findings', '')}"
            )
            if target not in results:
                print(f"  ▶ {target}_agent (cross-request)...")
                results[target] = self.agents[target].run(task, cross_context=cross_ctx)
            else:
                results[target]["cross_notes"] = results[target].get("cross_notes", "") + "\n" + cross_ctx

        return results

    def _synthesise(self, task: str, results: dict[str, dict]) -> str:
        parts = []
        for name, res in results.items():
            section = (
                f"### {name.upper()} AGENT\n"
                f"**Findings:** {res.get('findings','—')}\n"
                f"**Actions:** {res.get('actions','—')}\n"
                f"**Issues:** {res.get('issues','—')}\n"
                f"**Improvements:** {res.get('improvements','—')}"
            )
            parts.append(section)

        combined = "\n\n".join(parts)

        resp = self.client.messages.create(
            model="claude-opus-4-8",
            max_tokens=3000,
            thinking={"type": "adaptive"},
            messages=[{
                "role": "user",
                "content": (
                    f"Synthesise these sub-agent reports into an actionable summary.\n\n"
                    f"ORIGINAL TASK: {task}\n\n"
                    f"AGENT REPORTS:\n{combined}\n\n"
                    "Provide:\n"
                    "1. Executive summary (2-3 sentences)\n"
                    "2. Key findings (bullet list)\n"
                    "3. Prioritised recommended actions\n"
                    "4. Cross-cutting issues that span multiple agents"
                )
            }],
        )
        for block in resp.content:
            if block.type == "text":
                return block.text
        return combined
