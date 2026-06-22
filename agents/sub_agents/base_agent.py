"""Base class for all InnerLink sub-agents."""
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import anthropic

PROJECT_ROOT = Path(__file__).parent.parent.parent
KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"

_MAX_FILE_CHARS = 5000
_MAX_FILES = 6


class BaseAgent:
    name: str = "base"
    description: str = ""
    key_files: list[str] = []

    def __init__(self):
        self.client = anthropic.Anthropic()
        KNOWLEDGE_DIR.mkdir(exist_ok=True)
        self.knowledge_path = KNOWLEDGE_DIR / f"{self.name}.md"
        if not self.knowledge_path.exists():
            self.knowledge_path.write_text(self._initial_knowledge())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_knowledge(self) -> str:
        return self.knowledge_path.read_text()

    def run(self, task: str, cross_context: Optional[str] = None) -> dict:
        """Analyse task in domain, update knowledge, return structured result."""
        knowledge = self.get_knowledge()
        code_ctx = self._read_key_files()
        prompt = self._build_prompt(task, knowledge, code_ctx, cross_context or "")

        response = self.client.messages.create(
            model="claude-opus-4-8",
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=[{"type": "text", "text": self._system_prompt(), "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )

        text = next((b.text for b in response.content if b.type == "text"), "")
        result = self._parse(text)

        if result.get("knowledge_update"):
            self.knowledge_path.write_text(result["knowledge_update"])

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        return (
            f"You are the **{self.name}** sub-agent of the InnerLink emotion analysis system.\n"
            f"Domain: {self.description}\n\n"
            "You are a deep expert in your slice of the codebase. "
            "When analysing a task:\n"
            "  • Read the code carefully and cite file:line when relevant.\n"
            "  • Suggest concrete, copy-paste-ready fixes (not pseudocode).\n"
            "  • Identify cross-cutting issues that another sub-agent must handle.\n\n"
            "Always structure your reply with these exact section headers (no extra text before each):\n"
            "FINDINGS:\nACTIONS:\nISSUES:\nIMPROVEMENTS:\nNEEDS_HELP:\nKNOWLEDGE_UPDATE:"
        )

    def _build_prompt(self, task: str, knowledge: str, code_ctx: str, cross_ctx: str) -> str:
        parts = [
            f"TASK: {task}",
            "",
            "MY KNOWLEDGE BASE:",
            knowledge,
            "",
            "MY DOMAIN CODE (excerpts):",
            code_ctx,
        ]
        if cross_ctx:
            parts += ["", "CONTEXT FROM OTHER AGENTS:", cross_ctx]
        parts += [
            "",
            "Respond with the six required sections. "
            "In KNOWLEDGE_UPDATE write the *full* updated markdown for my knowledge file.",
        ]
        return "\n".join(parts)

    def _read_key_files(self) -> str:
        snippets = []
        for rel in self.key_files[:_MAX_FILES]:
            path = PROJECT_ROOT / rel
            if not path.exists():
                snippets.append(f"--- {rel} [NOT FOUND] ---")
                continue
            try:
                text = path.read_text(errors="replace")[:_MAX_FILE_CHARS]
                snippets.append(f"--- {rel} ---\n{text}")
            except Exception as exc:
                snippets.append(f"--- {rel} [READ ERROR: {exc}] ---")
        return "\n\n".join(snippets)

    def _parse(self, text: str) -> dict:
        sections = {
            "findings": "",
            "actions": "",
            "issues": "",
            "improvements": "",
            "needs_help": [],
            "knowledge_update": "",
            "raw": text,
        }
        header_map = {
            "FINDINGS:": "findings",
            "ACTIONS:": "actions",
            "ISSUES:": "issues",
            "IMPROVEMENTS:": "improvements",
            "NEEDS_HELP:": "_needs_help_raw",
            "KNOWLEDGE_UPDATE:": "knowledge_update",
        }
        current_key = None
        buf: list[str] = []

        def flush():
            if current_key is None:
                return
            val = "\n".join(buf).strip()
            if current_key == "_needs_help_raw":
                sections["needs_help"] = self._parse_needs_help(val)
            else:
                sections[current_key] = val

        for line in text.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            matched = next((h for h in header_map if upper.startswith(h)), None)
            if matched:
                flush()
                current_key = header_map[matched]
                remainder = stripped[len(matched):].strip()
                buf = [remainder] if remainder else []
            elif current_key:
                buf.append(line)

        flush()
        return sections

    def _parse_needs_help(self, text: str) -> list:
        result = []
        for line in text.splitlines():
            line = line.strip().lstrip("-•* ")
            if not line or line.lower() in ("none", "n/a", "-"):
                continue
            if ":" in line:
                agent, _, req = line.partition(":")
                agent = agent.strip().lower().replace(" ", "_").replace("-", "_")
                req = req.strip()
                if agent and req:
                    result.append({"agent": agent, "request": req})
        return result

    def _initial_knowledge(self) -> str:
        return (
            f"# {self.name} Knowledge Base\n"
            f"Last updated: {datetime.now():%Y-%m-%d}\n\n"
            "## Architecture\n*To be populated on first run*\n\n"
            "## Known Issues\n*None recorded yet*\n\n"
            "## Improvement Queue\n*None recorded yet*\n\n"
            "## Cross-Agent Dependencies\n*To be mapped*\n\n"
            "## Inter-Agent Requests (Pending)\n*None*\n\n"
            "## Recent History\n*No history yet*\n"
        )
