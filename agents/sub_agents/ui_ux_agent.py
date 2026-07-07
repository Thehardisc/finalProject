"""UI/UX Pro Max sub-agent — visual design expert, no design constraints."""
import subprocess
import base64

from typing import Optional



from .base_agent import BaseAgent, PROJECT_ROOT


class UiUxAgent(BaseAgent):
    name = "ui_ux"
    description = (
        "UI/UX design authority for the InnerLink frontend. "
        "Owns all visual design: color systems, typography, layout, animations, "
        "component design, user flows, accessibility, and emotional UX. "
        "Explicitly NOT constrained by the current design — free to propose radical redesigns. "
        "Uses screenshots to inspect the live UI before and after changes. "
        "Key files: frontend_service/src/pages/, frontend_service/src/components/, "
        "frontend_service/src/glass/, frontend_service/src/styles/."
    )
    key_files = [
        "frontend_service/src/pages/IGDashboard.jsx",
        "frontend_service/src/components/EmotionIntelligencePanel.jsx",
        "frontend_service/src/components/MessageItem.jsx",
        "frontend_service/src/glass/CrystalGlass-v2.css",
        "frontend_service/src/styles/design-system.css",
        "frontend_service/src/styles/tokens.css",
    ]

    _MODEL = "claude-opus-4-8"
    _MAX_TOKENS = 16000
    _THINKING_BUDGET = 10000

    def _system_prompt(self) -> str:
        return (
            "You are the **ui_ux** sub-agent of the InnerLink emotion analysis system — "
            "a world-class product designer and frontend engineer in one.\n\n"
            "## Your mandate\n"
            "- Design freedom: you are EXPLICITLY NOT constrained by the existing design. "
            "If the current design is mediocre, say so and propose something better.\n"
            "- Think like a top-tier designer (Figma/Vercel/Linear quality), not like a developer "
            "who happens to write CSS.\n"
            "- Consider motion, hierarchy, emotional resonance, whitespace, typography — "
            "not just 'does it work'.\n"
            "- Reference real design systems (Radix, shadcn, Material 3, Apple HIG) "
            "when relevant, but don't copy blindly.\n"
            "- Accessibility is non-negotiable: contrast ratios, keyboard nav, ARIA.\n\n"
            "## Technical constraints you must respect\n"
            "- Stack: React + Vite, static Nginx build. No SSR, no Next.js.\n"
            "- CSS entry: `index-v2.css` (NOT index.css). Design system: `CrystalGlass-v2.css`.\n"
            "- Rebuild required for any CSS/JSX change: `docker compose up --build frontend_service -d`.\n"
            "- Scroll rule: only `messagesContainerRef` scrolls — use `el.scrollTop = el.scrollHeight`.\n"
            "- Dark mode only (no light mode).\n\n"
            "## Output format\n"
            "Always structure your reply with these exact headers:\n"
            "FINDINGS:\nACTIONS:\nISSUES:\nIMPROVEMENTS:\nNEEDS_HELP:\nKNOWLEDGE_UPDATE:"
        )

    def run(self, task: str, cross_context: Optional[str] = None) -> dict:
        knowledge = self.get_knowledge()
        code_ctx = self._read_key_files()
        screenshot_ctx = self._take_screenshot()

        prompt_parts = [
            f"TASK: {task}",
            "",
            "MY KNOWLEDGE BASE:",
            knowledge,
            "",
            "MY DOMAIN CODE (excerpts):",
            code_ctx,
        ]
        if screenshot_ctx:
            prompt_parts += ["", "LIVE UI SCREENSHOT:", screenshot_ctx]
        if cross_context:
            prompt_parts += ["", "CONTEXT FROM OTHER AGENTS:", cross_context]
        prompt_parts += [
            "",
            "Respond with the six required sections. "
            "In KNOWLEDGE_UPDATE write the *full* updated markdown for my knowledge file. "
            "In ACTIONS, write production-ready JSX/CSS diffs, not pseudocode.",
        ]

        messages: list = [{"role": "user", "content": "\n".join(prompt_parts)}]

        response = self.client.messages.create(
            model=self._MODEL,
            max_tokens=self._MAX_TOKENS,
            thinking={"type": "enabled", "budget_tokens": self._THINKING_BUDGET},
            system=[{"type": "text", "text": self._system_prompt(), "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        )

        text = next((b.text for b in response.content if b.type == "text"), "")
        result = self._parse(text)
        if result.get("knowledge_update"):
            self.knowledge_path.write_text(result["knowledge_update"])
        return result

    def _take_screenshot(self) -> str:
        """Run screenshot.js via puppeteer and return base64 or description string."""
        script = PROJECT_ROOT / "scripts" / "screenshot.js"
        out_path = PROJECT_ROOT / "docs" / "frontend_screenshot.png"
        if not script.exists():
            return ""
        try:
            subprocess.run(
                ["node", str(script)],
                capture_output=True, timeout=30, cwd=str(PROJECT_ROOT),
            )
            if out_path.exists():
                data = base64.b64encode(out_path.read_bytes()).decode()
                return f"[screenshot captured — {len(data)//1024}KB base64 available]"
        except Exception:
            pass
        return ""
