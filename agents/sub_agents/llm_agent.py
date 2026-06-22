from .base_agent import BaseAgent


class LLMAgent(BaseAgent):
    name = "llm"
    description = (
        "Optional LLM reasoning layer (llm_reasoning_service). "
        "Default mode: RULE_BASED (no external API calls, no latency). "
        "Supports: OPENAI, GROQ providers via LLM_PROVIDER env var. "
        "Enriches emotion predictions with natural-language explanations. "
        "Owns: llm_reasoning_service/."
    )
    key_files = [
        "llm_reasoning_service/main.py",
        "shared/constants.py",
    ]
