# llm Knowledge Base
Last updated: 2026-06-21

## Architecture
Optional reasoning enrichment layer. Controlled by `LLM_PROVIDER` env var.

**Modes**:
- `RULE_BASED` (default): deterministic rule-based reasoning, zero latency, no API cost
- `OPENAI`: GPT-4o via OpenAI API
- `GROQ`: LLaMA/Mixtral via Groq API (fast inference)

**Position in pipeline**: reads emotion_stream, adds reasoning/explanation, re-publishes.

**When RULE_BASED**: produces template-based explanations from emotion labels + scores.
When LLM provider: sends enriched emotion context to LLM for natural-language explanation.

## Known Issues
- LLM path adds latency to the WebSocket response — only activate for non-real-time use cases.
- No retry/fallback if external LLM API is down (RULE_BASED is the implicit fallback).

## Improvement Queue
- **[Med]** Add circuit breaker: if external LLM fails 3× in a row, auto-switch to RULE_BASED.
- **[Low]** Cache LLM explanations for identical emotion label sets (Redis TTL=5min).

## Cross-Agent Dependencies
- Reads: emotion_stream from **meta_learner**
- Depends on: **infra** for LLM_PROVIDER env var + Redis

## Inter-Agent Requests (Pending)
*None*

## Recent History
*LLM mode not actively used — system running in RULE_BASED mode.*
