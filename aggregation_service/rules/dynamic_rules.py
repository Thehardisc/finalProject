"""
aggregation_service/rules/dynamic_rules.py — User-defined rule parsing and matching.

Supports runtime learning of conversation rules like:
  "when i say X it means Y"
  "define X as Y"

Rules are stored in Redis per-conversation and matched via word-boundary regex.
"""
import re
from typing import Tuple, Optional

from shared.utils.logger import get_logger

logger = get_logger("aggregation_service")

RULE_PATTERNS = [
    r"^when i say (.+?) it means (.+)$",
    r"^define (.+?) as (.+)$",
]


async def handle_dynamic_rules(
    conversation_id: str, text: str, r
) -> Tuple[Optional[str], Optional[str]]:
    """
    1. Parse text for new rules and store them in Redis.
    2. Check text against existing rules and return an override if matched.

    Returns (trigger, meaning) on match, or (None, None).
    """
    rule_key = f"conversation:{conversation_id}:rules"
    text_stripped = text.strip()

    # Learn new rule
    for pattern in RULE_PATTERNS:
        match = re.search(pattern, text_stripped, re.IGNORECASE)
        if match:
            trigger = match.group(1).strip().lower()
            meaning = match.group(2).strip().lower()
            logger.info(f"LEARNING RULE: '{trigger}' -> '{meaning}'")
            await r.hset(rule_key, trigger, meaning)
            await r.expire(rule_key, 86400 * 7)
            return None, None

    # Test against existing rules
    rules: dict = await r.hgetall(rule_key)
    logger.debug(f"Checking {len(rules)} rules for text: '{text_stripped}'")

    if rules:
        text_lower = text_stripped.lower()
        for trigger, meaning in rules.items():
            if re.search(rf"\b{re.escape(trigger)}\b", text_lower):
                logger.info(f"RULE MATCHED: '{trigger}' -> meaning '{meaning}'")
                return trigger, meaning

    return None, None
