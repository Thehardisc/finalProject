"""
Generates demo conversations using the Anthropic Claude API.

Each conversation follows a defined emotional arc (trajectory).
Output: list of message strings ready to be sent through the pipeline.

Usage:
    from data.generator import ConversationGenerator
    gen = ConversationGenerator(api_key="sk-ant-...")
    messages = gen.generate(trajectory)
"""

import json
import time
import random
import logging
from typing import List

logger = logging.getLogger("generator")


SYSTEM_PROMPT = """\
You are a realistic dialogue writer. Your task is to write a sequence of messages \
that a single person might send in a chat — as if they are texting or messaging someone. \
Each message should be a separate line. Keep each message between 10 and 80 words. \
Do NOT number the messages. Do NOT include speaker labels. \
Just return the messages, one per line, nothing else. \
Make the emotional arc feel authentic and gradual — no sudden unexplained jumps. \
Occasionally use casual language, contractions, and natural speech patterns.\
"""


class ConversationGenerator:
    def __init__(self, api_key: str):
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    def generate(self, trajectory: dict, retries: int = 3) -> List[str]:
        """
        Generate a list of messages following the given emotional arc.
        Returns a list of message strings.
        """
        target_length = trajectory["length"]
        prompt = trajectory["prompt"]

        user_message = (
            f"{prompt}\n\n"
            f"Write exactly {target_length} messages. "
            f"Each on its own line. No numbering, no labels."
        )

        for attempt in range(retries):
            try:
                response = self._client.messages.create(
                    model="claude-opus-4-7",
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                raw = response.content[0].text.strip()
                messages = [line.strip() for line in raw.splitlines() if line.strip()]

                # Validate: we expect close to the target number of messages
                if len(messages) < target_length - 2:
                    logger.warning(
                        f"Got {len(messages)} messages for '{trajectory['type']}' "
                        f"(expected ~{target_length}). Retrying..."
                    )
                    time.sleep(2)
                    continue

                # Trim or keep as-is — small overruns are fine
                return messages[:target_length + 2]

            except Exception as e:
                logger.error(f"Generation attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))

        raise RuntimeError(
            f"Failed to generate conversation for trajectory '{trajectory['type']}' "
            f"after {retries} attempts."
        )

    def generate_batch(
        self,
        trajectory: dict,
        count: int,
        delay_between: float = 1.0,
    ) -> List[List[str]]:
        """
        Generate `count` distinct conversations for a single trajectory type.
        Returns a list of message-lists.
        """
        results = []
        for i in range(count):
            logger.info(
                f"  Generating [{i+1}/{count}] '{trajectory['type']}'..."
            )
            try:
                msgs = self.generate(trajectory)
                results.append(msgs)
            except RuntimeError as e:
                logger.error(f"  Skipped: {e}")
            if i < count - 1:
                time.sleep(delay_between)
        return results
