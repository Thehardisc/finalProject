import asyncio
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.nlp_worker import run_nlp_worker
from shared.utils.logger import get_logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = get_logger("vader_service")


class VaderAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> dict:
        s = self.analyzer.polarity_scores(text)
        return {
            "vader_neg": s["neg"],
            "vader_neu": s["neu"],
            "vader_pos": s["pos"],
            "vader_compound": s["compound"],
        }


def _format_stats(data, scores, elapsed):
    text = data.get("text", "")
    return {
        "Message ID": data.get("message_id", "N/A"),
        "Text Snippet": (text[:30] + "...") if len(text) > 30 else text,
        "Latency": f"{elapsed:.2f}ms",
        "VADER Compound": scores["vader_compound"],
        "Positive": scores["vader_pos"],
        "Negative": scores["vader_neg"],
    }


if __name__ == "__main__":
    asyncio.run(run_nlp_worker(
        model_name="vader",
        group_name="vader_group",
        consumer_name="vader_worker_1",
        analyzer=VaderAnalyzer(),
        get_text=lambda d: d.get("text", "") or d.get("original_text", ""),
        format_stats=_format_stats,
        logger=logger,
    ))
