import asyncio
import sys
import os
import json
import time

# Add parent directory to path to import shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from shared.utils.redis_client import RedisClient
    from shared.utils.logger import get_logger
except ImportError:
    # setup fallback path
    import sys
    sys.path.append('/app')
    from shared.utils.redis_client import RedisClient
    from shared.utils.logger import get_logger

logger = get_logger("emojinet_service")

import emoji

# emojinet dataset
EMOJINET_DB = {
    "😂": {
        "definition": "face with tears of joy",
        "context": ["humor", "sarcasm", "laughter"],
        "sentiment": {"pos": 0.8, "neg": 0.0, "neu": 0.2},
        "emotions": {"joy": 0.9, "amusement": 0.95},
        "sarcasm_potential": 0.4
    },
    "😭": {
        "definition": "loudly crying face",
        "context": ["sadness", "overwhelmed", "laughter (slang)"],
        "sentiment": {"pos": 0.1, "neg": 0.8, "neu": 0.1},
        "emotions": {"sadness": 0.8, "grief": 0.6, "joy": 0.2}, 
        "sarcasm_potential": 0.6
    },
    "😍": {
        "definition": "smiling face with heart-eyes",
        "context": ["love", "adoration", "enthusiasm"],
        "sentiment": {"pos": 0.95, "neg": 0.0, "neu": 0.05},
        "emotions": {"love": 0.95, "admiration": 0.9, "joy": 0.8},
        "sarcasm_potential": 0.1
    },
    "🔥": {
        "definition": "fire",
        "context": ["hot", "trending", "excellent (slang)", "lit"],
        "sentiment": {"pos": 0.8, "neg": 0.0, "neu": 0.2},
        "emotions": {"excitement": 0.9, "admiration": 0.8, "joy": 0.7},
        "sarcasm_potential": 0.2
    },
    "💀": {
        "definition": "skull",
        "context": ["death", "danger", "dying of laughter (slang)"],
        "sentiment": {"pos": 0.4, "neg": 0.4, "neu": 0.2},
        "emotions": {"amusement": 0.9, "joy": 0.7, "fear": 0.1}, 
        "sarcasm_potential": 0.8
    },
    "🙃": {
        "definition": "upside-down face",
        "context": ["silly", "sarcasm", "irony", "passive-aggressive"],
        "sentiment": {"pos": 0.3, "neg": 0.4, "neu": 0.3},
        "emotions": {"amusement": 0.4, "annoyance": 0.6, "confusion": 0.3},
        "sarcasm_potential": 0.95
    },
    "🤔": {
        "definition": "thinking face",
        "context": ["thinking", "skepticism", "doubt"],
        "sentiment": {"pos": 0.1, "neg": 0.1, "neu": 0.8},
        "emotions": {"curiosity": 0.8, "confusion": 0.4, "disapproval": 0.2},
        "sarcasm_potential": 0.5
    },
    "🙄": {
        "definition": "face with rolling eyes",
        "context": ["disbelief", "annoyance", "impatience", "sarcasm"],
        "sentiment": {"pos": 0.0, "neg": 0.7, "neu": 0.3},
        "emotions": {"annoyance": 0.9, "disapproval": 0.8, "disgust": 0.5},
        "sarcasm_potential": 0.9
    },
    "💩": {
        "definition": "pile of poo",
        "context": ["crap", "silly", "bad"],
        "sentiment": {"pos": 0.2, "neg": 0.6, "neu": 0.2},
        "emotions": {"disgust": 0.8, "amusement": 0.5, "annoyance": 0.4},
        "sarcasm_potential": 0.3
    },
    "❤️": {
        "definition": "red heart",
        "context": ["love", "affection"],
        "sentiment": {"pos": 0.99, "neg": 0.0, "neu": 0.01},
        "emotions": {"love": 1.0, "caring": 0.9, "joy": 0.8},
        "sarcasm_potential": 0.05
    },
    "✨": {
         "definition": "sparkles",
         "context": ["magic", "beauty", "emphasis", "sarcasm (slang)"],
         "sentiment": {"pos": 0.7, "neg": 0.0, "neu": 0.3},
         "emotions": {"excitement": 0.7, "admiration": 0.6, "joy": 0.5},
         "sarcasm_potential": 0.7
    }
}

redis_client = RedisClient()
INPUT_STREAM = "message_stream"
GROUP_NAME = "emojinet_group"
CONSUMER_NAME = "emojinet_worker_1"
OUTPUT_STREAM = "partial_analysis_stream"

async def analyze_emojis(text):
    # Use distinct_emoji_list to handle multi-codepoint emojis correctly (like ❤️ which is \u2764\ufe0f)
    found_emojis = emoji.distinct_emoji_list(text)
    
    scores = {}
    aggregated_sentiment = {"pos": 0.0, "neg": 0.0, "neu": 0.0}
    emoji_count = 0

    if not found_emojis:
        return None

    logger.debug(f"Emojis found: {found_emojis}")

    for char in found_emojis:
        # Try direct match
        matched_entry = EMOJINET_DB.get(char)
        
        # If not found, try stripping variation selectors (\ufe0f)
        if not matched_entry:
            simple_char = char.replace('\ufe0f', '')
            matched_entry = EMOJINET_DB.get(simple_char)

        if matched_entry:
            entry = matched_entry
            
            # Aggregate Emotions
            for emo, score in entry.get("emotions", {}).items():
                scores[emo] = scores.get(emo, 0.0) + score
            
            # Aggregate Sentiment
            sent = entry.get("sentiment", {})
            aggregated_sentiment["pos"] += sent.get("pos", 0)
            aggregated_sentiment["neg"] += sent.get("neg", 0)
            aggregated_sentiment["neu"] += sent.get("neu", 0)
            
            emoji_count += 1
        else:
            # missing emojis handled later
            pass

    if emoji_count == 0:
        return None

    # Normalize scores
    final_scores = {k: v / emoji_count for k, v in scores.items()}
    
    # Explicitly add neutral=0.0 if not present to dilute other models' neutral scores
    if "neutral" not in final_scores:
        final_scores["neutral"] = 0.0
    
    return {
        "emotions": final_scores,
        "metadata": {
            "emoji_count": emoji_count,
            "emojis": found_emojis,
            "raw_sentiment": aggregated_sentiment
        }
    }

async def main():
    await redis_client.connect()
    r = redis_client.redis
    
    try:
        await r.xgroup_create(INPUT_STREAM, GROUP_NAME, mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"Error creating group: {e}")

    logger.info("EmojiNet Service started.")
    
    while True:
        try:
            streams = {INPUT_STREAM: ">"}
            messages = await r.xreadgroup(GROUP_NAME, CONSUMER_NAME, streams, count=10, block=2000)
            
            if messages:
                for stream, msgs in messages:
                    for message_id, data in msgs:
                        logger.debug(f"Emojinet received message {message_id}")
                        # Parse inputs
                        text = data.get("text", "")
                        id_val = data.get("message_id")
                        
                        # Analyze
                        start_time = time.time()
                        result = await analyze_emojis(text)
                        elapsed = (time.time() - start_time) * 1000
                        logger.debug(f"Emojinet inference took {elapsed:.2f}ms")
                        
                        if result:
                            # Publish Result
                            output_event = {
                                "message_id": id_val,
                                "model_name": "emojinet",
                                "scores": json.dumps(result["emotions"]),
                                "original_data": json.dumps(data),
                                "latency_ms": elapsed
                            }
                            await redis_client.publish_event(OUTPUT_STREAM, output_event)
                            
                            # Structured Stats Reporting
                            stats = {
                                "Message ID": id_val,
                                "Emoji Count": result["metadata"]["emoji_count"],
                                "Emojis Found": ", ".join(result["metadata"]["emojis"]),
                                "Dominant Map": max(result["emotions"].items(), key=lambda x: x[1])[0],
                                "Latency": f"{elapsed:.2f}ms"
                            }
                            logger.log_stats("EMOJINET INFERENCE", stats)
                        else:
                             # Publish empty
                             output_event = {
                                "message_id": id_val,
                                "model_name": "emojinet",
                                "scores": json.dumps({}),
                                "original_data": json.dumps(data),
                                "latency_ms": elapsed
                            }
                             await redis_client.publish_event(OUTPUT_STREAM, output_event)
                        
                        await r.xack(INPUT_STREAM, GROUP_NAME, message_id)
            
        except Exception as e:
            logger.log_exception("EMOJINET WORKER FATAL ERROR", e)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
