"""
End-to-end smoke test: prove a message ingested at the front door emerges
from the central_responder with a meta-learner verdict.

Pipeline exercised:
    ingestion_service  →  preprocessing  →  vader/bert/goemotions
                                          →  central_responder (aggregate + predict)
                                          →  emotion_stream (Redis)

We subscribe to Redis directly rather than the WebSocket so we don't need to
implement the JWT auth dance — the WebSocket is just a thin bridge over the
same emotion_stream entries this test reads.

Prerequisites:
    1. `docker compose up -d` (stack is running)
    2. `.env` populated; this test reads INTERNAL_API_KEY from the environment
       (export it before running, e.g. `export $(grep -v '^#' .env | xargs)`).
    3. `pip install redis requests`

Run:
    pytest tests/test_e2e_smoke.py -v
"""
import os
import time
import uuid

import pytest
import requests
import redis


INGESTION_URL = os.environ.get("INGESTION_URL", "http://localhost:8000")
REDIS_HOST    = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT    = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASS    = os.environ.get("REDIS_PASSWORD") or None
API_KEY       = os.environ.get("INTERNAL_API_KEY")

WAIT_BUDGET_SEC = 25  # cold-start with model loading can be slow


@pytest.fixture(scope="module")
def redis_client():
    if not API_KEY:
        pytest.skip("INTERNAL_API_KEY not set in environment.")
    try:
        c = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS,
            decode_responses=True, socket_connect_timeout=2,
        )
        c.ping()
    except Exception as e:
        pytest.skip(f"Redis at {REDIS_HOST}:{REDIS_PORT} unreachable — is the stack up? ({e})")
    return c


def test_message_flows_through_pipeline(redis_client):
    """POST a message → it emerges from emotion_stream within the budget."""
    # Capture stream cursor BEFORE posting so we only see fresh entries.
    try:
        latest_entries = redis_client.xrevrange("emotion_stream", count=1)
        last_id = latest_entries[0][0] if latest_entries else "0-0"
    except Exception:
        last_id = "0-0"

    payload = {
        "conversation_id": f"e2e-smoke-{uuid.uuid4().hex[:8]}",
        "user_id":         "e2e-smoke-user",
        "text":            "I'm so excited about how this is going! 🎉",
        "metadata":        {"source": "e2e_smoke_test"},
    }

    resp = requests.post(
        f"{INGESTION_URL}/messages",
        json=payload,
        headers={"X-API-Key": API_KEY},
        timeout=5,
    )
    assert resp.status_code in (200, 202), f"ingestion rejected: {resp.status_code} {resp.text}"
    message_id = resp.json()["message_id"]

    # Poll the stream for our specific message_id.
    deadline = time.time() + WAIT_BUDGET_SEC
    matched_entry = None
    while time.time() < deadline:
        try:
            entries = redis_client.xread({"emotion_stream": last_id}, block=1000, count=20)
        except Exception:
            time.sleep(0.5)
            continue
        if not entries:
            continue
        for _, msgs in entries:
            for entry_id, data in msgs:
                last_id = entry_id
                if data.get("message_id") == message_id:
                    matched_entry = data
                    break
            if matched_entry:
                break
        if matched_entry:
            break

    assert matched_entry is not None, (
        f"No emotion_stream entry for message_id={message_id} within {WAIT_BUDGET_SEC}s. "
        f"Pipeline broken or starting up slowly?"
    )

    # Contract assertions: the published event has the fields the WebSocket
    # bridge in api_service expects to forward to the frontend.
    assert "dominant_emotion" in matched_entry,        "missing dominant_emotion"
    assert "emotions"          in matched_entry,       "missing emotions"
    assert "pipeline_log"      in matched_entry,       "missing pipeline_log"

    import json
    pipeline_log = json.loads(matched_entry["pipeline_log"])
    assert pipeline_log.get("aggregated"),             "pipeline_log.aggregated is empty"
    assert pipeline_log.get("dominant_selected"),      "pipeline_log.dominant_selected is empty"
    assert "decision_mode" in pipeline_log,            "pipeline_log.decision_mode missing"
