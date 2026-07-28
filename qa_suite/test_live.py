import json
import os
import time
import uuid

import pytest

import thresholds as T

pytestmark = pytest.mark.e2e

API_URL = os.environ.get("API_URL", "http://localhost:8001")


def _log(entry):
    return json.loads(entry["pipeline_log"])


def test_sequential_context(post_and_wait, redis_client):
    conv_id = f"qa-seq-{uuid.uuid4().hex[:8]}"
    turns = [
        "Hey, how's the project going?",
        "Wait, that's not what we agreed on at all.",
        "This is really starting to frustrate me.",
        "Honestly I'm furious about how this was handled.",
    ]
    valences = []
    for text in turns:
        entry, _ = post_and_wait(text, conv_id=conv_id)
        snap = _log(entry).get("context_snapshot", {})
        if "cur_valence" in snap:
            valences.append(float(snap["cur_valence"]))
    if len(valences) >= 2:
        assert len(set(round(v, 4) for v in valences)) > 1, (
            f"cur_valence identical across turns ({valences}) — context not tracked across messages")
    prior = redis_client.get(f"trajectory:{conv_id}:prior")
    if prior is not None:
        vals = json.loads(prior)
        assert len(vals) == 28, f"trajectory prior should be 28-dim, got {len(vals)}"


def test_sarcasm_signal(post_and_wait):
    entry, _ = post_and_wait("oh great, exactly what I needed today 🙄")
    log = _log(entry)
    assert "sarcasm_score" in log, "pipeline_log missing sarcasm_score"
    score = float(log["sarcasm_score"])
    assert 0.0 <= score <= 1.0, f"sarcasm_score out of range: {score}"
    if score > 0.5:
        assert log.get("inversion_applied") is True, (
            f"sarcasm_score {score:.2f} > 0.5 but inversion_applied not set")


def test_payload_contract(post_and_wait):
    entry, _ = post_and_wait("I'm so excited about how this is going! 🎉")
    assert "dominant_emotion" in entry, "missing dominant_emotion"
    assert "emotions" in entry, "missing emotions"
    log = _log(entry)
    for key in ("aggregated", "dominant_selected", "meta_confidence", "gate_weights_alpha",
                "sarcasm_score", "vad", "dynamics", "appraisal", "context_snapshot"):
        assert key in log, f"pipeline_log missing {key}"
    assert len(log["gate_weights_alpha"]) == 5, (
        f"gate_weights_alpha should be 5 elems [vader,bert,goe,vad,ctx], got {log['gate_weights_alpha']}")


def test_latency_budget(post_and_wait):
    post_and_wait("warm up the models")
    _, latency = post_and_wait("a quick warm message")
    print(f"\n[perf] warm POST→emotion_stream latency: {latency:.2f}s "
          f"(budget {T.PERF_LATENCY_BUDGET_SEC}s)")
    assert latency <= T.PERF_LATENCY_BUDGET_SEC, (
        f"warm latency {latency:.2f}s exceeds budget {T.PERF_LATENCY_BUDGET_SEC}s")


def test_throughput_burst(post_and_wait):
    conv_id = f"qa-burst-{uuid.uuid4().hex[:8]}"
    t0 = time.time()
    latencies = []
    for i in range(T.BURST_N):
        entry, latency = post_and_wait(f"burst message number {i}", conv_id=conv_id)
        assert "dominant_emotion" in entry
        latencies.append(latency)
    elapsed = time.time() - t0
    rate = T.BURST_N / elapsed if elapsed else 0.0
    print(f"\n[perf] processed {T.BURST_N} messages in {elapsed:.1f}s "
          f"({rate:.2f} msg/s, sequential)")

    half = len(latencies) // 2
    first_avg = sum(latencies[:half]) / half
    second_avg = sum(latencies[half:]) / (len(latencies) - half)
    print(f"[stability] avg latency first-half={first_avg:.2f}s  second-half={second_avg:.2f}s")
    assert second_avg <= first_avg * T.SOAK_REGRESSION_FACTOR + 0.5, (
        f"latency degraded under sustained sequential load: {first_avg:.2f}s -> {second_avg:.2f}s "
        f"(exceeds {T.SOAK_REGRESSION_FACTOR}x + 0.5s tolerance) — possible stability regression")


@pytest.fixture
def conversation(make_user):
    """Two registered users + a direct conversation between them, as the API models it."""
    session, user_id = make_user()
    _, other_user_id = make_user()
    resp = session.post(f"{API_URL}/conversations",
                         json={"user_id": user_id, "target_user_id": other_user_id}, timeout=5)
    assert resp.status_code == 200, f"create_conversation failed: {resp.status_code} {resp.text}"
    return session, user_id, resp.json()["conversation_id"]


def test_message_persists_and_is_retrievable(conversation, post_and_wait):
    session, user_id, conv_id = conversation
    text = "I'm really grateful you stuck with me through all of this."
    entry, _ = post_and_wait(text, conv_id=conv_id, user_id=user_id)
    message_id = entry["message_id"]

    deadline = time.time() + T.PERSIST_WAIT_SEC
    row = None
    while time.time() < deadline:
        resp = session.get(f"{API_URL}/conversation/{conv_id}/messages", timeout=5)
        assert resp.status_code == 200, f"messages fetch failed: {resp.status_code} {resp.text}"
        row = next((r for r in resp.json() if r["id"] == message_id), None)
        if row is not None and row.get("emotions"):
            break
        time.sleep(1)

    assert row is not None, (
        f"message {message_id} never appeared in persisted history for {conv_id} "
        f"within {T.PERSIST_WAIT_SEC}s — persistence_service may be lagging or down")
    assert row["content"] == text, f"persisted text mismatch: {row['content']!r} != {text!r}"
    assert row.get("emotions"), f"persisted row has no emotion analysis attached: {row}"


def test_conversation_insights_after_analyze(conversation, post_and_wait):
    session, user_id, conv_id = conversation
    turns = [
        "Everything about this launch has been perfect so far.",
        "Wait, the numbers just came in and they're way off.",
        "This is a disaster, I don't know how we recover from this.",
    ]
    for text in turns:
        post_and_wait(text, conv_id=conv_id, user_id=user_id)

    t0 = time.time()
    resp = session.post(f"{API_URL}/conversation/{conv_id}/analyze", timeout=5)
    assert resp.status_code == 200, f"analyze trigger failed: {resp.status_code} {resp.text}"

    deadline = time.time() + T.ANALYSIS_WAIT_SEC
    state = None
    while time.time() < deadline:
        r = session.get(f"{API_URL}/conversation/{conv_id}/emotional-state", timeout=5)
        assert r.status_code == 200, f"emotional-state fetch failed: {r.status_code} {r.text}"
        body = r.json()
        if body.get("status") == "active":
            state = body
            break
        time.sleep(1)
    latency = time.time() - t0

    assert state is not None, (
        f"conversation insights never became available for {conv_id} within {T.ANALYSIS_WAIT_SEC}s")
    print(f"\n[perf] analyze->emotional-state latency: {latency:.2f}s (budget {T.ANALYSIS_LATENCY_BUDGET_SEC}s)")
    assert latency <= T.ANALYSIS_LATENCY_BUDGET_SEC, (
        f"insights latency {latency:.2f}s exceeds budget {T.ANALYSIS_LATENCY_BUDGET_SEC}s")
    assert state["dominant_emotion"], "insights missing a dominant_emotion"
    assert state["trajectory"] in {"escalating", "de-escalating", "stable"}, (
        f"unexpected trajectory value: {state['trajectory']!r}")
    assert isinstance(state["arc"], list) and state["arc"], "insights arc should be a non-empty list"
