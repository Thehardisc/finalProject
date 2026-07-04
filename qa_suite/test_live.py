import json
import uuid

import pytest

import thresholds as T

pytestmark = pytest.mark.e2e


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
    import time
    conv_id = f"qa-burst-{uuid.uuid4().hex[:8]}"
    t0 = time.time()
    for i in range(T.BURST_N):
        entry, _ = post_and_wait(f"burst message number {i}", conv_id=conv_id)
        assert "dominant_emotion" in entry
    elapsed = time.time() - t0
    rate = T.BURST_N / elapsed if elapsed else 0.0
    print(f"\n[perf] processed {T.BURST_N} messages in {elapsed:.1f}s "
          f"({rate:.2f} msg/s, sequential)")
