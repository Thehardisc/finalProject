import json                                                    # parse pipeline_log / trajectory prior
import uuid                                                    # unique conversation ids

import pytest                                                  # test framework

import thresholds as T                                         # perf budget + burst size

pytestmark = pytest.mark.e2e                                   # whole module needs the live stack


def _log(entry):                                               # parse the pipeline_log JSON string
    return json.loads(entry["pipeline_log"])


def test_sequential_context(post_and_wait, redis_client):      # context evolves across turns
    conv_id = f"qa-seq-{uuid.uuid4().hex[:8]}"                 # one conversation
    turns = [                                                  # 4 escalating messages
        "Hey, how's the project going?",
        "Wait, that's not what we agreed on at all.",
        "This is really starting to frustrate me.",
        "Honestly I'm furious about how this was handled.",
    ]
    valences = []                                              # collected cur_valence per turn
    for text in turns:                                         # send each turn
        entry, _ = post_and_wait(text, conv_id=conv_id)       # POST + await result
        snap = _log(entry).get("context_snapshot", {})        # CDM snapshot (optional)
        if "cur_valence" in snap:                             # if context engine participated
            valences.append(float(snap["cur_valence"]))       # record valence
    if len(valences) >= 2:                                    # only assert if we have a series
        assert len(set(round(v, 4) for v in valences)) > 1, (  # valence must change across turns
            f"cur_valence identical across turns ({valences}) — context not tracked across messages")
    prior = redis_client.get(f"trajectory:{conv_id}:prior")   # trajectory prior (optional)
    if prior is not None:                                     # if the LSTM is present
        vals = json.loads(prior)                              # parse it
        assert len(vals) == 28, f"trajectory prior should be 28-dim, got {len(vals)}"  # 28-dim


def test_sarcasm_signal(post_and_wait):                        # sarcasm score present + consistent
    entry, _ = post_and_wait("oh great, exactly what I needed today 🙄")  # sarcastic line
    log = _log(entry)                                         # pipeline log
    assert "sarcasm_score" in log, "pipeline_log missing sarcasm_score"  # field exists
    score = float(log["sarcasm_score"])                       # the score
    assert 0.0 <= score <= 1.0, f"sarcasm_score out of range: {score}"  # in range
    if score > 0.5:                                           # if flagged sarcastic
        assert log.get("inversion_applied") is True, (        # inversion must be applied
            f"sarcasm_score {score:.2f} > 0.5 but inversion_applied not set")


def test_payload_contract(post_and_wait):                      # published event has all WS fields
    entry, _ = post_and_wait("I'm so excited about how this is going! 🎉")  # send a message
    assert "dominant_emotion" in entry, "missing dominant_emotion"  # top-level field
    assert "emotions" in entry, "missing emotions"            # top-level field
    log = _log(entry)                                         # pipeline log
    for key in ("aggregated", "dominant_selected", "meta_confidence", "gate_weights_alpha",  # required keys
                "sarcasm_score", "vad", "dynamics", "appraisal", "context_snapshot"):
        assert key in log, f"pipeline_log missing {key}"     # each must be present
    assert len(log["gate_weights_alpha"]) == 5, (             # gate vector is 5 elements
        f"gate_weights_alpha should be 5 elems [vader,bert,goe,vad,ctx], got {log['gate_weights_alpha']}")


def test_latency_budget(post_and_wait):                        # warm round-trip latency probe
    post_and_wait("warm up the models")                       # absorb cold-start
    _, latency = post_and_wait("a quick warm message")        # measure warm latency
    print(f"\n[perf] warm POST→emotion_stream latency: {latency:.2f}s "  # report
          f"(budget {T.PERF_LATENCY_BUDGET_SEC}s)")
    assert latency <= T.PERF_LATENCY_BUDGET_SEC, (            # soft budget
        f"warm latency {latency:.2f}s exceeds budget {T.PERF_LATENCY_BUDGET_SEC}s")


def test_throughput_burst(post_and_wait):                      # light sequential throughput probe
    import time                                               # timing
    conv_id = f"qa-burst-{uuid.uuid4().hex[:8]}"              # one conversation
    t0 = time.time()                                          # start clock
    for i in range(T.BURST_N):                                # send a burst
        entry, _ = post_and_wait(f"burst message number {i}", conv_id=conv_id)  # each message
        assert "dominant_emotion" in entry                   # each must emerge
    elapsed = time.time() - t0                                # total time
    rate = T.BURST_N / elapsed if elapsed else 0.0           # messages/sec
    print(f"\n[perf] processed {T.BURST_N} messages in {elapsed:.1f}s "  # report
          f"({rate:.2f} msg/s, sequential)")
