"""Availability / partial-failure resilience.

Stops the optional context_engine_service container mid-suite to verify the
rest of the pipeline degrades gracefully (zero context vector, no crash) per
shared/module_registry.py, then restarts it. This mutates the live docker
stack, so it is opt-in only: skipped unless QA_ALLOW_FAULT_INJECTION=1 is set
explicitly, e.g.:

    QA_ALLOW_FAULT_INJECTION=1 ./run_qa.sh full qa_suite/test_availability.py
"""
import os
import shutil
import subprocess
import time

import pytest

pytestmark = pytest.mark.e2e

SERVICE = "context_engine_service"
RESTART_WAIT_SEC = 15


def _compose(*args):
    return subprocess.run(["docker", "compose", *args], capture_output=True, text=True, timeout=30)


@pytest.fixture
def context_engine_outage():
    if os.environ.get("QA_ALLOW_FAULT_INJECTION") != "1":
        pytest.skip(
            "fault-injection is opt-in — set QA_ALLOW_FAULT_INJECTION=1 to stop/start "
            f"{SERVICE} for this test")
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not available")

    stopped = _compose("stop", SERVICE)
    if stopped.returncode != 0:
        pytest.skip(f"could not stop {SERVICE} (is the stack up?): {stopped.stderr.strip()}")
    try:
        yield
    finally:
        started = _compose("start", SERVICE)
        if started.returncode != 0:
            pytest.fail(
                f"CRITICAL: failed to restart {SERVICE} after the outage test — "
                f"restore it manually with `docker compose start {SERVICE}`. "
                f"({started.stderr.strip()})")
        time.sleep(RESTART_WAIT_SEC)


def test_pipeline_survives_optional_module_outage(context_engine_outage, post_and_wait):
    entry, _ = post_and_wait("I can't believe how well this turned out, thank you all so much!")
    assert "dominant_emotion" in entry, (
        f"pipeline produced no result with {SERVICE} down — optional-module fallback broken")
    import json
    log = json.loads(entry["pipeline_log"])
    assert log.get("aggregated"), "pipeline_log.aggregated empty during optional-module outage"
