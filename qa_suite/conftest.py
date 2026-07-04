import logging
import os
import sys
import time
import uuid

import pytest

for _name in ("httpcore", "httpx", "huggingface_hub", "transformers", "urllib3", "filelock"):
    logging.getLogger(_name).setLevel(logging.WARNING)
logging.raiseExceptions = False

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_SVC = os.path.join(_ROOT, "central_responder_service")
for _p in (_HERE, _SVC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_GOE_MODEL = "SamLowe/roberta-base-go_emotions"
DEFAULT_BERT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_MODEL_PATH = os.path.join(_SVC, "models", "meta_weights.pkl")

_RPT = {"v": 0, "phase": "", "buf": [], "tr": None}


def pytest_configure(config):
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning:vaderSentiment.vaderSentiment")
    config.addinivalue_line("filterwarnings", "ignore::pytest.PytestConfigWarning")
    config.addinivalue_line("filterwarnings", "ignore::sklearn.exceptions.InconsistentVersionWarning")
    config.addinivalue_line("markers", "slow: large data-driven battery — opt-in via -m slow")
    _RPT["v"] = config.getoption("verbose")
    _RPT["config"] = config


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    _RPT["buf"], _RPT["phase"] = [], "setup"
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    _RPT["phase"] = "call"
    try:
        return (yield)
    finally:
        _RPT["phase"] = "teardown"


def _record(result, expected=None, ok=None):
    return {"text": result["text"], "label": result["final_label"], "conf": result["final_conf"],
            "vader": result["vader"], "vad": result["vad"], "top5": result["sorted_goe"][:5],
            "sarcasm": result.get("sarcasm", 0.0), "gate": result.get("gate_alpha"),
            "expected": expected, "ok": ok}


def _emit(a, i):
    if a.get("live"):
        return [f"  [{i}] LIVE in : {a['text']!r}",
                f"      dominant : {a['label']}   (round-trip {a['latency']:.2f}s)"]
    g = a["gate"]
    gate = ("[v,b,g,vad,ctx] " + " ".join(f"{x:.2f}" for x in g)) if g else "n/a (fallback mode)"
    top5 = " · ".join(f"{l} {s:.2f}" for l, s in a["top5"])
    vd = a["vad"]
    lines = [f"  [{i}] input    : {a['text']!r}",
             f"      predicted: {a['label']}  (conf {a['conf']:.2f})"]
    if a["expected"] is not None:
        if a["ok"] is None:
            lines.append(f"      expected : {a['expected']}")
        else:
            lines.append(f"      expected : {a['expected']}   {'✓' if a['ok'] else '✗ MISMATCH'}")
    lines += [f"      vader    : compound {a['vader']['vader_compound']:+.2f}  "
              f"neg {a['vader']['vader_neg']:.2f} neu {a['vader']['vader_neu']:.2f} pos {a['vader']['vader_pos']:.2f}",
              f"      vad      : valence {vd[0]:+.2f} arousal {vd[1]:+.2f} dominance {vd[2]:+.2f}",
              f"      top-5    : {top5}",
              f"      signals  : sarcasm {a['sarcasm']:.2f}   gate {gate}"]
    return lines


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    if _RPT["v"] < 3 or report.when != "call":
        return
    actions = _RPT["buf"]
    summary = (report.capstdout or "").rstrip()
    if not actions and not summary:
        return
    tr = _RPT["tr"]
    if tr is None:
        tr = _RPT["tr"] = _RPT["config"].pluginmanager.get_plugin("terminalreporter")
    if tr is None:
        return
    tr.write_line("")
    tr.write_line(f"━━ {report.nodeid} — {report.outcome.upper()} · {len(actions)} action(s) ━━", bold=True)
    for i, a in enumerate(actions, 1):
        for ln in _emit(a, i):
            tr.write_line(ln)
    for ln in summary.splitlines():
        s = ln.strip()
        if (s[:1] == "[" and s[1:5].isdigit()) or any(f"] [{lv}" in ln for lv in ("INFO", "WARNING", "ERROR", "DEBUG", "CRITICAL")):
            continue
        tr.write_line(f"  {ln}")


@pytest.fixture(scope="session")
def meta_model():
    from meta_learner import load_meta_learner
    return load_meta_learner(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))


@pytest.fixture(scope="session")
def analyzers():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        from transformers import pipeline as hf_pipeline
    except Exception as e:
        pytest.skip(f"NLP deps unavailable: {e}")
    try:
        vader = SentimentIntensityAnalyzer()
        bert = hf_pipeline("text-classification",
                           model=os.environ.get("BERT_MODEL", DEFAULT_BERT_MODEL),
                           top_k=None, device=-1)
        goe = hf_pipeline("text-classification",
                          model=os.environ.get("GOE_MODEL", DEFAULT_GOE_MODEL),
                          top_k=None, device=-1)
    except Exception as e:
        pytest.skip(f"Could not load NLP models (offline/no cache?): {e}")
    return vader, bert, goe


@pytest.fixture(scope="session")
def analyze(analyzers, meta_model):
    from shared.constants import EMOTION_LABELS, ML_DIM, VAD_DIM
    from meta_learner import build_feature_vector, predict_with_meta_learner
    from training.eval_sentences import _entropy_norm
    vader, bert, goe = analyzers
    _vad0 = ML_DIM - VAD_DIM

    _cache = {}

    def _analyze(text):
        key = text or ""
        if key not in _cache:
            v = vader.polarity_scores(key)
            vader_scores = {f"vader_{k}": v[k] for k in ("neg", "neu", "pos", "compound")}
            bert_scores = {r["label"]: r["score"] for r in bert(key[:512])[0]}
            goe_scores = {r["label"]: r["score"] for r in goe(key[:512])[0]}
            fv = build_feature_vector({
                "vader": vader_scores, "basic_bert": bert_scores, "go_emotions": goe_scores})
            final_label, final_conf, all_scores, sarcasm, conflict, gate_alpha = \
                predict_with_meta_learner(meta_model, fv)
            sorted_goe = sorted(goe_scores.items(), key=lambda kv: kv[1], reverse=True)
            _cache[key] = {
                "text": text, "vader": vader_scores, "compound": vader_scores["vader_compound"],
                "bert": bert_scores, "goe": goe_scores, "sorted_goe": sorted_goe, "fv": fv,
                "final_label": final_label, "final_conf": float(final_conf), "all_scores": all_scores,
                "gate_alpha": gate_alpha, "sarcasm": float(sarcasm),
                "vad": tuple(float(x) for x in fv[0, _vad0:_vad0 + VAD_DIM]),
                "entropy": _entropy_norm(goe_scores, EMOTION_LABELS)}
        result = _cache[key]
        if _RPT["v"] >= 3 and _RPT["phase"] == "call":
            _RPT["buf"].append(_record(result))
        return result

    return _analyze


@pytest.fixture
def io(analyze):
    def check(text, accept=None, family=None, valid=False, note=None):
        r = analyze(text)
        ok, exp = None, note
        if accept is not None:
            ok, exp = r["final_label"] in accept, "one of {" + ", ".join(sorted(accept)) + "}"
        elif family is not None:
            ok, exp = r["final_label"] in family, "family {" + ", ".join(sorted(family)) + "}"
        elif valid:
            from shared.constants import EMOTION_LABELS
            ok, exp = r["final_label"] in EMOTION_LABELS, "any valid label"
        if _RPT["v"] >= 3 and _RPT["phase"] == "call" and _RPT["buf"]:
            _RPT["buf"][-1]["expected"], _RPT["buf"][-1]["ok"] = exp, ok
        return r, ok
    return check


INGESTION_URL = os.environ.get("INGESTION_URL", "http://localhost:8000")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASS = os.environ.get("REDIS_PASSWORD") or None
API_KEY = os.environ.get("INTERNAL_API_KEY")
WAIT_BUDGET_SEC = 25


@pytest.fixture(scope="session")
def redis_client():
    if not API_KEY:
        pytest.skip("INTERNAL_API_KEY not set — export it (e.g. `export $(grep -v '^#' .env | xargs)`).")
    try:
        import redis
    except Exception as e:
        pytest.skip(f"redis client lib unavailable: {e}")
    try:
        c = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS,
                        decode_responses=True, socket_connect_timeout=2)
        c.ping()
    except Exception as e:
        pytest.skip(f"Redis at {REDIS_HOST}:{REDIS_PORT} unreachable — is the stack up? ({e})")
    return c


@pytest.fixture(scope="session")
def post_and_wait(redis_client):

    import requests

    def _post_and_wait(text, conv_id=None, user_id="qa-suite-user"):
        conv_id = conv_id or f"qa-{uuid.uuid4().hex[:8]}"
        try:
            tail = redis_client.xrevrange("emotion_stream", count=1)
            last_id = tail[0][0] if tail else "0-0"
        except Exception:
            last_id = "0-0"
        t0 = time.time()
        resp = requests.post(
            f"{INGESTION_URL}/messages",
            json={"conversation_id": conv_id, "user_id": user_id,
                  "text": text, "metadata": {"source": "qa_suite"}},
            headers={"X-API-Key": API_KEY}, timeout=5)
        assert resp.status_code in (200, 202), f"ingestion rejected: {resp.status_code} {resp.text}"
        message_id = resp.json()["message_id"]
        deadline, matched = time.time() + WAIT_BUDGET_SEC, None
        while time.time() < deadline and matched is None:
            try:
                entries = redis_client.xread({"emotion_stream": last_id}, block=1000, count=20)
            except Exception:
                time.sleep(0.5)
                continue
            for _, msgs in entries or []:
                for entry_id, data in msgs:
                    last_id = entry_id
                    if data.get("message_id") == message_id:
                        matched = data
                        break
                if matched:
                    break
        latency = time.time() - t0
        assert matched is not None, (
            f"No emotion_stream entry for message_id={message_id} within {WAIT_BUDGET_SEC}s.")
        if _RPT["v"] >= 3 and _RPT["phase"] == "call":
            _RPT["buf"].append({"live": True, "text": text,
                                "label": matched.get("dominant_emotion"), "latency": latency})
        return matched, latency

    return _post_and_wait
