import logging                                                # stdlib logging (to quiet 3rd-party noise)
import os                                                      # paths + env vars
import sys                                                     # sys.path bootstrap
import time                                                    # latency timing in post_and_wait
import uuid                                                    # unique conversation ids

import pytest                                                  # test framework / fixtures

for _name in ("httpcore", "httpx", "huggingface_hub", "transformers", "urllib3", "filelock"):  # noisy libs
    logging.getLogger(_name).setLevel(logging.WARNING)         # silence their INFO/DEBUG chatter
logging.raiseExceptions = False                                # don't print logging errors on closed-stream at exit

_HERE = os.path.dirname(__file__)                              # qa_suite/ dir
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))             # repo root
_SVC = os.path.join(_ROOT, "central_responder_service")        # service dir holding meta_learner
for _p in (_HERE, _SVC, _ROOT):                                # make all three importable
    if _p not in sys.path:                                     # avoid duplicate entries
        sys.path.insert(0, _p)                                 # prepend so local modules win

DEFAULT_GOE_MODEL = "SamLowe/roberta-base-go_emotions"          # production GoEmotions checkpoint
DEFAULT_BERT_MODEL = "j-hartmann/emotion-english-distilroberta-base"  # Ekman BERT checkpoint
DEFAULT_MODEL_PATH = os.path.join(_SVC, "models", "meta_weights.pkl")  # trained meta-learner pkl

_RPT = {"v": 0, "phase": "", "buf": [], "tr": None}            # verbosity, phase, per-test action records, terminal writer


def pytest_configure(config):                                  # runs once at startup
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning:vaderSentiment.vaderSentiment")  # vader noise
    config.addinivalue_line("filterwarnings", "ignore::pytest.PytestConfigWarning")  # stale shared-ini option
    config.addinivalue_line("filterwarnings", "ignore::sklearn.exceptions.InconsistentVersionWarning")  # pkl sklearn ver
    config.addinivalue_line("markers", "slow: large data-driven battery — opt-in via -m slow")  # register marker
    _RPT["v"] = config.getoption("verbose")                   # -v level (3 == -vvv) enables the full action report
    _RPT["config"] = config                                   # keep config; terminalreporter isn't registered yet here


@pytest.hookimpl(wrapper=True)                                # wrap each test to scope its action buffer
def pytest_runtest_protocol(item, nextitem):                  # once per test
    _RPT["buf"], _RPT["phase"] = [], "setup"                  # fresh buffer; fixtures run in 'setup'
    return (yield)                                            # run setup/call/teardown, pass result through


@pytest.hookimpl(wrapper=True)                                # mark the call phase
def pytest_runtest_call(item):                                # the test body itself
    _RPT["phase"] = "call"                                    # only call-phase analyze() calls are recorded
    try:                                                      # run the test body
        return (yield)                                        # pass the result (or exception) through
    finally:                                                  # always reset afterwards
        _RPT["phase"] = "teardown"                            # stop recording


def _record(result, expected=None, ok=None):                 # build one action record from an analyze result
    return {"text": result["text"], "label": result["final_label"], "conf": result["final_conf"],  # io
            "vader": result["vader"], "vad": result["vad"], "top5": result["sorted_goe"][:5],  # signals
            "sarcasm": result.get("sarcasm", 0.0), "gate": result.get("gate_alpha"),  # extra signals
            "expected": expected, "ok": ok}                  # expected/verdict (set by io.check when known)


def _emit(a, i):                                             # format one action into report lines
    if a.get("live"):                                       # live (post_and_wait) record — fewer fields
        return [f"  [{i}] LIVE in : {a['text']!r}",         # the sent message
                f"      dominant : {a['label']}   (round-trip {a['latency']:.2f}s)"]  # broadcast emotion + latency
    g = a["gate"]                                           # gate weights (None in fallback)
    gate = ("[v,b,g,vad,ctx] " + " ".join(f"{x:.2f}" for x in g)) if g else "n/a (fallback mode)"  # gate string
    top5 = " · ".join(f"{l} {s:.2f}" for l, s in a["top5"])  # top-5 emotion distribution
    vd = a["vad"]                                           # (valence, arousal, dominance)
    lines = [f"  [{i}] input    : {a['text']!r}",           # the input text
             f"      predicted: {a['label']}  (conf {a['conf']:.2f})"]  # predicted label + confidence
    if a["expected"] is not None:                          # expected line, only when the test told us
        if a["ok"] is None:                                # note-only (no pass/fail verdict for this action)
            lines.append(f"      expected : {a['expected']}")  # show the note without a tick
        else:                                              # a real per-action verdict
            lines.append(f"      expected : {a['expected']}   {'✓' if a['ok'] else '✗ MISMATCH'}")  # tick/cross
    lines += [f"      vader    : compound {a['vader']['vader_compound']:+.2f}  "  # VADER sentiment
              f"neg {a['vader']['vader_neg']:.2f} neu {a['vader']['vader_neu']:.2f} pos {a['vader']['vader_pos']:.2f}",
              f"      vad      : valence {vd[0]:+.2f} arousal {vd[1]:+.2f} dominance {vd[2]:+.2f}",  # VAD lexicon
              f"      top-5    : {top5}",                    # top-5 emotions
              f"      signals  : sarcasm {a['sarcasm']:.2f}   gate {gate}"]  # sarcasm + gate weights
    return lines                                           # caller writes these


@pytest.hookimpl(trylast=True)                              # after the result is known
def pytest_runtest_logreport(report):                       # fired for setup/call/teardown of each test
    if _RPT["v"] < 3 or report.when != "call":              # only under -vvv, only the call phase
        return                                              # otherwise silent
    actions = _RPT["buf"]                                   # per-input records gathered during the test body
    summary = (report.capstdout or "").rstrip()            # the test's own printed summary (empty under -s)
    if not actions and not summary:                        # nothing to report
        return
    tr = _RPT["tr"]                                        # terminal writer (shows even without -s)
    if tr is None:                                         # fetch lazily — now terminalreporter is registered
        tr = _RPT["tr"] = _RPT["config"].pluginmanager.get_plugin("terminalreporter")
    if tr is None:                                         # still unavailable -> give up quietly
        return
    tr.write_line("")                                      # spacer
    tr.write_line(f"━━ {report.nodeid} — {report.outcome.upper()} · {len(actions)} action(s) ━━", bold=True)  # header
    for i, a in enumerate(actions, 1):                     # each recorded action
        for ln in _emit(a, i):                             # each formatted line
            tr.write_line(ln)                              # write outside capture
    for ln in summary.splitlines():                        # the aggregate test's printed summary, if any
        s = ln.strip()                                     # drop leaked library logger lines (timestamp/level)
        if (s[:1] == "[" and s[1:5].isdigit()) or any(f"] [{lv}" in ln for lv in ("INFO", "WARNING", "ERROR", "DEBUG", "CRITICAL")):
            continue                                       # skip logging noise, keep our [edge]/[coverage]/... summaries
        tr.write_line(f"  {ln}")                           # indented under the header


@pytest.fixture(scope="session")                               # build once per run
def meta_model():                                              # the trained meta-learner (or None)
    from meta_learner import load_meta_learner                 # imported lazily so live-only runs skip it
    return load_meta_learner(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))  # None -> rule-based fallback


@pytest.fixture(scope="session")                               # heavy HF models loaded once
def analyzers():                                               # returns (vader, bert, goe)
    try:                                                       # deps may be absent in some envs
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # lexicon sentiment
        from transformers import pipeline as hf_pipeline       # HF inference pipeline
    except Exception as e:                                     # missing libs
        pytest.skip(f"NLP deps unavailable: {e}")              # skip rather than error
    try:                                                       # model load may fail offline
        vader = SentimentIntensityAnalyzer()                   # VADER analyzer
        bert = hf_pipeline("text-classification",              # BERT Ekman classifier
                           model=os.environ.get("BERT_MODEL", DEFAULT_BERT_MODEL),
                           top_k=None, device=-1)              # all scores, CPU
        goe = hf_pipeline("text-classification",               # GoEmotions classifier
                          model=os.environ.get("GOE_MODEL", DEFAULT_GOE_MODEL),
                          top_k=None, device=-1)               # all 28 scores, CPU
    except Exception as e:                                     # no cache / no network
        pytest.skip(f"Could not load NLP models (offline/no cache?): {e}")  # skip
    return vader, bert, goe                                    # tuple of analyzers


@pytest.fixture(scope="session")                               # composed once
def analyze(analyzers, meta_model):                            # returns analyze(text) -> dict
    from shared.constants import EMOTION_LABELS, ML_DIM, VAD_DIM  # labels + feature-block offsets
    from meta_learner import build_feature_vector, predict_with_meta_learner  # real pipeline fns
    from training.eval_sentences import _entropy_norm          # reuse entropy helper
    vader, bert, goe = analyzers                               # unpack analyzers
    _vad0 = ML_DIM - VAD_DIM                                   # feature-vector index where the VAD block starts (39)

    _cache = {}                                                # memoize by text (generated convs reuse ~224 sentences)

    def _analyze(text):                                        # run one message through the stack (cached by text)
        key = text or ""                                       # cache key
        if key not in _cache:                                  # compute only on first sight of this text
            v = vader.polarity_scores(key)                     # VADER 4 scores (empty-safe)
            vader_scores = {f"vader_{k}": v[k] for k in ("neg", "neu", "pos", "compound")}  # prefix keys
            bert_scores = {r["label"]: r["score"] for r in bert(key[:512])[0]}  # 7 Ekman scores
            goe_scores = {r["label"]: r["score"] for r in goe(key[:512])[0]}    # 28 GoE scores
            fv = build_feature_vector({                        # build the 116-dim feature vector
                "vader": vader_scores, "basic_bert": bert_scores, "go_emotions": goe_scores})
            final_label, final_conf, all_scores, sarcasm, conflict, gate_alpha = \
                predict_with_meta_learner(meta_model, fv)      # run the meta-learner (or fallback)
            sorted_goe = sorted(goe_scores.items(), key=lambda kv: kv[1], reverse=True)  # GoE by score desc
            _cache[key] = {                                    # everything a test (and the report) might need
                "text": text, "vader": vader_scores, "compound": vader_scores["vader_compound"],  # input + valence
                "bert": bert_scores, "goe": goe_scores, "sorted_goe": sorted_goe, "fv": fv,  # raw model outputs
                "final_label": final_label, "final_conf": float(final_conf), "all_scores": all_scores,  # decision
                "gate_alpha": gate_alpha, "sarcasm": float(sarcasm),  # gate weights + sarcasm score
                "vad": tuple(float(x) for x in fv[0, _vad0:_vad0 + VAD_DIM]),  # VAD lexicon (valence/arousal/dominance)
                "entropy": _entropy_norm(goe_scores, EMOTION_LABELS)}  # normalised GoE entropy
        result = _cache[key]                                   # cached (or just-computed) analysis
        if _RPT["v"] >= 3 and _RPT["phase"] == "call":         # -vvv: record this action for the per-test report
            _RPT["buf"].append(_record(result))               # full block emitted after the test finishes
        return result                                          # hand back the analysis

    return _analyze                                            # hand the closure to tests


@pytest.fixture                                               # per-test helper that records expected + verdict
def io(analyze):                                              # tests call io(text, accept=...) instead of bare analyze
    def check(text, accept=None, family=None, valid=False, note=None):  # analyze + annotate the report block
        r = analyze(text)                                    # run it (auto-records the action under -vvv)
        ok, exp = None, note                                 # default verdict + expected text
        if accept is not None:                               # expect the label in an accepted set
            ok, exp = r["final_label"] in accept, "one of {" + ", ".join(sorted(accept)) + "}"
        elif family is not None:                             # expect the label in an emotion family
            ok, exp = r["final_label"] in family, "family {" + ", ".join(sorted(family)) + "}"
        elif valid:                                          # expect any of the 28 canonical labels
            from shared.constants import EMOTION_LABELS      # label list
            ok, exp = r["final_label"] in EMOTION_LABELS, "any valid label"
        if _RPT["v"] >= 3 and _RPT["phase"] == "call" and _RPT["buf"]:  # annotate the record analyze just added
            _RPT["buf"][-1]["expected"], _RPT["buf"][-1]["ok"] = exp, ok  # fill expected + verdict
        return r, ok                                         # result + verdict for the test to assert on
    return check                                             # hand the helper to the test


INGESTION_URL = os.environ.get("INGESTION_URL", "http://localhost:8000")  # ingestion REST door
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")         # redis host
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))           # redis port
REDIS_PASS = os.environ.get("REDIS_PASSWORD") or None          # redis password (None in dev)
API_KEY = os.environ.get("INTERNAL_API_KEY")                   # required X-API-Key header
WAIT_BUDGET_SEC = 25                                           # max wait for a result (cold start)


@pytest.fixture(scope="session")                               # one redis client per run
def redis_client():                                            # live redis handle; self-skips if down
    if not API_KEY:                                            # no key -> can't talk to the stack
        pytest.skip("INTERNAL_API_KEY not set — export it (e.g. `export $(grep -v '^#' .env | xargs)`).")
    try:                                                       # redis lib may be absent
        import redis                                           # redis client
    except Exception as e:
        pytest.skip(f"redis client lib unavailable: {e}")      # skip
    try:                                                       # stack may be down
        c = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS,
                        decode_responses=True, socket_connect_timeout=2)  # connect
        c.ping()                                               # verify reachable
    except Exception as e:
        pytest.skip(f"Redis at {REDIS_HOST}:{REDIS_PORT} unreachable — is the stack up? ({e})")  # skip
    return c                                                   # ready client


@pytest.fixture(scope="session")                               # POST+poll helper, built once
def post_and_wait(redis_client):                               # returns post_and_wait(text, ...) -> (entry, latency)
    import json                                                # parse pipeline_log
    import requests                                            # HTTP POST

    def _post_and_wait(text, conv_id=None, user_id="qa-suite-user"):  # send + await the result
        conv_id = conv_id or f"qa-{uuid.uuid4().hex[:8]}"      # default unique conversation
        try:                                                   # remember the stream tip
            tail = redis_client.xrevrange("emotion_stream", count=1)  # latest entry
            last_id = tail[0][0] if tail else "0-0"            # cursor (only see fresh entries)
        except Exception:
            last_id = "0-0"                                    # fall back to start
        t0 = time.time()                                       # start the latency clock
        resp = requests.post(                                  # POST the message to ingestion
            f"{INGESTION_URL}/messages",
            json={"conversation_id": conv_id, "user_id": user_id,
                  "text": text, "metadata": {"source": "qa_suite"}},
            headers={"X-API-Key": API_KEY}, timeout=5)
        assert resp.status_code in (200, 202), f"ingestion rejected: {resp.status_code} {resp.text}"  # accepted?
        message_id = resp.json()["message_id"]                 # id to match on
        deadline, matched = time.time() + WAIT_BUDGET_SEC, None  # budget + result holder
        while time.time() < deadline and matched is None:      # poll until found or timeout
            try:
                entries = redis_client.xread({"emotion_stream": last_id}, block=1000, count=20)  # read new
            except Exception:
                time.sleep(0.5)                                # transient error -> retry
                continue
            for _, msgs in entries or []:                      # iterate returned streams
                for entry_id, data in msgs:                    # iterate entries
                    last_id = entry_id                         # advance cursor
                    if data.get("message_id") == message_id:   # our message?
                        matched = data                         # capture it
                        break
                if matched:
                    break
        latency = time.time() - t0                             # round-trip time
        assert matched is not None, (                          # must have emerged
            f"No emotion_stream entry for message_id={message_id} within {WAIT_BUDGET_SEC}s.")
        if _RPT["v"] >= 3 and _RPT["phase"] == "call":         # -vvv: record this live action for the report
            _RPT["buf"].append({"live": True, "text": text,    # mark as a live record
                                "label": matched.get("dominant_emotion"), "latency": latency})  # dominant + latency
        return matched, latency                                # raw stream entry + latency

    return _post_and_wait                                      # hand the closure to tests
