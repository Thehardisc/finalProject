
#!/bin/bash

set -e

echo ""
echo "=============================================================="
echo "    [System]  Emotion Analysis System - Launcher"
echo "=============================================================="
echo ""

API_KEY=$(grep "^INTERNAL_API_KEY=" .env | cut -d'=' -f2 | tr -d '"'\'' ')
API_KEY=${API_KEY:-dev-secret-key}

MAX_EMPATHETIC_SAMPLES_VAL=$(grep '^MAX_EMPATHETIC_SAMPLES=' .env 2>/dev/null | cut -d'=' -f2)
MIN_DB_SAMPLES_VAL=$(grep '^MIN_DB_SAMPLES=' .env 2>/dev/null | cut -d'=' -f2)
RETRAIN_INTERVAL_VAL=$(grep '^RETRAIN_INTERVAL_SECONDS=' .env 2>/dev/null | cut -d'=' -f2)
ACCURACY_GATE_VAL=$(grep '^ACCURACY_GATE=' .env 2>/dev/null | cut -d'=' -f2)
LLM_PROVIDER_VAL=$(grep '^LLM_PROVIDER=' .env 2>/dev/null | cut -d'=' -f2)
LOG_LEVEL_VAL=$(grep '^LOG_LEVEL=' .env 2>/dev/null | cut -d'=' -f2)
RATE_LIMIT_MAX_VAL=$(grep '^RATE_LIMIT_MAX=' .env 2>/dev/null | cut -d'=' -f2)

echo "[Config] Resolved environment from .env:"
echo "──────────────────────────────────────────────────────────"
echo "   LOG_LEVEL               = ${LOG_LEVEL_VAL:-INFO}"
echo "   MAX_EMPATHETIC_SAMPLES  = ${MAX_EMPATHETIC_SAMPLES_VAL:-25000}  (bootstrap dataset cap — runs once)"
echo "   MIN_DB_SAMPLES          = ${MIN_DB_SAMPLES_VAL:-50}  (min DB rows to trigger a continuous cycle)"
echo "   RETRAIN_INTERVAL_SECS   = ${RETRAIN_INTERVAL_VAL:-1800}s"
echo "   ACCURACY_GATE           = ${ACCURACY_GATE_VAL:-0.40}  (min test accuracy to deploy model)"
echo "   LLM_PROVIDER            = ${LLM_PROVIDER_VAL:-RULE_BASED}"
echo "   RATE_LIMIT_MAX          = ${RATE_LIMIT_MAX_VAL:-60} req/min per user"
echo "   API_KEY                 = ${API_KEY:0:6}...  (redacted)"
echo "──────────────────────────────────────────────────────────"
echo ""

echo "[Invariant] Checking feature-vector parity (inference vs trainer)..."
if command -v python3 &> /dev/null && python3 -c "import pytest, numpy" &> /dev/null; then
    if python3 -m pytest qa_suite/unit/test_feature_parity.py -q &> /tmp/parity_check.log; then
        echo "   [OK] Feature-vector parity holds."
    else
        echo "   [FAIL] Feature-vector parity VIOLATION — inference and trainer disagree."
        echo "          Launch aborted; fix build_feature_vector before starting."
        echo "          Details:"; tail -20 /tmp/parity_check.log | sed 's/^/          /'
        exit 1
    fi
else
    echo "   [SKIP] python3 + pytest/numpy not on host — parity not checked (runs in CI)."
fi
echo ""

echo "[Data] Checking Claude-generated register datasets (.cache/<register>_samples.csv)..."
ANTHROPIC_KEY=$(grep '^ANTHROPIC_API_KEY=' .env 2>/dev/null | cut -d'=' -f2 | tr -d '"'\'' ')
MISSING_REGISTERS=""
for REG in synthetic hyperbole banter; do
    [ -f ".cache/${REG}_samples.csv" ] || MISSING_REGISTERS="${MISSING_REGISTERS} ${REG}"
done
if [ -z "${MISSING_REGISTERS}" ]; then
    echo "   [OK] All register datasets present."
elif [ -n "${ANTHROPIC_KEY}" ] && command -v python3 &> /dev/null && python3 -c "import anthropic" &> /dev/null; then
    for REG in ${MISSING_REGISTERS}; do
        echo "   [INFO] Generating '${REG}' dataset via Claude API (one-time)..."
        if ANTHROPIC_API_KEY="${ANTHROPIC_KEY}" python3 central_responder_service/trainer/data/register_gen.py "${REG}"; then
            echo "   [OK] ${REG}_samples.csv created."
        else
            echo "   [Warn] ${REG} generation failed — trainer will skip that set."
        fi
    done
else
    echo "   [SKIP] Missing:${MISSING_REGISTERS} — ANTHROPIC_API_KEY not set (or anthropic pkg absent);"
    echo "          trainer will train without those sets. Generate later with:"
    echo "          python3 central_responder_service/trainer/data/register_gen.py all"
fi
echo ""

echo "[Models] Syncing trained artifacts to .cache/ (what you train is what you run)..."
SARC_SRC="central_responder_service/models/sarcasm_clf.pt"
SARC_DST=".cache/sarcasm_clf.pt"
if [ -f "${SARC_SRC}" ] && { [ ! -f "${SARC_DST}" ] || [ "${SARC_SRC}" -nt "${SARC_DST}" ]; }; then
    mkdir -p .cache
    cp "${SARC_SRC}" "${SARC_DST}"
    [ -f "central_responder_service/models/sarcasm_clf_config.json" ] && \
        cp "central_responder_service/models/sarcasm_clf_config.json" ".cache/sarcasm_clf_config.json"
    echo "   [OK] sarcasm_clf.pt → .cache/ (newer model deployed; service loads it on start)"
elif [ -f "${SARC_DST}" ]; then
    echo "   [OK] sarcasm classifier in .cache/ is up to date."
else
    echo "   [SKIP] no sarcasm_clf.pt trained yet — sarcasm_score stays 0.0"
    echo "          (train with: python3 conversation_state_learner/train_sarcasm_classifier.py)"
fi
echo ""

echo "[Search] Detecting host environment..."

if command -v uname &> /dev/null && [ "$(uname -s)" = "Darwin" ]; then
    if [ ! -f ".cache/meta_weights.pkl" ] && [ -f "scripts/train.sh" ]; then
        echo "   [INFO] No meta_weights.pkl found on Mac host."
        echo "   [INFO] Running local MPS-accelerated trainer before starting services..."
        ./scripts/train.sh
    fi
fi

if command -v nvidia-smi &> /dev/null; then
    echo "   [OK] NVIDIA GPU detected! Launching with GPU acceleration..."
    COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build"
else
    echo "   [INFO] No NVIDIA GPU detected. Launching in CPU mode..."
    COMPOSE_CMD="docker compose up -d --build"
fi

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOGDIR="logs/run_${TIMESTAMP}"
mkdir -p "${LOGDIR}"
mkdir -p logs/live && find logs/live -maxdepth 1 -type f -delete

echo ""
echo "[Launch] ${COMPOSE_CMD}"
echo ""
INIT_LOG="${LOGDIR}/init.log"
{
    echo "=============================================================="
    echo "    InnerLink Launcher — Init Log"
    echo "    Timestamp : ${TIMESTAMP}"
    echo "=============================================================="
    echo ""
    echo "[Config] Resolved environment from .env:"
    echo "──────────────────────────────────────────────────────────"
    echo "   LOG_LEVEL               = ${LOG_LEVEL_VAL:-INFO}"
    echo "   MAX_EMPATHETIC_SAMPLES  = ${MAX_EMPATHETIC_SAMPLES_VAL:-25000}"
    echo "   MIN_DB_SAMPLES          = ${MIN_DB_SAMPLES_VAL:-50}"
    echo "   RETRAIN_INTERVAL_SECS   = ${RETRAIN_INTERVAL_VAL:-1800}s"
    echo "   ACCURACY_GATE           = ${ACCURACY_GATE_VAL:-0.40}"
    echo "   LLM_PROVIDER            = ${LLM_PROVIDER_VAL:-RULE_BASED}"
    echo "   RATE_LIMIT_MAX          = ${RATE_LIMIT_MAX_VAL:-60} req/min"
    echo "   API_KEY                 = ${API_KEY:0:6}...  (redacted)"
    echo "──────────────────────────────────────────────────────────"
    echo ""
    echo "[Hardware] Compose command: ${COMPOSE_CMD}"
    echo ""
    echo "=============================================================="
    echo "[Docker Startup] build / create / start output"
    echo "=============================================================="
} > "${INIT_LOG}"
$COMPOSE_CMD 2>&1 | tee -a "${INIT_LOG}"
COMPOSE_RC=${PIPESTATUS[0]}
if [ "${COMPOSE_RC}" -ne 0 ]; then
    echo ""
    echo "   [Fail] docker compose exited with code ${COMPOSE_RC}. See ${INIT_LOG}"
    exit "${COMPOSE_RC}"
fi
echo "" | tee -a "${INIT_LOG}"

echo "[Logs] Per-service logs (real-time): logs/live/<service>.log"
echo "[Logs] Run artifacts: ${LOGDIR}/"

LAUNCH_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

_strip_ansi() { sed $'s/\033\\[[0-9;]*[A-Za-z]//g'; }


(
    SINCE="${LAUNCH_TIME}"
    > "${LOGDIR}/important.log"
    while true; do
        docker compose logs -f --since "${SINCE}" 2>&1 | _strip_ansi \
            | grep -v '"GET /health HTTP' | grep -v 'GET /health HTTP' \
            | grep -v 'OPTIONS /health' | grep -v '"GET / HTTP/1.1" 200' \
            >> "${LOGDIR}/important.log" || true
        SINCE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        sleep 2
    done
) &

(
    SINCE="${LAUNCH_TIME}"
    > "${LOGDIR}/errors.log"
    while true; do
        docker compose logs -f --since "${SINCE}" 2>&1 | _strip_ansi \
            | grep -E '\[ERROR\s*\]|\[CRITICAL\]|\[WARNING\s*\]|Traceback|Exception:|FATAL|CRASH|ROLLBACK' \
            | grep -v 'uvicorn.error' \
            >> "${LOGDIR}/errors.log" || true
        SINCE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        sleep 2
    done
) &

(
    SINCE="${LAUNCH_TIME}"
    > "${LOGDIR}/all.log"
    while true; do
        docker compose logs -f --since "${SINCE}" 2>&1 | _strip_ansi >> "${LOGDIR}/all.log" || true
        SINCE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        sleep 2
    done
) &
LOGFILE="${LOGDIR}/all.log"

echo ""
echo "[Wait] Waiting for all containers to start..."

EXPECTED_SERVICES=$(docker compose config --services | wc -l | tr -d ' ')
MAX_WAIT=120  # seconds
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    RUNNING=$(docker compose ps --status running --format json 2>/dev/null | grep -c '"State":"running"' || docker compose ps --status running 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')
    
    if [ "$RUNNING" -ge "$EXPECTED_SERVICES" ]; then
        echo "   [OK] All $EXPECTED_SERVICES containers are running!"
        break
    fi
    
    echo "   [Wait] $RUNNING / $EXPECTED_SERVICES containers running... (${ELAPSED}s)"
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "   [Warn] Timeout! Not all containers started within ${MAX_WAIT}s."
    echo "   Check logs: $LOGFILE"
    docker compose ps
    exit 1
fi

echo ""
echo "[Health] Running service health checks..."
PASS=0
FAIL=0
FAILED_CHECKS=""   # accumulates the name+reason of every failed check for the init report

check_service() {
    local name=$1
    local url=$2
    local max_retries=${3:-10}
    local attempt=0

    while [ $attempt -lt $max_retries ]; do
        if curl -s --max-time 3 -H "X-API-Key: $API_KEY" "$url" > /dev/null 2>&1; then
            echo "   [OK] $name - responding"
            PASS=$((PASS + 1))
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done

    echo "   [Fail] $name - NOT responding at $url"
    FAIL=$((FAIL + 1))
    FAILED_CHECKS="${FAILED_CHECKS}   - ${name}: no response at ${url} after ${max_retries} tries"$'\n'
    return 1
}

if docker compose exec -T redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
    echo "   [OK] Redis - PONG"
    PASS=$((PASS + 1))
else
    echo "   [Fail] Redis - not responding"
    FAIL=$((FAIL + 1))
    FAILED_CHECKS="${FAILED_CHECKS}   - Redis: ping did not return PONG"$'\n'
fi

if docker compose exec -T db pg_isready -U user -d emotion_db 2>/dev/null | grep -q "accepting"; then
    echo "   [OK] PostgreSQL - accepting connections"
    PASS=$((PASS + 1))
else
    echo "   [Fail] PostgreSQL - not ready"
    FAIL=$((FAIL + 1))
    FAILED_CHECKS="${FAILED_CHECKS}   - PostgreSQL: not accepting connections"$'\n'
fi

check_service "Ingestion Service (API :8000)" "http://localhost:8000/health"
check_service "API Service (WebSocket :8001)" "http://localhost:8001/conversation/conv-1/state"
check_service "Frontend (UI :5173)" "http://localhost:5173"

# A freshly (re)built central_responder container needs ~40s to import torch and
# load meta_weights.pkl before it logs its mode — poll instead of a one-shot grep.
ML_DEADLINE=90
ML_ELAPSED=0
ML_MODE=""
while [ $ML_ELAPSED -lt $ML_DEADLINE ]; do
    ML_LAST=$(docker compose logs central_responder_service 2>/dev/null \
        | grep -E "Running in META-LEARNER mode|Falling back to Rule-Based Aggregation" | tail -1)
    if [ -n "${ML_LAST}" ]; then
        case "${ML_LAST}" in
            *"META-LEARNER mode"*) ML_MODE="meta" ;;
            *)                     ML_MODE="rule" ;;
        esac
        break
    fi
    sleep 3
    ML_ELAPSED=$((ML_ELAPSED + 3))
done

if [ "${ML_MODE}" = "meta" ]; then
    echo "   [OK] Meta-Learner - loaded and active"
    PASS=$((PASS + 1))
elif [ "${ML_MODE}" = "rule" ]; then
    echo "   [Fail] Meta-Learner - service fell back to RULE-BASED mode (no compatible meta_weights.pkl)"
    FAIL=$((FAIL + 1))
    FAILED_CHECKS="${FAILED_CHECKS}   - Meta-Learner: fell back to rule-based aggregation (no compatible meta_weights.pkl; trainer will hot-reload once a model passes the gate)"$'\n'
else
    echo "   [Fail] Meta-Learner - no mode line logged within ${ML_DEADLINE}s (check central_responder_service logs)"
    FAIL=$((FAIL + 1))
    FAILED_CHECKS="${FAILED_CHECKS}   - Meta-Learner: neither 'META-LEARNER mode' nor rule-based fallback logged within ${ML_DEADLINE}s"$'\n'
fi

echo "[Test] Running end-to-end pipeline test..."

PIPE_RETRIES=10
PIPE_ATTEMPT=0
HTTP_CODE=""
while [ $PIPE_ATTEMPT -lt $PIPE_RETRIES ]; do
    RESPONSE=$(curl -s -w "\n%{http_code}" --max-time 5 -X POST http://localhost:8000/messages \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d '{"conversation_id": "healthcheck", "user_id": "system", "text": "I am happy!"}' 2>/dev/null)
    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    BODY=$(echo "$RESPONSE" | head -1)
    [ "$HTTP_CODE" = "200" ] && break
    PIPE_ATTEMPT=$((PIPE_ATTEMPT + 1))
    sleep 2
done

if [ "$HTTP_CODE" = "200" ]; then
    echo "   [OK] Pipeline test - message accepted (HTTP 200)"
    PASS=$((PASS + 1))
else
    echo "   [Fail] Pipeline test - failed (HTTP ${HTTP_CODE:-no-response}) after ${PIPE_RETRIES} tries"
    FAIL=$((FAIL + 1))
    FAILED_CHECKS="${FAILED_CHECKS}   - Pipeline test: POST /messages returned HTTP ${HTTP_CODE:-no-response} (body: ${BODY:-empty})"$'\n'
fi

echo ""
echo "=============================================================="
if [ $FAIL -eq 0 ]; then
    echo "    [Yay]  ALL CHECKS PASSED ($PASS/$PASS)"
else
    echo "    [Warn]   $PASS passed, $FAIL failed"
    echo ""
    echo "    Failed checks:"
    printf '%s' "${FAILED_CHECKS}"
fi
echo "=============================================================="
echo ""
echo "    [Web]  Frontend:     http://localhost:5173"
echo "    [API]  API:          http://localhost:8001"
echo "    [In]   Ingestion:    http://localhost:8000"
echo ""
echo "    [Log]  Per-service (real-time): logs/live/<service>.log"
echo "    [Log]  Init report:  ${LOGDIR}/init.log"
echo "    [Log]  Errors only:  ${LOGDIR}/errors.log"
echo "    [Log]  Important:    ${LOGDIR}/important.log"
echo "    [Log]  Full dump:    ${LOGDIR}/all.log"
echo ""
echo "    Tip: tail -f logs/live/trainer_service.log"
echo "    Tip: tail -f logs/live/central_responder_service.log"
echo "    Tip: tail -f ${LOGDIR}/errors.log"
echo "=============================================================="

{
    echo "[Health Checks]"
    echo "   PASS : ${PASS}"
    echo "   FAIL : ${FAIL}"
    if [ $FAIL -eq 0 ]; then
        echo "   Result : ALL CHECKS PASSED"
    else
        echo "   Result : ${FAIL} check(s) FAILED"
        echo ""
        echo "   Failed checks:"
        printf '%s' "${FAILED_CHECKS}"
        echo "   (Note: failed health checks are NOT container log errors —"
        echo "    errors.log only captures ERROR/WARN/CRITICAL lines emitted by"
        echo "    the services. Check all.log / the per-service logs above.)"
    fi
    echo ""
    echo "[URLs]"
    echo "   Frontend  : http://localhost:5173"
    echo "   API       : http://localhost:8001"
    echo "   Ingestion : http://localhost:8000"
    echo ""
    echo "[Log Files]"
    echo "   logs/live/<service>.log  (real-time, written directly by each service)"
    echo "   ${LOGDIR}/errors.log"
    echo "   ${LOGDIR}/important.log"
    echo "   ${LOGDIR}/all.log"
    echo ""
    echo "=============================================================="
    echo "[Container Startup Logs] snapshot since launch (all services)"
    echo "=============================================================="
    docker compose logs --since "${LAUNCH_TIME}" 2>&1
    echo ""
} >> "${INIT_LOG}"

echo ""
echo "   [Log] Init report saved: ${INIT_LOG}"

echo ""
