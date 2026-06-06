#!/bin/bash
# start.sh - Universal AI Stack Launcher (Mac & Linux)
# Automatically detects NVIDIA GPU, launches Docker, waits for readiness, and verifies health.

set -e

echo ""
echo "=============================================================="
echo "    [System]  Emotion Analysis System - Launcher"
echo "=============================================================="
echo ""

# Load environment
# Extract API key for health checks
API_KEY=$(grep "^INTERNAL_API_KEY=" .env | cut -d'=' -f2 | tr -d '"'\'' ')
API_KEY=${API_KEY:-dev-secret-key}

# Load and display all config from .env so operators can verify before launch
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

# Feature-vector parity is the #1 invariant (CLAUDE.md): inference and trainer must
# produce identical 107-dim vectors. Catch drift before spending minutes on a build.
echo "[Invariant] Checking feature-vector parity (inference vs trainer)..."
if command -v python3 &> /dev/null && python3 -c "import pytest, numpy" &> /dev/null; then
    if python3 -m pytest central_responder_service/training/test_feature_parity.py -q &> /tmp/parity_check.log; then
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

echo "[Search] Detecting host environment..."

if command -v uname &> /dev/null && [ "$(uname -s)" = "Darwin" ]; then
    if [ ! -f "models/meta_weights.pkl" ] && [ -f "train.sh" ]; then
        echo "   [INFO] No meta_weights.pkl found on Mac host."
        echo "   [INFO] Running local MPS-accelerated trainer before starting services..."
        ./train.sh
    fi
fi

if command -v nvidia-smi &> /dev/null; then
    echo "   [OK] NVIDIA GPU detected! Launching with GPU acceleration..."
    COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build"
else
    echo "   [INFO] No NVIDIA GPU detected. Launching in CPU mode..."
    COMPOSE_CMD="docker compose up -d --build"
fi

# Setup logging — create timestamped run directory and the live log dir.
# logs/live/ is bind-mounted into every Python container so services write
# directly to host files at write time (zero lag, no pipe, no ANSI codes).
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOGDIR="logs/run_${TIMESTAMP}"
mkdir -p "${LOGDIR}"
mkdir -p logs/live && find logs/live -maxdepth 1 -type f -delete

echo ""
echo "[Launch] ${COMPOSE_CMD}"
echo ""
# Capture build/start output directly into init.log while showing it live.
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

# Strip ANSI escape codes (docker compose emits colour/cursor codes even to files)
_strip_ansi() { sed $'s/\033\\[[0-9;]*[A-Za-z]//g'; }

# Per-service logs are written directly by each container via LOG_DIR=/app/logs
# bound to ./logs/live/ — no watchdog needed, zero lag.

# Combined aggregate logs — watchdog pattern for all-in-one views
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

# Wait for containers
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

# Health checks
echo ""
echo "[Health] Running service health checks..."
PASS=0
FAIL=0
FAILED_CHECKS=""   # accumulates the name+reason of every failed check for the init report

# Helper function
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

# Check Redis
if docker compose exec -T redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
    echo "   [OK] Redis - PONG"
    PASS=$((PASS + 1))
else
    echo "   [Fail] Redis - not responding"
    FAIL=$((FAIL + 1))
    FAILED_CHECKS="${FAILED_CHECKS}   - Redis: ping did not return PONG"$'\n'
fi

# Check PostgreSQL
if docker compose exec -T db pg_isready -U user -d emotion_db 2>/dev/null | grep -q "accepting"; then
    echo "   [OK] PostgreSQL - accepting connections"
    PASS=$((PASS + 1))
else
    echo "   [Fail] PostgreSQL - not ready"
    FAIL=$((FAIL + 1))
    FAILED_CHECKS="${FAILED_CHECKS}   - PostgreSQL: not accepting connections"$'\n'
fi

# Check HTTP services
check_service "Ingestion Service (API :8000)" "http://localhost:8000/health"
check_service "API Service (WebSocket :8001)" "http://localhost:8001/conversation/conv-1/state"
check_service "Frontend (UI :5173)" "http://localhost:5173"

# Check Meta-Learner loaded
if docker compose logs central_responder_service 2>/dev/null | grep -q "Running in META-LEARNER mode"; then
    echo "   [OK] Meta-Learner - loaded and active"
    PASS=$((PASS + 1))
else
    echo "   [Fail] Meta-Learner - not loaded (check central_responder_service logs)"
    FAIL=$((FAIL + 1))
    FAILED_CHECKS="${FAILED_CHECKS}   - Meta-Learner: 'Running in META-LEARNER mode' not found in central_responder_service logs"$'\n'
fi

echo "[Test] Running end-to-end pipeline test..."

# Retry like the other HTTP checks — ingestion can still be warming up right
# after the container reports "running", so a single shot races the cold start.
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

# Print summary
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

# Append health check results to init.log
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
