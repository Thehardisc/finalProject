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
MAX_SAMPLES_VAL=$(grep '^MAX_SAMPLES=' .env 2>/dev/null | cut -d'=' -f2)
RETRAIN_INTERVAL_VAL=$(grep '^RETRAIN_INTERVAL_SECONDS=' .env 2>/dev/null | cut -d'=' -f2)
ACCURACY_GATE_VAL=$(grep '^ACCURACY_GATE=' .env 2>/dev/null | cut -d'=' -f2)
LLM_PROVIDER_VAL=$(grep '^LLM_PROVIDER=' .env 2>/dev/null | cut -d'=' -f2)
LOG_LEVEL_VAL=$(grep '^LOG_LEVEL=' .env 2>/dev/null | cut -d'=' -f2)
RATE_LIMIT_MAX_VAL=$(grep '^RATE_LIMIT_MAX=' .env 2>/dev/null | cut -d'=' -f2)

echo "[Config] Resolved environment from .env:"
echo "──────────────────────────────────────────────────────────"
echo "   LOG_LEVEL               = ${LOG_LEVEL_VAL:-INFO}"
echo "   MAX_SAMPLES             = ${MAX_SAMPLES_VAL:-2500}  (GoEmotions dataset size per run)"
echo "   RETRAIN_INTERVAL_SECS   = ${RETRAIN_INTERVAL_VAL:-1800}s"
echo "   ACCURACY_GATE           = ${ACCURACY_GATE_VAL:-0.40}  (min test accuracy to deploy model)"
echo "   LLM_PROVIDER            = ${LLM_PROVIDER_VAL:-RULE_BASED}"
echo "   RATE_LIMIT_MAX          = ${RATE_LIMIT_MAX_VAL:-60} req/min per user"
echo "   API_KEY                 = ${API_KEY:0:6}...  (redacted)"
echo "──────────────────────────────────────────────────────────"
echo ""

echo "[Search] Detecting host environment..."

if command -v nvidia-smi &> /dev/null; then
    echo "   [OK] NVIDIA GPU detected! Launching with GPU acceleration..."
    COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build"
else
    echo "   [INFO] No NVIDIA GPU detected. Launching in CPU mode..."
    COMPOSE_CMD="docker compose up -d --build"
fi

echo ""
$COMPOSE_CMD
echo ""

# Setup logging
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOGDIR="logs/run_${TIMESTAMP}"
mkdir -p "${LOGDIR}"

echo "[Logs] Writing per-service logs to: ${LOGDIR}/"

# Spawn a background log process for each service into its own file
SERVICES=$(docker compose config --services 2>/dev/null)
for SERVICE in $SERVICES; do
    docker compose logs -f "${SERVICE}" > "${LOGDIR}/${SERVICE}.log" 2>&1 &
done

# Also write a combined log that strips noisy health-check pings
# Filters out: GET /health, OPTIONS /health, 200 OK health responses
docker compose logs -f 2>&1 | grep -v '"GET /health HTTP' | grep -v 'GET /health HTTP' | grep -v 'OPTIONS /health' | grep -v '"GET / HTTP/1.1" 200' > "${LOGDIR}/important.log" 2>&1 &

# Errors-only log: captures actual ERROR/CRITICAL/FATAL lines from services
# Excludes false positives like logger names (uvicorn.error) and Python deprecation warnings
docker compose logs -f 2>&1 | grep -E '\] ERROR |\] CRITICAL |\] WARNING |FATAL|CRASH|ROLLBACK|Traceback' | grep -v 'uvicorn.error' > "${LOGDIR}/errors.log" 2>&1 &

# Keep a full combined log as well (for debugging)
docker compose logs -f > "${LOGDIR}/all.log" 2>&1 &
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
    return 1
}

# Check Redis
if docker compose exec -T redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
    echo "   [OK] Redis - PONG"
    PASS=$((PASS + 1))
else
    echo "   [Fail] Redis - not responding"
    FAIL=$((FAIL + 1))
fi

# Check PostgreSQL
if docker compose exec -T db pg_isready -U user -d emotion_db 2>/dev/null | grep -q "accepting"; then
    echo "   [OK] PostgreSQL - accepting connections"
    PASS=$((PASS + 1))
else
    echo "   [Fail] PostgreSQL - not ready"
    FAIL=$((FAIL + 1))
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
fi

echo "[Test] Running end-to-end pipeline test..."

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8000/messages \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"conversation_id": "healthcheck", "user_id": "system", "text": "I am happy!"}' 2>/dev/null)

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -1)

if [ "$HTTP_CODE" = "200" ]; then
    echo "   [OK] Pipeline test - message accepted (HTTP 200)"
    PASS=$((PASS + 1))
else
    echo "   [Fail] Pipeline test - failed (HTTP $HTTP_CODE)"
    FAIL=$((FAIL + 1))
fi

# Print summary
echo ""
echo "=============================================================="
if [ $FAIL -eq 0 ]; then
    echo "    [Yay]  ALL CHECKS PASSED ($PASS/$PASS)"
else
    echo "    [Warn]   $PASS passed, $FAIL failed"
fi
echo "=============================================================="
echo ""
echo "    [Web]  Frontend:     http://localhost:5173"
echo "    [API]  API:          http://localhost:8001"
echo "    [In]   Ingestion:    http://localhost:8000"
echo ""
echo "    [Logs] Directory:    ${LOGDIR}/"
echo "    [Log]  Errors only:  ${LOGDIR}/errors.log              (ERROR/WARN/CRITICAL)"
echo "    [Log]  Brain only:   ${LOGDIR}/important.log           (no health pings)"
echo "    [Log]  Responder:    ${LOGDIR}/central_responder_service.log"
echo "    [Log]  Database:     ${LOGDIR}/db.log"
echo "    [Log]  Full dump:    ${LOGDIR}/all.log"
echo ""
echo "    Tip: tail -f ${LOGDIR}/errors.log"
echo "    Tip: tail -f ${LOGDIR}/important.log"
echo "=============================================================="
echo ""
