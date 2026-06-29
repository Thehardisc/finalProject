#!/usr/bin/env bash
#
# run_qa.sh — run the emotion-engine QA suite.
#
# Usage:
#   ./qa_suite/run_qa.sh              # fast offline tests (no docker stack needed)
#   ./qa_suite/run_qa.sh offline      # same as default
#   ./qa_suite/run_qa.sh slow         # big batteries: per-emotion corpus + ~950 fuzz
#   ./qa_suite/run_qa.sh live         # live @e2e tests (needs stack up + .env)
#   ./qa_suite/run_qa.sh all          # offline + live, excluding slow batteries
#   ./qa_suite/run_qa.sh full         # EVERYTHING: offline + slow + live (~1000 cases)
#   ./qa_suite/run_qa.sh calibrate    # derive/refresh thresholds from the battery
#
# Extra pytest args pass through, e.g.:  ./qa_suite/run_qa.sh offline -k emoji -v
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Prefer the project venv if present.
if [ -x "$ROOT/.venv/bin/python" ]; then
    PY="$ROOT/.venv/bin/python"
else
    PY="$(command -v python3 || command -v python)"
fi

load_env() {
    if [ -f "$ROOT/.env" ]; then
        set -a
        # shellcheck disable=SC1091
        source "$ROOT/.env"
        set +a
        echo "[run_qa] loaded .env"
    else
        echo "[run_qa] WARNING: no .env found — live tests may skip (INTERNAL_API_KEY unset)" >&2
    fi
}

ensure_model() {
    # If the trained meta-learner pkl is missing, copy it out of the running
    # central_responder container so the 3 model-gated invariants run instead of
    # skip. Falls back silently (suite still green) if the container/model isn't there.
    local m="$ROOT/central_responder_service/models/meta_weights.pkl"
    [ -f "$m" ] && return 0
    local cid
    cid="$(docker compose ps -q central_responder_service 2>/dev/null)"
    if [ -z "$cid" ]; then
        echo "[run_qa] no central_responder container — running in fallback (3 invariants skip)"
        return 0
    fi
    echo "[run_qa] fetching meta_weights.pkl from container $cid ..."
    if docker cp "$cid:/app/models/meta_weights.pkl" "$m" 2>/dev/null; then
        docker cp "$cid:/app/models/meta_weights_meta.json" "${m%.pkl}_meta.json" 2>/dev/null || true
        echo "[run_qa] model ready — invariants will run"
    else
        echo "[run_qa] could not fetch model — running in fallback (3 invariants skip)"
    fi
}

MODE="${1:-offline}"
[ $# -gt 0 ] && shift || true

case "$MODE" in
    offline)
        ensure_model
        echo "[run_qa] offline suite (no stack required)"
        exec "$PY" -m pytest qa_suite/ -m "not slow and not e2e" "$@"
        ;;
    slow)
        ensure_model
        echo "[run_qa] slow batteries (real GoEmotions 2000 + corpus + edge + fuzz — a few minutes)"
        exec "$PY" -m pytest qa_suite/ -m slow "$@"
        ;;
    live|e2e)
        load_env
        echo "[run_qa] live suite (@e2e — requires the docker stack)"
        exec "$PY" -m pytest qa_suite/test_live.py -m e2e "$@"
        ;;
    all)
        ensure_model
        load_env
        echo "[run_qa] offline + live (excluding slow batteries)"
        exec "$PY" -m pytest qa_suite/ -m "not slow" "$@"
        ;;
    full)
        ensure_model
        load_env
        echo "[run_qa] FULL suite — offline + slow + live (~1000 cases)"
        exec "$PY" -m pytest qa_suite/ "$@"
        ;;
    calibrate)
        echo "[run_qa] calibrating thresholds from the sentence battery"
        exec "$PY" qa_suite/calibrate.py "$@"
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Use: offline | slow | live | all | full | calibrate" >&2
        exit 2
        ;;
esac
