#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if [ -x "$ROOT/.venv/bin/python" ]; then
    PY="$ROOT/.venv/bin/python"
else
    PY="$(command -v python3 || command -v python)"
fi

load_env() {
    if [ -f "$ROOT/.env" ]; then
        set -a
        source "$ROOT/.env"
        set +a
        echo "[run_qa] loaded .env"
    else
        echo "[run_qa] WARNING: no .env found — live tests may skip (INTERNAL_API_KEY unset)" >&2
    fi
}

ensure_model() {
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
