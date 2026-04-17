#!/bin/bash
# start.sh - Universal AI Stack Launcher (Mac & Linux)
# Automatically detects if your machine has an NVIDIA GPU and injects the hardware profiles.

echo "Detecting host environment..."

if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU Detetected! Launching with GPU acceleration..."
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
else
    echo "ℹ️ No NVIDIA GPU detected (or running on Apple Silicon)."
    echo "   Launching aggressively in CPU mode..."
    docker compose up -d
fi

# ── LOGGING SYSTEM ───────────────────────────────────────────────────────────
# Ensure the logs folder exists
mkdir -p logs

# Generate a unique filename using the current timestamp
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOGFILE="logs/run_${TIMESTAMP}.log"

echo "📝 Streaming all background processes into: ${LOGFILE}"
# Execute docker compose logs in "follow" mode in the background
# This will capture every print/error from every service and write it to the file.
docker compose logs -f > "${LOGFILE}" 2>&1 &
