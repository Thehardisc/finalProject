#!/bin/bash

set -e  # Exit immediately if any command fails

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

mkdir -p logs
LOG_FILE="logs/local_trainer.log"
echo "Logging all output to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "====================================================="
echo "  Starting GPU Accelerated Trainer for Apple Silicon"
echo "====================================================="


if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing/Updating dependencies..."
pip install --upgrade pip
pip install -r central_responder_service/requirements.txt

echo "Starting Trainer with dual PYTHONPATH (MPS Batch Size: 1024)..."

mkdir -p .cache

cd central_responder_service

if [ -f "../.env" ]; then
    echo "Loading .env file..."
    export $(grep -v '^#' ../.env | xargs)
fi

export PYTHONPATH="..:."
export MODEL_PATH="../.cache/meta_weights.pkl"

python3 -c "
import sys
from trainer.cycle import run_one_cycle
print('Starting one_cycle execution...')
run_one_cycle()
print('Training completed successfully!')
"

echo "====================================================="
echo "  Training Complete. Trainer has been shut down."
echo "====================================================="
