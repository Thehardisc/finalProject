#!/bin/bash
# train.sh — Local Mac MPS Acceleration script
# This script sets up the local environment, runs the training cycle on the GPU,
# and shuts down the Mac when finished.

set -e  # Exit immediately if any command fails

# Ensure we are in the root directory
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

# Set up logging to both console and file
mkdir -p logs
LOG_FILE="logs/local_trainer.log"
echo "Logging all output to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "====================================================="
echo "  Starting GPU Accelerated Trainer for Apple Silicon"
echo "====================================================="


# 1. Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

# 2. Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# 3. Install requirements
echo "Installing/Updating dependencies..."
pip install --upgrade pip
pip install -r central_responder_service/requirements.txt

# 4. Run the trainer
echo "Starting Trainer with dual PYTHONPATH (MPS Batch Size: 1024)..."

# Ensure models directory exists locally
mkdir -p .cache

cd central_responder_service

# Load the .env variables so python can see the Anthropic API key
if [ -f "../.env" ]; then
    echo "Loading .env file..."
    export $(grep -v '^#' ../.env | xargs)
fi

export PYTHONPATH="..:."
export MODEL_PATH="../.cache/meta_weights.pkl"

# Run the python script directly to avoid sys.path issues
python3 -c "
import sys
from trainer.cycle import run_one_cycle
print('Starting one_cycle execution...')
run_one_cycle()
print('Training completed successfully!')
"

# 5. Finish
echo "====================================================="
echo "  Training Complete. Trainer has been shut down."
echo "====================================================="
