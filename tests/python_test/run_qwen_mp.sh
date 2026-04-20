#!/bin/bash

# Get the absolute path to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Inject the environment variables
cd "$SCRIPT_DIR"
source .venv/bin/activate
export LD_PRELOAD="$PROJECT_ROOT/target/release/libcuda.so:$PROJECT_ROOT/target/release/libcuda.so.1"

# Run the PyTorch concurrent Qwen test using multi-processing
echo "Running test with LD_PRELOAD=$LD_PRELOAD"
python test_qwen_mp.py
