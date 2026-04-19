#!/bin/bash

# Get the absolute path to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Inject the environment variables
export LD_PRELOAD="$PROJECT_ROOT/target/release/libgreen_ctx_router.so:$LD_PRELOAD"
export GREEN_CTX_TRACE=info

# Run the PyTorch concurrent test
echo "Running test with LD_PRELOAD=$LD_PRELOAD"
cd "$SCRIPT_DIR"
source .venv/bin/activate
python test_concurrent.py
