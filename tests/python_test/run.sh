#!/bin/bash

# Get the absolute path to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Inject the environment variables
export LD_LIBRARY_PATH="$PROJECT_ROOT/target/release:$LD_LIBRARY_PATH"
export GREEN_CTX_TRACE=info

# Run the PyTorch concurrent test using uv
echo "Running test with LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
cd "$SCRIPT_DIR"
uv run python test_concurrent.py
