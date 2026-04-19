#!/bin/bash
export LD_PRELOAD=/scratch/gpfs/LI/samyakg/Research/green-ctx-router/target/release/libgreen_ctx_router.so:$LD_PRELOAD
export GREEN_CTX_TRACE=info
cd tests/python_test
source .venv/bin/activate
python test_concurrent.py
