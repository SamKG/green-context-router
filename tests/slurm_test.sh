#!/bin/bash
export LD_PRELOAD=/scratch/gpfs/LI/samyakg/Research/green-ctx-router/target/release/libgreen_ctx_router.so:$LD_PRELOAD
export GREEN_CTX=$1
export RUST_LOG=info
./tests/bin/test_app
