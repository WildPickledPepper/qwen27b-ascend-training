#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_DIR:?Set the original Qwen3.5-27B directory}"
: "${CPU_ADAM_EXTENSION:?Set the existing verified CPUAdam binary}"
: "${VEOMNI_SOURCE:?Set the separate pinned VeOmni source directory}"
test "$(git -C "$VEOMNI_SOURCE" rev-parse HEAD)" = eba362dd4ca43b6cba53dc69c1d378cf3eda57b6
export ASCEND_RT_VISIBLE_DEVICES=2,3 DS_ACCELERATOR=npu DS_BUILD_OPS=0
export TASK_QUEUE_ENABLE=0 OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24
export TOKENIZERS_PARALLELISM=false HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/triton-cache}"
python run_benchmark.py
python run_matched.py
