#!/usr/bin/env bash
# Source the already verified torch-npu/DeepSpeed environment before running.
set -euo pipefail
: "${MODEL_DIR:?Set the original read-only model directory}"
: "${DATA_DIR:?Set the independent train/validation data directory}"
: "${CPU_ADAM_EXTENSION:?Set the existing verified CPUAdam binary}"
: "${VEOMNI_SOURCE:?Set the independent pinned VeOmni checkout}"
test "$(git -C "$VEOMNI_SOURCE" rev-parse HEAD)" = eba362dd4ca43b6cba53dc69c1d378cf3eda57b6
output_dir="${1:?Choose a new output directory}"
test ! -e "$output_dir"
# The measured CPU ranges and NUMA nodes are specific to this server.
export ASCEND_RT_VISIBLE_DEVICES=2,3 DS_ACCELERATOR=npu DS_BUILD_OPS=0
export TASK_QUEUE_ENABLE=0 OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/triton-cache}"
# Run warm_gdn.py separately before distributed startup when the JIT cache is cold.
python -m torch.distributed.run --standalone --nproc_per_node=2 train_fusion.py \
 --model "$MODEL_DIR" --data "$DATA_DIR" --output "$output_dir" \
 --config prefetch-large-groups.json --micro-batch 16 --global-batch 32 \
 --checkpoint selective --numa-spread --cpu-threads 24 --cpu-mode pipeline --fuse-gdn --steps 3
