#!/usr/bin/env bash
# Run from this directory after sourcing the existing matched training environment.
# CPU/NUMA layout is specific to the measured server; check before reuse elsewhere.
set -euo pipefail
: "${MODEL_DIR:?Set the original read-only model directory}"
: "${DATA_DIR:?Set the independent training/validation data directory}"
: "${CPU_ADAM_EXTENSION:?Set the previously verified CPUAdam shared object}"
output_dir="${1:?Choose a new output directory}"
test ! -e "$output_dir"
export ASCEND_RT_VISIBLE_DEVICES=2,3 DS_ACCELERATOR=npu DS_BUILD_OPS=0
export TASK_QUEUE_ENABLE=0 OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
python -m torch.distributed.run --standalone --nproc_per_node=2 train_advanced.py \
 --model "$MODEL_DIR" --data "$DATA_DIR" --output "$output_dir" \
 --config prefetch-large-groups.json --micro-batch 16 --global-batch 32 \
 --checkpoint selective --numa-spread --cpu-threads 24 --steps 3
