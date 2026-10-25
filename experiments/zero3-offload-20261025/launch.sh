#!/usr/bin/env bash
set -eo pipefail
: "${TRAIN_ENV:?Set existing isolated training env.sh}"
: "${MODEL_DIR:?Set original read-only model directory}"
: "${DATA_DIR:?Set independent train/validation directory}"
: "${CPU_ADAM_EXTENSION:?Set verified CPUAdam extension}"
source "$TRAIN_ENV"
export ASCEND_RT_VISIBLE_DEVICES=2,3 QWEN_NPU_DEVICES=2,3
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
export TASK_QUEUE_ENABLE=0
exec timeout --signal=TERM --kill-after=30s 2400 python -m torch.distributed.run \
 --standalone --nproc_per_node=2 train_zero3.py --steps 4 --output dual-run1 \
 --model "$MODEL_DIR" --data "$DATA_DIR"
