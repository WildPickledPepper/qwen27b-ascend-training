#!/usr/bin/env bash
set -eo pipefail
: "${SFT_RUN:?Set an isolated run directory}"
: "${MODEL_DIR:?Set original model directory}"
: "${TRAINING_PYTHON:?Set training environment Python}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Select one physical NPU}"
cd "$SFT_RUN"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec timeout --signal=TERM --kill-after=30s 2400 "$TRAINING_PYTHON" -u "$SFT_RUN/train_baseline.py" \
 --model "$MODEL_DIR" --model_type qwen3_5 \
 --tuner_type lora --torch_dtype bfloat16 --bf16 true --target_modules q_proj v_proj in_proj_qkv \
 --lora_rank 4 --lora_alpha 16 --lora_dropout 0 --freeze_vit true --freeze_aligner true \
 --dataset "$SFT_RUN/train.jsonl" --val_dataset "$SFT_RUN/validation.jsonl" --split_dataset_ratio 0 \
 --max_length 128 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
 --gradient_accumulation_steps 4 --gradient_checkpointing true --use_cache false \
 --learning_rate 0.0001 --max_steps 8 --warmup_ratio 0 --lr_scheduler_type constant \
 --logging_steps 1 --eval_on_start true --eval_strategy steps --eval_steps 8 \
 --save_strategy steps --save_steps 8 --save_total_limit 1 --save_only_model true \
 --output_dir "$SFT_RUN/output-baseline" --add_version false --report_to none \
 --enable_thinking false --enable_npu_model_patch false --attn_impl eager \
 --dataloader_num_workers 0 --dataset_num_proc 1 --seed 20261023 --data_seed 20261023 \
 --callbacks single_card_audit --padding_free false
