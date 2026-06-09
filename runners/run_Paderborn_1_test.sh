#!/bin/bash
# Evaluate a non-overwriting Paderborn run.
# Usage examples:
#   RUN_NAME=20260609-120000_loads-N15_M07_F10_model-MAF_blocks-2_hidden-32_win-2048_stride-1024_bs-256_lr-0.002 bash runners/run_Paderborn_1_test.sh
#   CKPT_PATH=checkpoints/Paderborn/my_run/model.pth bash runners/run_Paderborn_1_test.sh

ARGS=()
if [[ -n "${CKPT_PATH:-}" ]]; then
    ARGS+=(--ckpt_path="$CKPT_PATH")
elif [[ -n "${RUN_NAME:-}" ]]; then
    ARGS+=(--run_name="$RUN_NAME")
else
    echo "Set RUN_NAME or CKPT_PATH to choose a Paderborn checkpoint." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python3 test.py \
    --name=paderborn \
    --n_blocks=2 \
    --batch_size=256 \
    --window_size=2048 \
    --stride_size=1024 \
    --load_setting=N15_M07_F10 \
    "${ARGS[@]}"
