#!/bin/bash
# Usage: bash runners/run_Paderborn_1_test.sh <run_name>

if [ -z "$1" ]; then
    echo "Usage: bash runners/run_Paderborn_1_test.sh <run_name>" >&2
    exit 1
fi

RUN_NAME="$1"

CUDA_VISIBLE_DEVICES=0 python3 test.py \
    --name=paderborn \
    --run_name="$RUN_NAME" \
    --n_blocks=2 \
    --batch_size=256 \
    --window_size=2048 \
    --stride_size=1024 \
    --load_setting=N15_M07_F10
