#!/bin/bash

# Change RUN_NAME to the result folder name you want to evaluate.
RUN_NAME="S0_A_raw_ch1_win2048"

CUDA_VISIBLE_DEVICES=0 python3 test.py \
    --name=paderborn \
    --run_name="$RUN_NAME" \
    --n_blocks=2 \
    --batch_size=256 \
    --window_size=2048 \
    --stride_size=1024 \
    --sampling_rate=64000 \
    --load_setting=N15_M07_F10
