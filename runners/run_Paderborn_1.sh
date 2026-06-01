#!/bin/bash
# ==============================================================================
# Created by Dayoon (2026) for Paderborn Bearing Dataset Baseline
# Description: Runner script for MTGFlow on Paderborn dataset (Setting 0)
# ==============================================================================

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --name=paderborn \
    --n_blocks=2 \
    --batch_size=256 \
    --window_size=2048 \
    --stride_size=1024 \
    --load_setting=N15_M07_F10