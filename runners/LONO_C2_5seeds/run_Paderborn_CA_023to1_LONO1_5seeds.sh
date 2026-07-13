#!/bin/bash
# ==============================================================================
# Created by Dayoon (2026) for Paderborn Bearing Dataset Baseline
# Description: Runner script for MTGFlow on Paderborn dataset (Setting 0)
# ==============================================================================

# Change RUN_NAME when you change this experiment setting.
RUN_NAME="CA_raw_vib_023to1_LONO1"

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --name=paderborn \
    --run_name="$RUN_NAME" \
    --n_blocks=2 \
    --batch_size=256 \
    --window_size=2048 \
    --stride_size=1024 \
    --sampling_rate=64000 \
    --train_load_setting N15_M07_F10 N15_M01_F10 N15_M07_F04 \
    --test_load_setting N09_M07_F10 \
    --train_ids K003 K004 K005 K006 \
    --val_ids K002 \
    --test_norm_ids K001 \
    --use_meta \
    --meta_emb_dim 8 \
    --seeds 2024 2025 2026 2027 2028
