#!/bin/bash
# ==============================================================================
# 진단 전용 재실행(123to0_LONO6_lossobs)의 평가 스크립트. 학습 스크립트와
# RUN_NAME/split 구성이 반드시 일치해야 함.
# ==============================================================================

# Change RUN_NAME when you change this experiment setting.
RUN_NAME="CA_raw_vib_123to0_LONO6_lossobs"

CUDA_VISIBLE_DEVICES=0 python3 test.py \
    --name=paderborn \
    --run_name="$RUN_NAME" \
    --n_blocks=2 \
    --batch_size=256 \
    --window_size=2048 \
    --stride_size=1024 \
    --sampling_rate=64000 \
    --train_load_setting N09_M07_F10 N15_M01_F10 N15_M07_F04 \
    --test_load_setting N15_M07_F10 \
    --train_ids K002 K003 K004 K005 \
    --val_ids K001 \
    --test_norm_ids K006 \
    --use_meta \
    --meta_emb_dim 8 \
    --seeds 2024 2025 2026 2027 2028
