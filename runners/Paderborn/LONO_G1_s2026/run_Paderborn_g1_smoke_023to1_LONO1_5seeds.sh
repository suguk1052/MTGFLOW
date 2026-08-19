#!/bin/bash
# ==============================================================================
# 작업 G-1 스모크(파이프라인 동작 확인용, 성능 판정 아님). 1 fold(023to1 LONO1) × 3 epoch.
# 결과를 24-fold 게이트(g1_*)와 분리하려 RUN_NAME 접두는 g1_smoke_. seed 2026.
# ==============================================================================

RUN_NAME="g1_smoke_023to1_LONO1"

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --name=paderborn \
    --run_name="$RUN_NAME" \
    --n_blocks=2 \
    --batch_size=256 \
    --window_size=2048 \
    --stride_size=1024 \
    --sampling_rate=64000 \
    --epochs=3 \
    --train_load_setting N15_M07_F10 N15_M01_F10 N15_M07_F04 \
    --test_load_setting N09_M07_F10 \
    --train_ids K003 K004 K005 K006 \
    --val_ids K002 \
    --test_norm_ids K001 \
    --seeds 2026 \
    --amp_normalize \
    --amp_branch
