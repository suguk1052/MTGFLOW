#!/bin/bash
# ==============================================================================
# 작업 E 1단계 Phase B — order tracking 효과 격리 (023→1 LONO2, 저진폭 K002 target)
# raw no-meta seed2026 학습. 기존 B3 LONO2와 동일 인자.
# ==============================================================================

RUN_NAME="E_raw_023to1_LONO2"

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
    --train_ids K001 K004 K005 K006 \
    --val_ids K003 \
    --test_norm_ids K002 \
    --seeds 2026
