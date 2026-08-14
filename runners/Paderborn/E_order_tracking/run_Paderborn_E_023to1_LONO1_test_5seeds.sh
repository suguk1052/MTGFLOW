#!/bin/bash
# ==============================================================================
# 작업 E 1단계 Phase B — 023→1 LONO1 raw 평가(아카이브 B3 seed2026 AUROC 0.088 재현 대조용).
# raw+OT 및 per-fault 비교는 analysis/diagnose_E_order_tracking_eval.py에서 수행.
# ==============================================================================

RUN_NAME="E_raw_023to1_LONO1"

CUDA_VISIBLE_DEVICES=0 python3 test.py \
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
    --seeds 2026
