#!/bin/bash
# ==============================================================================
# 대조군: C1_measured(pooled, 실측 meta) LONO4. pooled+meta 조합에서도 val_loss<->
# test_AUROC 괴리/시드 불안정이 있는지 확인하기 위한 최소 대조 실험
# (3seed, --log_test_auroc). 기존 LONO_C1_measured_5seeds 결과는 덮어쓰지 않음.
# ==============================================================================

RUN_NAME="CA_raw_vib_0123C_LONO4_measured_lossobs3s"

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --name=paderborn \
    --run_name="$RUN_NAME" \
    --n_blocks=2 \
    --batch_size=256 \
    --window_size=2048 \
    --stride_size=1024 \
    --sampling_rate=64000 \
    --load_setting N15_M07_F10 N09_M07_F10 N15_M01_F10 N15_M07_F04 \
    --train_ids K001 K002 K003 K006 \
    --val_ids K005 \
    --test_norm_ids K004 \
    --use_meta \
    --meta_source measured \
    --log_test_auroc \
    --seeds 2024 2025 2026
