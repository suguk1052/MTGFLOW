#!/bin/bash
# ==============================================================================
# run_Paderborn_CA_013to2_LONO1_measured_lossobs3s.sh 짝 평가 스크립트.
# ==============================================================================

RUN_NAME="CA_raw_vib_013to2_LONO1_measured_lossobs3s"

CUDA_VISIBLE_DEVICES=0 python3 test.py \
    --name=paderborn \
    --run_name="$RUN_NAME" \
    --n_blocks=2 \
    --batch_size=256 \
    --window_size=2048 \
    --stride_size=1024 \
    --sampling_rate=64000 \
    --train_load_setting N15_M07_F10 N09_M07_F10 N15_M07_F04 \
    --test_load_setting N15_M01_F10 \
    --train_ids K003 K004 K005 K006 \
    --val_ids K002 \
    --test_norm_ids K001 \
    --use_meta \
    --meta_source measured \
    --seeds 2024 2025 2026
