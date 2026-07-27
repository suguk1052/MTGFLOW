#!/bin/bash
# ==============================================================================
# FiLM-measured (A안: condition C 변조, measured-mean, 항등 초기화) — C2 023to1 LONO5
# Change RUN_NAME when you change this experiment setting.
# ==============================================================================

RUN_NAME="CA_raw_vib_023to1_LONO5_FiLM_measured_5seeds"

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
    --train_ids K001 K002 K003 K004 \
    --val_ids K006 \
    --test_norm_ids K005 \
    --use_meta \
    --meta_source measured \
    --measured_meta_stats mean \
    --meta_inject film \
    --seeds 2024 2025 2026 2027 2028
