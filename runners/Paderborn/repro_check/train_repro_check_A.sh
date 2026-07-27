#!/bin/bash
# ==============================================================================
# 재현성 검증용 임시 스크립트 (TODO §0-(1)). 기존 LONO_B2_5seeds LONO1(no-meta)
# config와 동일, 시드 1개(2024)만 사용. A/B 두 번 독립 실행해 결과를 비교한다.
# ==============================================================================
RUN_NAME="repro_check_LONO1_A"

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --name=paderborn \
    --run_name="$RUN_NAME" \
    --n_blocks=2 \
    --batch_size=256 \
    --window_size=2048 \
    --stride_size=1024 \
    --sampling_rate=64000 \
    --load_setting N15_M07_F10 N09_M07_F10 N15_M01_F10 N15_M07_F04 \
    --train_ids K003 K004 K005 K006 \
    --val_ids K002 \
    --test_norm_ids K001 \
    --seeds 2024
