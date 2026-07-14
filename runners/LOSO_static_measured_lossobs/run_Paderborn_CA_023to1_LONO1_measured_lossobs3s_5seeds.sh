#!/bin/bash
# ==============================================================================
# LOSO 전반 진단(static/measured 공통): val_loss<->test_AUROC 괴리(gap)가
# 123to0만의 현상인지 나머지 LOSO split(023to1/013to2/012to3) 전반의 현상인지
# 확정하기 위한 재실행. --log_test_auroc로 epoch별 test AUROC 기록.
# 기존 LONO_C2_5seeds / LONO_C2_measured_5seeds 결과는 덮어쓰지 않음(_lossobs3s 접미사).
# ==============================================================================

RUN_NAME="CA_raw_vib_023to1_LONO1_measured_lossobs3s"

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
    --meta_source measured \
    --log_test_auroc \
    --seeds 2024 2025 2026
