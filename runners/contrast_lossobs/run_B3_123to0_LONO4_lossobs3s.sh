#!/bin/bash
# ==============================================================================
# 대조군: B3(no-meta) LOSO 123to0_LONO4. C2(static meta)가 무너지던 바로 그
# split/LONO에서 no-meta도 val_loss<->test_AUROC 괴리를 겪는지 확인하기 위한
# 최소 대조 실험(3seed, --log_test_auroc). 기존 LONO_B3_5seeds 결과는 덮어쓰지 않음
# (RUN_NAME에 _lossobs3s 접미사).
# ==============================================================================

RUN_NAME="raw_vib_123to0_LONO4_lossobs3s"

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --name=paderborn \
    --run_name="$RUN_NAME" \
    --n_blocks=2 \
    --batch_size=256 \
    --window_size=2048 \
    --stride_size=1024 \
    --sampling_rate=64000 \
    --train_load_setting N09_M07_F10 N15_M01_F10 N15_M07_F04 \
    --test_load_setting N15_M07_F10 \
    --train_ids K001 K002 K003 K006 \
    --val_ids K005 \
    --test_norm_ids K004 \
    --log_test_auroc \
    --seeds 2024 2025 2026
