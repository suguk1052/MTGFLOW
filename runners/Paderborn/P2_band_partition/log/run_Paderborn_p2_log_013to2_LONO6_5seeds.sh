#!/bin/bash
# ==============================================================================
# 작업 P-2(Band partition scheme): --amp_n_bands 6 고정 + --amp_band_scheme log, 나머지 G-3a 동일.
# 파일명 _5seeds는 slurm_run.sh 짝 매칭(_5seeds->_test_5seeds) 규약 토큰. P-G1: 처음부터 5-seed 전량(2024~2028) 한 잡.
# ==============================================================================

RUN_NAME="p2_log_013to2_LONO6"

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --name=paderborn \
    --run_name="$RUN_NAME" \
    --n_blocks=2 \
    --batch_size=256 \
    --window_size=2048 \
    --stride_size=1024 \
    --sampling_rate=64000 \
    --train_load_setting N15_M07_F10 N09_M07_F10 N15_M07_F04 \
    --test_load_setting N15_M01_F10 \
    --train_ids K002 K003 K004 K005 \
    --val_ids K001 \
    --test_norm_ids K006 \
    --seeds 2024 2025 2026 2027 2028 \
    --amp_normalize \
    --amp_branch \
    --amp_n_bands 6 \
    --amp_band_scheme log
