#!/bin/bash
# ==============================================================================
# 작업 G-1: dual-branch shape/amplitude disentangle. 파일명 _5seeds는 slurm_run.sh
# 짝 매칭(_5seeds->_test_5seeds) 규약을 위한 토큰. 2026 재사용, 나머지 4 seed(2024/2025/2027/2028) 추가 실행 → 최종 5-seed.
# ==============================================================================

RUN_NAME="g1_013to2_LONO2"

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
    --train_ids K001 K004 K005 K006 \
    --val_ids K003 \
    --test_norm_ids K002 \
    --seeds 2024 2025 2027 2028 \
    --amp_normalize \
    --amp_branch
