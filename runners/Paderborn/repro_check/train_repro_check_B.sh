#!/bin/bash
# ==============================================================================
# 재현성 검증용 임시 스크립트 (TODO §0-(1)). train_repro_check_A.sh와 완전히
# 동일한 config, RUN_NAME만 다름. 같은 시드(2024)로 독립 재학습해 A와 비교한다.
# ==============================================================================
RUN_NAME="repro_check_LONO1_B"

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
