#!/bin/bash
# ==============================================================================
# 작업 F-2: C1C2Vy 다채널(all=shape-only 전채널). seed 2026. 파일명 _5seeds는 slurm_run.sh
# 짝 매칭(_5seeds->_test_5seeds) 규약을 위한 레거시 토큰일 뿐, 실제는 seed 2026 하나.
# ==============================================================================

RUN_NAME="f2_all_013to2_LONO1"

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
    --seeds 2026 \
    --sensor_mode C1C2Vy \
    --amp_normalize_channels all \
    --amp_normalize \
    --rms_lambda 0
