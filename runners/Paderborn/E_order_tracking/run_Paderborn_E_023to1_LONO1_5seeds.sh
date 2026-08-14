#!/bin/bash
# ==============================================================================
# 작업 E 1단계 Phase B — order tracking 효과 격리 (023→1 LONO1, 고진폭 K001 target)
# raw no-meta seed2026 학습. 023→1은 source=1500rpm 단일이라 OT는 train에서 identity →
# 이 raw checkpoint가 raw·raw+OT 양쪽 모델(OT는 test-time 변환). 기존 B3 LONO1과 동일 인자.
# (파일명 _5seeds는 slurm_run.sh 페어링 규칙용 접미사; 실제 seed는 2026 단일.)
# ==============================================================================

RUN_NAME="E_raw_023to1_LONO1"

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
    --seeds 2026
