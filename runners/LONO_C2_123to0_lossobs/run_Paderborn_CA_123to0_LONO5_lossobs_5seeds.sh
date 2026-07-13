#!/bin/bash
# ==============================================================================
# 진단 전용 재실행: 123to0 static meta(C2)에서 시드별로 학습이 어떻게 무너지는지
# epoch별 train/val loss + test AUROC(관찰용, checkpoint 선택에는 미사용)를
# train_log.jsonl로 남기기 위한 재실행. 기존 LONO_C2_5seeds 결과는 덮어쓰지 않음
# (RUN_NAME에 _lossobs 접미사). meta 정규화/모델 구조는 기존과 동일(변경 없음).
# ==============================================================================

# Change RUN_NAME when you change this experiment setting.
RUN_NAME="CA_raw_vib_123to0_LONO5_lossobs"

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
    --train_ids K001 K002 K003 K004 \
    --val_ids K006 \
    --test_norm_ids K005 \
    --use_meta \
    --meta_emb_dim 8 \
    --log_test_auroc \
    --seeds 2024 2025 2026 2027 2028
