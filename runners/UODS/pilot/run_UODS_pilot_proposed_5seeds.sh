#!/bin/bash
# ==============================================================================
# UODS 외부 검증 — "1-split × 5-seed extended pilot" proposed (동결: N=6 + log 밴드 + amp branch + Fisher-tail).
# split = Vieira tuning/run_0 기반 manifest(2/1/2) 하나. seeds = 2024~2028(P/G 트랙 동일). 구조·하이퍼 동결(재선택 없음).
# main.py가 seed별 run_name에 _s{seed}를 붙여 checkpoint 폴더를 자동 분기(seed·model 경로 비충돌).
# ==============================================================================
RUN_NAME="uods_pilot_proposed"

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --name=uods \
    --run_name="$RUN_NAME" \
    --uods_split_manifest results/UODS/splits/uods_split_tuning_run_0.json \
    --n_blocks=2 \
    --batch_size=256 \
    --window_size=2048 \
    --stride_size=1024 \
    --sampling_rate=42000 \
    --seeds 2024 2025 2026 2027 2028 \
    --amp_normalize \
    --amp_branch \
    --amp_n_bands 6 \
    --amp_band_scheme log
