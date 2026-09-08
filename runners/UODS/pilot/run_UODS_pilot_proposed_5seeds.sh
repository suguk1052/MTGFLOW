#!/bin/bash
# ==============================================================================
# UODS 외부 검증 pilot — proposed (동결: N=6 + log 밴드 + amp branch + Fisher-tail).
# split = Vieira tuning/run_0 기반 manifest(2/1/2), 단일 seed(2026). 구조·하이퍼 동결(재선택 없음).
# 파일명 _5seeds는 slurm_run.sh 짝 매칭(_5seeds->_test_5seeds) 규약 토큰(실제 seed는 1개).
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
    --seeds 2026 \
    --amp_normalize \
    --amp_branch \
    --amp_n_bands 6 \
    --amp_band_scheme log
