#!/bin/bash
# ==============================================================================
# UODS 외부 검증 pilot — raw MTGFlow(B3, flow_NLL). 1차 비교(raw vs proposed)의 raw 기준.
# 동일 split(tuning/run_0)·동일 seed(2026)·동일 window/전처리. amp 인자 없음(순수 flow).
# ==============================================================================
RUN_NAME="uods_pilot_raw"

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --name=uods \
    --run_name="$RUN_NAME" \
    --uods_split_manifest results/UODS/splits/uods_split_tuning_run_0.json \
    --n_blocks=2 \
    --batch_size=256 \
    --window_size=2048 \
    --stride_size=1024 \
    --sampling_rate=42000 \
    --seeds 2026
