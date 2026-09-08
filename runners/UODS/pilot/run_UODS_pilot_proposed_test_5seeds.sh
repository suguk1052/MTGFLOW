#!/bin/bash
# ==============================================================================
# UODS pilot proposed — per-window score dump(GPU forward-only). slurm_run.sh의 "test" 슬롯.
# UODS는 test.py 대신 analysis dump→fusion 경로로 평가(동결 재현 경로와 동일 성격).
# ==============================================================================
CUDA_VISIBLE_DEVICES=0 python3 analysis/dump_uods_window_scores.py \
    --run_name uods_pilot_proposed_s2026 \
    --overwrite
