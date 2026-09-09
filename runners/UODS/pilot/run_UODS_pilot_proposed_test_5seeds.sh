#!/bin/bash
# ==============================================================================
# UODS extended pilot proposed — 5 seed per-window score dump(GPU forward-only). slurm_run.sh "test" 슬롯.
# UODS는 test.py 대신 analysis dump→fusion 경로로 평가(동결 재현 경로와 동일 성격).
# seed별 dump → results/UODS/uods_window_scores/uods_pilot_proposed_s<seed>.npz (5개, 비충돌).
# ==============================================================================
for SEED in 2024 2025 2026 2027 2028; do
    CUDA_VISIBLE_DEVICES=0 python3 analysis/dump_uods_window_scores.py \
        --run_name uods_pilot_proposed_s${SEED} \
        --overwrite
done
