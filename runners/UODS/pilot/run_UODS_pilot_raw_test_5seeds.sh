#!/bin/bash
# ==============================================================================
# UODS pilot raw — per-window flow_NLL dump(GPU forward-only). slurm_run.sh의 "test" 슬롯.
# ==============================================================================
CUDA_VISIBLE_DEVICES=0 python3 analysis/dump_uods_window_scores.py \
    --run_name uods_pilot_raw_s2026 \
    --overwrite
