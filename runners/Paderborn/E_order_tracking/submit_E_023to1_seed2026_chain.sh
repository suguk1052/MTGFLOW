#!/bin/bash
# ==============================================================================
# 작업 E 1단계 Phase B 배치 제출 — 023→1 LONO1/LONO2 raw seed2026 학습(+raw 평가).
# 각 잡 = (main.py 학습 && test.py raw 평가). LONO1 → LONO2 afterok 체인.
# raw+OT / per-fault 비교는 학습 완료 후 analysis/diagnose_E_order_tracking_eval.py로 수행.
# 사용: bash runners/Paderborn/E_order_tracking/submit_E_023to1_seed2026_chain.sh
#       DRY_RUN=1 ... 로 미리보기.
# ==============================================================================
set -euo pipefail
cd "$HOME/DCP/MTGFLOW"

bash runners/slurm_run.sh \
  runners/Paderborn/E_order_tracking/run_Paderborn_E_023to1_LONO1_5seeds.sh \
  runners/Paderborn/E_order_tracking/run_Paderborn_E_023to1_LONO2_5seeds.sh
