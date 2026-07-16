#!/bin/bash
# ==============================================================================
# FiLM-measured 체인2 (n16) — 5셀=25run, afterok 순차 체인. 셀=(config×LONO)의 5 seed.
#  제출: bash runners/LONO_FiLM_measured_5seeds/submit_chain2_n16.sh
#  미리보기: DRY_RUN=1 bash runners/LONO_FiLM_measured_5seeds/submit_chain2_n16.sh
# ==============================================================================
set -euo pipefail
DIR="runners/LONO_FiLM_measured_5seeds"

NODELIST=n16 bash runners/slurm_run.sh \
  "$DIR/run_Paderborn_CA_0123_LONO6_FiLM_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_123to0_LONO1_FiLM_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_123to0_LONO2_FiLM_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_123to0_LONO3_FiLM_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_123to0_LONO4_FiLM_m_5seeds.sh"
