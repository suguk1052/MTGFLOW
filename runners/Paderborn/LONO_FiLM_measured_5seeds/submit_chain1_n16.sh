#!/bin/bash
# ==============================================================================
# FiLM-measured 체인1 (n16) — 5셀=25run, afterok 순차 체인. 셀=(config×LONO)의 5 seed.
#  제출: bash runners/Paderborn/LONO_FiLM_measured_5seeds/submit_chain1_n16.sh
#  미리보기: DRY_RUN=1 bash runners/Paderborn/LONO_FiLM_measured_5seeds/submit_chain1_n16.sh
# ==============================================================================
set -euo pipefail
DIR="runners/Paderborn/LONO_FiLM_measured_5seeds"

NODELIST=n16 bash runners/slurm_run.sh \
  "$DIR/run_Paderborn_CA_0123_LONO1_FiLM_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_0123_LONO2_FiLM_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_0123_LONO3_FiLM_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_0123_LONO4_FiLM_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_0123_LONO5_FiLM_m_5seeds.sh"
