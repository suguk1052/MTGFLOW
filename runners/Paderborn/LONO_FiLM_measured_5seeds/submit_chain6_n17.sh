#!/bin/bash
# ==============================================================================
# FiLM-measured 체인6 (n17) — 5셀=25run, afterok 순차 체인. 셀=(config×LONO)의 5 seed.
#  제출: bash runners/Paderborn/LONO_FiLM_measured_5seeds/submit_chain6_n17.sh
#  미리보기: DRY_RUN=1 bash runners/Paderborn/LONO_FiLM_measured_5seeds/submit_chain6_n17.sh
# ==============================================================================
set -euo pipefail
DIR="runners/Paderborn/LONO_FiLM_measured_5seeds"

NODELIST=n17 bash runners/slurm_run.sh \
  "$DIR/run_Paderborn_CA_013to2_LONO2_FiLM_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_013to2_LONO3_FiLM_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_013to2_LONO4_FiLM_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_013to2_LONO5_FiLM_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_013to2_LONO6_FiLM_m_5seeds.sh"
