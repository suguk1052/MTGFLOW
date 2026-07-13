#!/bin/bash
# ==============================================================================
# LONO_C2_measured_5seeds : C2_measured 4개 LOSO split(023to1 -> 012to3 -> 013to2 -> 123to0)
#  x LONO1~6 순차 실행. 각 실험 = (학습 && test) 한 잡, 실험 간 afterok 체인.
#  n17 V100-16 1장 (LONO_B3_5seeds / LONO_C2_5seeds와 동시에 돌리면
#  n17의 GPU 3장에 하나씩 배정됨).
#  실제 제출: bash runners/LONO_C2_measured_5seeds/submit_C2_measured_4splits_5seeds_chain.sh
#  미리보기:  DRY_RUN=1 bash runners/LONO_C2_measured_5seeds/submit_C2_measured_4splits_5seeds_chain.sh
# ==============================================================================
set -euo pipefail
DIR="runners/LONO_C2_measured_5seeds"

bash runners/slurm_run.sh \
  "$DIR/run_Paderborn_CA_023to1_LONO1_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_023to1_LONO2_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_023to1_LONO3_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_023to1_LONO4_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_023to1_LONO5_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_023to1_LONO6_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_012to3_LONO1_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_012to3_LONO2_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_012to3_LONO3_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_012to3_LONO4_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_012to3_LONO5_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_012to3_LONO6_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_013to2_LONO1_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_013to2_LONO2_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_013to2_LONO3_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_013to2_LONO4_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_013to2_LONO5_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_013to2_LONO6_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_123to0_LONO1_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_123to0_LONO2_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_123to0_LONO3_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_123to0_LONO4_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_123to0_LONO5_m_5seeds.sh" \
  "$DIR/run_Paderborn_CA_123to0_LONO6_m_5seeds.sh"
