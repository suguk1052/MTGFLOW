#!/bin/bash
# ==============================================================================
# LONO_D_ampnorm_s2026 : 작업 D(진폭 confound 교정, Route 2) no-meta, seed 2026.
#  4 LOSO split(023to1 -> 012to3 -> 013to2 -> 123to0) x LONO1~6 = 24 fold.
#  각 실험 = (학습(--amp_normalize) && test) 한 잡, 실험 간 afterok 체인. n17 V100-16 1장.
#  실제 제출: bash runners/Paderborn/LONO_D_ampnorm_s2026/submit_D_ampnorm_s2026_chain.sh
#  미리보기:  DRY_RUN=1 bash runners/Paderborn/LONO_D_ampnorm_s2026/submit_D_ampnorm_s2026_chain.sh
#  게이트 진단(학습 완료 후): conda run -n mtgflow python analysis/diagnose_amp_correction.py
# ==============================================================================
set -euo pipefail
DIR="runners/Paderborn/LONO_D_ampnorm_s2026"

bash runners/slurm_run.sh \
  "$DIR/run_Paderborn_ampnorm_023to1_LONO1_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_023to1_LONO2_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_023to1_LONO3_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_023to1_LONO4_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_023to1_LONO5_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_023to1_LONO6_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_012to3_LONO1_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_012to3_LONO2_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_012to3_LONO3_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_012to3_LONO4_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_012to3_LONO5_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_012to3_LONO6_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_013to2_LONO1_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_013to2_LONO2_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_013to2_LONO3_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_013to2_LONO4_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_013to2_LONO5_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_013to2_LONO6_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_123to0_LONO1_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_123to0_LONO2_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_123to0_LONO3_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_123to0_LONO4_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_123to0_LONO5_5seeds.sh" \
  "$DIR/run_Paderborn_ampnorm_123to0_LONO6_5seeds.sh"
