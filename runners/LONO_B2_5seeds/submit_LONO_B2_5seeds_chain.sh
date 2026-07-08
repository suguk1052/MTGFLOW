#!/bin/bash
# ==============================================================================
# LONO_B2_5seeds : LONO3 -> LONO4 -> LONO5 -> LONO6 순차 실행
#  각 실험 = (학습 && test) 한 잡, 실험 간 afterok 체인. n17 V100 1장.
#  실제 제출: bash runners/LONO_B2_5seeds/submit_LONO_B2_5seeds_chain.sh
#  미리보기:  DRY_RUN=1 bash runners/LONO_B2_5seeds/submit_LONO_B2_5seeds_chain.sh
# ==============================================================================
set -euo pipefail
DIR="runners/LONO_B2_5seeds"

bash runners/slurm_run.sh \
  "$DIR/run_Paderborn_0123_LONO3_5seeds.sh" \
  "$DIR/run_Paderborn_0123_LONO4_5seeds.sh" \
  "$DIR/run_Paderborn_0123_LONO5_5seeds.sh" \
  "$DIR/run_Paderborn_0123_LONO6_5seeds.sh"
