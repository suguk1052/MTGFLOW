#!/bin/bash
# ==============================================================================
# FiLM-measured 전체 6체인 제출 (체인1~3 n16, 체인4~6 n17). 각 체인=독립 afterok 체인.
#  ★ 제출 전 sinfo로 n16이 V100-16 파티션에 정상 스케줄되는지 확인할 것.
#  미리보기: DRY_RUN=1 bash runners/LONO_FiLM_measured_5seeds/submit_all_6chains.sh
# ==============================================================================
set -euo pipefail
DIR="runners/LONO_FiLM_measured_5seeds"

DRY_RUN="${DRY_RUN:-0}" bash "$DIR/submit_chain1_n16.sh"
DRY_RUN="${DRY_RUN:-0}" bash "$DIR/submit_chain2_n16.sh"
DRY_RUN="${DRY_RUN:-0}" bash "$DIR/submit_chain3_n16.sh"
DRY_RUN="${DRY_RUN:-0}" bash "$DIR/submit_chain4_n17.sh"
DRY_RUN="${DRY_RUN:-0}" bash "$DIR/submit_chain5_n17.sh"
DRY_RUN="${DRY_RUN:-0}" bash "$DIR/submit_chain6_n17.sh"
