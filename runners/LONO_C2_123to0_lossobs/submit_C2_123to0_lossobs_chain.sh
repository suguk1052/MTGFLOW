#!/bin/bash
# ==============================================================================
# LONO_C2_123to0_lossobs : 123to0 split의 C2(static-meta) 6개 LONO를 "고치기"가
#  아니라 "무너지는 과정을 관찰"하기 위해 재실행. main.py에 추가한 관찰용
#  train_log.jsonl(+ --log_test_auroc)로 epoch별 train/val loss, test AUROC를
#  기록한다. 모델/정규화는 기존 LONO_C2_5seeds와 동일 - 결과 재현 확인 + 실패
#  시드의 loss curve 확보가 목적. RUN_NAME에 _lossobs를 붙여 기존 결과와 분리.
#  실제 제출: bash runners/LONO_C2_123to0_lossobs/submit_C2_123to0_lossobs_chain.sh
#  미리보기:  DRY_RUN=1 bash runners/LONO_C2_123to0_lossobs/submit_C2_123to0_lossobs_chain.sh
# ==============================================================================
set -euo pipefail
DIR="runners/LONO_C2_123to0_lossobs"

bash runners/slurm_run.sh \
  "$DIR/run_Paderborn_CA_123to0_LONO1_lossobs_5seeds.sh" \
  "$DIR/run_Paderborn_CA_123to0_LONO2_lossobs_5seeds.sh" \
  "$DIR/run_Paderborn_CA_123to0_LONO3_lossobs_5seeds.sh" \
  "$DIR/run_Paderborn_CA_123to0_LONO4_lossobs_5seeds.sh" \
  "$DIR/run_Paderborn_CA_123to0_LONO5_lossobs_5seeds.sh" \
  "$DIR/run_Paderborn_CA_123to0_LONO6_lossobs_5seeds.sh"
