#!/bin/bash
# ==============================================================================
# LOSO_static_measured_lossobs : 나머지 3개 LOSO split(023to1/013to2/012to3)의
#  static+measured x LONO1,4,6 x 3seed(--log_test_auroc)를 n17 GPU 3장에
#  3개의 독립 체인(각 6개 실험, afterok 순차)으로 병렬 제출한다.
#  목적: val_loss<->test_AUROC gap이 123to0만의 현상인지 LOSO 전반의 현상인지 확정.
#  (LONO6은 일반성 확인을 위해 추가 - 버퍼시간 감소 감수)
#  처방 실험(checkpoint 기준 변경 등)은 포함하지 않음 - 조사 전용.
#
#  기존 runners/slurm_run.sh를 그대로 재사용(n17 고정, --dependency=afterok 체인).
#  다른 노드는 전혀 건드리지 않음.
#
#  실제 제출: bash runners/LOSO_static_measured_lossobs/submit_3chains_n17.sh
#  미리보기:  DRY_RUN=1 bash runners/LOSO_static_measured_lossobs/submit_3chains_n17.sh
# ==============================================================================
set -euo pipefail
DIR="runners/LOSO_static_measured_lossobs"

echo "=== Chain 1: 023to1 (static+measured x LONO1,4,6) ==="
bash runners/slurm_run.sh \
  "$DIR/run_Paderborn_CA_023to1_LONO1_static_lossobs3s_5seeds.sh" \
  "$DIR/run_Paderborn_CA_023to1_LONO1_measured_lossobs3s_5seeds.sh" \
  "$DIR/run_Paderborn_CA_023to1_LONO4_static_lossobs3s_5seeds.sh" \
  "$DIR/run_Paderborn_CA_023to1_LONO4_measured_lossobs3s_5seeds.sh" \
  "$DIR/run_Paderborn_CA_023to1_LONO6_static_lossobs3s_5seeds.sh" \
  "$DIR/run_Paderborn_CA_023to1_LONO6_measured_lossobs3s_5seeds.sh"

echo "=== Chain 2: 013to2 (static+measured x LONO1,4,6) ==="
bash runners/slurm_run.sh \
  "$DIR/run_Paderborn_CA_013to2_LONO1_static_lossobs3s_5seeds.sh" \
  "$DIR/run_Paderborn_CA_013to2_LONO1_measured_lossobs3s_5seeds.sh" \
  "$DIR/run_Paderborn_CA_013to2_LONO4_static_lossobs3s_5seeds.sh" \
  "$DIR/run_Paderborn_CA_013to2_LONO4_measured_lossobs3s_5seeds.sh" \
  "$DIR/run_Paderborn_CA_013to2_LONO6_static_lossobs3s_5seeds.sh" \
  "$DIR/run_Paderborn_CA_013to2_LONO6_measured_lossobs3s_5seeds.sh"

echo "=== Chain 3: 012to3 (static+measured x LONO1,4,6) ==="
bash runners/slurm_run.sh \
  "$DIR/run_Paderborn_CA_012to3_LONO1_static_lossobs3s_5seeds.sh" \
  "$DIR/run_Paderborn_CA_012to3_LONO1_measured_lossobs3s_5seeds.sh" \
  "$DIR/run_Paderborn_CA_012to3_LONO4_static_lossobs3s_5seeds.sh" \
  "$DIR/run_Paderborn_CA_012to3_LONO4_measured_lossobs3s_5seeds.sh" \
  "$DIR/run_Paderborn_CA_012to3_LONO6_static_lossobs3s_5seeds.sh" \
  "$DIR/run_Paderborn_CA_012to3_LONO6_measured_lossobs3s_5seeds.sh"
