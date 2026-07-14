#!/bin/bash
# ==============================================================================
# 대조군 4개(B3 123to0_LONO4 / B3 012to3_LONO4 / B2 pooled_LONO4 /
# C1_measured pooled_LONO4)를 "독립 잡"으로 동시 제출한다 (slurm_run.sh의
# afterok 순차 체인이 아니라 병렬). 목적: no-meta/pooled도 val_loss<->test_AUROC
# 괴리를 겪는지 최소 대조로 확인.
#
# ⚠️ 이 스크립트는 기본 정책(n17 V100-16 1장 고정, runners/slurm_run.sh)의
# 예외다. 사용자가 이번 대조군 실험 한정으로 "GPU 3장(n17 외 유휴 노드 포함)
# 병렬"을 명시적으로 승인했으므로 --nodelist를 고정하지 않고 V100-16 파티션
# 안에서 스케줄러가 유휴 노드(제출 시점 n13/n14/n17)에 자동 배정하게 한다.
# slurm_run.sh 자체는 건드리지 않음 - 이후 실험은 기존처럼 n17 고정 사용.
#
# 실제 제출: bash runners/contrast_lossobs/submit_contrast_parallel.sh
# 미리보기:  DRY_RUN=1 bash runners/contrast_lossobs/submit_contrast_parallel.sh
# ==============================================================================
set -euo pipefail

ROOT="/home/dayoon/DCP/MTGFLOW"
SIF="/home/dayoon/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="/home/dayoon/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"
DIR="runners/contrast_lossobs"

# n17 고정 없이 V100-16 파티션 내 유휴 노드에 스케줄러가 자동 배정하도록 함
SLURM_RES=(--partition=V100-16 --gres=gpu:V100-16:1 --cpus-per-task=10)

mkdir -p "$LOGDIR"

JOBS=(
  "run_B3_123to0_LONO4_lossobs3s|run_B3_123to0_LONO4_test_lossobs3s"
  "run_B3_012to3_LONO4_lossobs3s|run_B3_012to3_LONO4_test_lossobs3s"
  "run_B2_pooled_LONO4_lossobs3s|run_B2_pooled_LONO4_test_lossobs3s"
  "run_C1_measured_pooled_LONO4_lossobs3s|run_C1_measured_pooled_LONO4_test_lossobs3s"
)

for pair in "${JOBS[@]}"; do
  train_name="${pair%%|*}"
  test_name="${pair##*|}"
  train="$DIR/${train_name}.sh"
  test_rel="$DIR/${test_name}.sh"

  [[ -f "$ROOT/$train"    ]] || { echo "학습 스크립트 없음: $ROOT/$train" >&2; exit 1; }
  [[ -f "$ROOT/$test_rel" ]] || { echo "test 스크립트 없음: $ROOT/$test_rel" >&2; exit 1; }

  inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && bash '$train' && bash '$test_rel'"
  wrap="singularity exec --nv $SIF bash -lc \"$inner\""

  cmd=(sbatch "${SLURM_RES[@]}" -J "$train_name" -o "$LOGDIR/%x_%j.out" --wrap "$wrap")

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    out="$("${cmd[@]}")"; echo "$out"
    echo "  -> $train_name (train&&test) submitted"
  fi
done
