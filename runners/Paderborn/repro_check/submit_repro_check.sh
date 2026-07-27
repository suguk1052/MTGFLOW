#!/bin/bash
# ==============================================================================
# 재현성 검증 잡 제출 (n17 V100-16 1장 고정). runners/slurm_run.sh와 동일한
# singularity+conda 패턴을 쓰되, train/test 파일명 매칭 규칙 없이 단일 드라이버
# (run_repro_check.sh)를 그대로 실행한다.
#
# 사용법:
#   bash runners/Paderborn/repro_check/submit_repro_check.sh
#   DRY_RUN=1 bash runners/Paderborn/repro_check/submit_repro_check.sh   # 미리보기만
# ==============================================================================
set -euo pipefail

ROOT="/home/dayoon/DCP/MTGFLOW"
SIF="/home/dayoon/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="/home/dayoon/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"

SLURM_RES=(--partition=V100-16 --gres=gpu:V100-16:1 --nodelist=n17 --cpus-per-task=10)

mkdir -p "$LOGDIR"

inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && bash runners/Paderborn/repro_check/run_repro_check.sh"
wrap="singularity exec --nv $SIF bash -lc \"$inner\""

cmd=(sbatch "${SLURM_RES[@]}" -J repro_check -o "$LOGDIR/%x_%j.out" --wrap "$wrap")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"; echo
else
  out="$("${cmd[@]}")"; echo "$out"
fi
