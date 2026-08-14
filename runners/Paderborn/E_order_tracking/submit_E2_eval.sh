#!/bin/bash
# 작업 E-2 평가 잡 — OT × RMS-norm 2×2 재추론(diagnose_E2_rmsnorm_ot_eval.py). n17 1 GPU.
#   8 rmsnorm 학습잡 완료 후 실행. raw/raw+OT/rmsnorm(s2026)/rmsnorm+OT는 기존 체크포인트 재사용.
set -euo pipefail
ROOT="$HOME/DCP/MTGFLOW"; SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"; LOGDIR="$ROOT/runners/slurm_logs"
NODELIST="${NODELIST:-n17}"; mkdir -p "$LOGDIR"
inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && python analysis/diagnose_E2_rmsnorm_ot_eval.py"
wrap="singularity exec --nv $SIF bash -lc \"$inner\""
sbatch --partition=V100-16 --gres=gpu:V100-16:1 --nodelist="$NODELIST" --cpus-per-task=10 \
  -J E2_eval -o "$LOGDIR/%x_%j.out" --wrap "$wrap"
