#!/bin/bash
# 작업 E-1 Phase B 평가 잡 — raw vs raw+OT 재추론(diagnose_E_order_tracking_eval.py). n17 1 GPU.
set -euo pipefail
ROOT="$HOME/DCP/MTGFLOW"; SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"; LOGDIR="$ROOT/runners/slurm_logs"
NODELIST="${NODELIST:-n17}"; mkdir -p "$LOGDIR"
inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && python analysis/diagnose_E_order_tracking_eval.py"
wrap="singularity exec --nv $SIF bash -lc \"$inner\""
sbatch --partition=V100-16 --gres=gpu:V100-16:1 --nodelist="$NODELIST" --cpus-per-task=10 \
  -J E_eval -o "$LOGDIR/%x_%j.out" --wrap "$wrap"
