#!/bin/bash
# B-4 KAN-AD 전량 제출 — 24 fold × 5 seed = 120 GPU 잡. n17 idx0-59 / n16 idx60-119, %CONC(기본3).
set -euo pipefail
ROOT="$HOME/DCP/MTGFLOW"; SIF="$HOME/containers/anomaly-base_cu118-u20.sif"; CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"; DISPATCH="runners/Paderborn/B4_kanad/b4_dispatch.sh"; mkdir -p "$LOGDIR"
CONC="${CONC:-3}"; CPUS="${CPUS:-4}"; MEM="${MEM:-16G}"
submit() {  # $1=range $2=node
  local inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && bash $DISPATCH \$SLURM_ARRAY_TASK_ID"
  local wrap="singularity exec --nv $SIF bash -lc \"$inner\""
  local cmd=(sbatch --parsable --partition=V100-16 --gres=gpu:V100-16:1 --nodelist="$2"
             --cpus-per-task="$CPUS" --mem="$MEM" --array="$1" -J "b4_kanad_$2"
             -o "$LOGDIR/%x_%A_%a.out" --wrap "$wrap")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
}
echo "B-4 전량: 120잡, conc=$CONC/node, mem=$MEM" >&2
submit "0-59%${CONC}" n17; submit "60-119%${CONC}" n16
