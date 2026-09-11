#!/bin/bash
# ==============================================================================
# B-2 Deep SVDD 전량 제출 — 24 fold × 5 seed = 120 GPU 잡.
#   index 0-59 → n17, 60-119 → n16. 노드당 동시 CONC(기본 3=GPU 상한).
#   OOM/메모리 폴백: CONC=2 로 재제출(미완료 fold는 deep_svdd.py가 스킵).
# 사용:
#   bash runners/Paderborn/B2_deep_svdd/submit_b2_full.sh          # %3 (기본)
#   CONC=2 bash .../submit_b2_full.sh                              # 폴백(노드당 2 GPU)
#   DRY_RUN=1 bash .../submit_b2_full.sh                          # 미리보기
# ==============================================================================
set -euo pipefail
ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"
DISPATCH="runners/Paderborn/B2_deep_svdd/b2_dispatch.sh"
mkdir -p "$LOGDIR"

CONC="${CONC:-3}"      # 노드당 동시 GPU 잡(기본 3=GPU 상한, 폴백 2)
CPUS="${CPUS:-4}"
MEM="${MEM:-12G}"      # 파일럿 host MaxRSS 8.5GB → 12G 여유

submit() {  # $1=array_range  $2=node
  local arr="$1" node="$2"
  local inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && bash $DISPATCH \$SLURM_ARRAY_TASK_ID"
  local wrap="singularity exec --nv $SIF bash -lc \"$inner\""
  local cmd=(sbatch --parsable --partition=V100-16 --gres=gpu:V100-16:1 --nodelist="$node"
             --cpus-per-task="$CPUS" --mem="$MEM" --array="$arr" -J "b2_dsvdd_${node}"
             -o "$LOGDIR/%x_%A_%a.out" --wrap "$wrap")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
}

echo "B-2 전량 제출: 120잡(24fold×5seed), conc=$CONC/node, cpus=$CPUS, mem=$MEM" >&2
jid1=$(submit "0-59%${CONC}"   n17)
jid2=$(submit "60-119%${CONC}" n16)
echo "$jid1"; echo "$jid2"
