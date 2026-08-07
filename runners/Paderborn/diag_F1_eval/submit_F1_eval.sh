#!/bin/bash
# ==============================================================================
# 작업 F-1 정밀 재평가 — 단발 Slurm GPU 잡(n17 V100-16 1장).
#   analysis/diagnose_F1_evaluation.py 를 재추론(무학습)으로 전 24 fold(4 LOSO × 6 LONO)
#   rmsnorm(D) vs raw(B3) paired 평가 → results/Paderborn/diag_F1_evaluation/*.json,
#   reports/report_F1_evaluation.md 생성.
#   slurm_run.sh의 singularity --nv 패턴을 미러링(학습 아님 → afterok 체인·test 짝 불필요).
#
# 사용법:  bash runners/Paderborn/diag_F1_eval/submit_F1_eval.sh
#          DRY_RUN=1 bash runners/Paderborn/diag_F1_eval/submit_F1_eval.sh   # 미리보기
# ==============================================================================
set -euo pipefail

ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"
NODELIST="${NODELIST:-n17}"
SLURM_RES=(--partition=V100-16 --gres=gpu:V100-16:1 --nodelist="$NODELIST" --cpus-per-task=10)

mkdir -p "$LOGDIR"

inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && python -u analysis/diagnose_F1_evaluation.py"
wrap="singularity exec --nv $SIF bash -lc \"$inner\""

cmd=(sbatch "${SLURM_RES[@]}" -J F1_eval -o "$LOGDIR/%x_%j.out" --wrap "$wrap")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"; echo
else
  out="$("${cmd[@]}")"; echo "$out"
  jid="$(echo "$out" | grep -oE '[0-9]+' | tail -1)"
  echo "  -> F1_eval submitted as job $jid (log: $LOGDIR/F1_eval_${jid}.out)"
fi
