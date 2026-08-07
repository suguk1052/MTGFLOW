#!/bin/bash
# ==============================================================================
# 작업 F-2 진단 — 단발 Slurm GPU 잡(n17 V100-16 1장). 학습 아님(재추론).
#   analysis/diagnose_F2_evaluation.py 로 학습된 F-2 체크포인트(all/mixed) 재추론 →
#   Vy 기준선(F-1 rmsnorm) 대비 3-way 비교. gate 기본 LONO 1,2.
#   산출: results/Paderborn/diag_F2_evaluation/*.json, reports/report_F2_evaluation.md.
#
#   전제: (1) F-2 gate 체인 완료(submit_F2_gate_chain.sh), (2) F-1 fold JSON 존재
#         (results/Paderborn/diag_F1_evaluation/*.json).
#
#   사용법:  bash runners/Paderborn/LONO_F2_c1c2vy_s2026/submit_F2_eval.sh
#            DRY_RUN=1 bash ...                      # 미리보기
#            LONOS="1 2 3 4 5 6" bash ...            # 전체 확장 진단
# ==============================================================================
set -euo pipefail

ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"
NODELIST="${NODELIST:-n17}"
LONOS="${LONOS:-1 2}"
SLURM_RES=(--partition=V100-16 --gres=gpu:V100-16:1 --nodelist="$NODELIST" --cpus-per-task=10)

mkdir -p "$LOGDIR"

inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && python -u analysis/diagnose_F2_evaluation.py --lonos $LONOS"
wrap="singularity exec --nv $SIF bash -lc \"$inner\""
cmd=(sbatch "${SLURM_RES[@]}" -J F2_eval -o "$LOGDIR/%x_%j.out" --wrap "$wrap")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"; echo
else
  out="$("${cmd[@]}")"; echo "$out"
  jid="$(echo "$out" | grep -oE '[0-9]+' | tail -1)"
  echo "  -> F2_eval submitted as job $jid (log: $LOGDIR/F2_eval_${jid}.out)"
fi
