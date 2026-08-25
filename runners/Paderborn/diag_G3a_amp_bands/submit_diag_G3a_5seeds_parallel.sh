#!/bin/bash
# ==============================================================================
# 작업 G-3a(K=6 band amplitude) 5-seed 정밀 진단 재추론 병렬 제출.
#  - 각 잡 = 한 seed에 대해 24 fold(4 LOSO × 6 LONO) 재추론:
#      analysis/diagnose_G3a_amp_bands.py --seed <s>
#    (g3a checkpoint S_shape/S_amp/S_total + B3 raw paired) → results/Paderborn/diag_G3a_amp_bands/aggregate_s<s>.json
#  - 학습 아님(무학습 재추론). GPU 1장/잡, node round-robin(n17,n16). 5 잡 병렬.
#  - 이미 aggregate_s<s>.json 있으면 스킵.
#  실제 제출: bash runners/Paderborn/diag_G3a_amp_bands/submit_diag_G3a_5seeds_parallel.sh
#  미리보기:  DRY_RUN=1 bash runners/Paderborn/diag_G3a_amp_bands/submit_diag_G3a_5seeds_parallel.sh
#  집계 리포트(제출 후, CPU): conda run -n mtgflow python analysis/report_G3a_5seeds.py
# ==============================================================================
set -euo pipefail

ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"
OUTDIR="$ROOT/results/Paderborn/diag_G3a_amp_bands"

NODES=(n17 n16)
SEEDS=(2024 2025 2026 2027 2028)

mkdir -p "$LOGDIR" "$OUTDIR"

i=0
submitted=0
skipped=0
for seed in "${SEEDS[@]}"; do
  if [[ -f "$OUTDIR/aggregate_s${seed}.json" ]]; then
    echo "[skip] seed $seed (aggregate_s${seed}.json 이미 존재)"
    skipped=$((skipped+1))
    continue
  fi

  node="${NODES[$((i % ${#NODES[@]}))]}"
  i=$((i+1))

  res=(--partition=V100-16 --gres=gpu:V100-16:1 --nodelist="$node" --cpus-per-task=10)
  inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && python analysis/diagnose_G3a_amp_bands.py --seed $seed"
  wrap="singularity exec --nv $SIF bash -lc \"$inner\""
  cmd=(sbatch "${res[@]}" -J "diag_G3a_s${seed}" -o "$LOGDIR/%x_%j.out" --wrap "$wrap")

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[$node] seed $seed:"; printf '  %q ' "${cmd[@]}"; echo
  else
    out="$("${cmd[@]}")"; jid="$(echo "$out" | grep -oE '[0-9]+' | tail -1)"
    echo "  -> diag_G3a seed $seed on $node submitted as job $jid"
  fi
  submitted=$((submitted+1))
done

echo ""
echo "제출: $submitted, 스킵(완료): $skipped"
echo "완료 후 5-seed 집계: conda run -n mtgflow python analysis/report_G3a_5seeds.py"
