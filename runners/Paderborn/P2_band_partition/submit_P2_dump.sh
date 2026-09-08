#!/bin/bash
# 작업 P-2: log·energy 전체 dump(24 fold × 5 seed). scheme×seed당 1잡(각 24 fold 일괄), n17/n16 분산.
# 사용: bash runners/Paderborn/P2_band_partition/submit_P2_dump.sh   (DRY_RUN=1로 미리보기)
set -euo pipefail
ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"
NODES=(n17 n16)
SCHEMES="${SCHEMES:-energy log}"
SEEDS="${SEEDS:-2024 2025 2026 2027 2028}"
mkdir -p "$LOGDIR"
i=0; submitted=0
for scheme in $SCHEMES; do
  prefix="p2_${scheme}"; out_dir="results/Paderborn/p2_${scheme}_window_scores"
  for seed in $SEEDS; do
    node="${NODES[$((i % ${#NODES[@]}))]}"; i=$((i+1))
    args="--seed $seed --g3a_prefix $prefix --out_dir $out_dir"
    inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && python analysis/dump_G3a_window_scores.py $args"
    wrap="singularity exec --nv $SIF bash -lc \"$inner\""
    job_name="p2_dump_${scheme}_s${seed}"
    cmd=(sbatch --partition=V100-16 --gres=gpu:V100-16:1 --nodelist="$node" --cpus-per-task=10 -J "$job_name" -o "$LOGDIR/%x_%j.out" --wrap "$wrap")
    if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "[$node] $job_name ($args)"; else
      out="$("${cmd[@]}")"; jid="$(echo "$out" | grep -oE '[0-9]+' | tail -1)"; echo "  -> $job_name on $node job $jid"; fi
    submitted=$((submitted+1))
  done
done
echo "==== P2 dump 제출: submitted=$submitted ===="
