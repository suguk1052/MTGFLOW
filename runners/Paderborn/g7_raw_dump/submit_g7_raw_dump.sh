#!/bin/bash
# ==============================================================================
# G-7 Step 0 — raw(B3) per-window score dump 를 GPU로 제출 (seed별 1잡, 병렬).
#  - slurm_run.sh 는 train/test 쌍(_5seeds.sh↔_test_5seeds.sh) 전용이라 단일 analysis
#    잡에는 못 씀. 그 안의 singularity exec --nv ... bash -lc '...' 패턴만 그대로 차용.
#  - 잡 하나 = 한 seed 의 24 fold(4 split × 6 LONO) forward-only 재추론 (학습 아님).
#  - 노드는 허락된 n17(3장)·n16(2장)만: seed 2024/2025/2026→n17, 2027/2028→n16.
#
# 사용법 (MTGFLOW 기준):
#   DRY_RUN=1 bash runners/Paderborn/g7_raw_dump/submit_g7_raw_dump.sh   # 미리보기
#   bash runners/Paderborn/g7_raw_dump/submit_g7_raw_dump.sh            # 실제 제출
# ==============================================================================
set -euo pipefail

ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"
mkdir -p "$LOGDIR"

SEEDS=(2024 2025 2026 2027 2028)

node_for_seed() {  # n17 3장 + n16 2장으로 분산
  case "$1" in
    2024|2025|2026) echo "n17" ;;
    *)              echo "n16" ;;
  esac
}

for seed in "${SEEDS[@]}"; do
  node="$(node_for_seed "$seed")"
  job_name="g7rawdump_s${seed}"
  inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && python analysis/dump_B3_raw_window_scores.py --seeds $seed"
  wrap="singularity exec --nv $SIF bash -lc \"$inner\""
  cmd=(sbatch --partition=V100-16 --gres=gpu:V100-16:1 --nodelist="$node" --cpus-per-task=10 \
       -J "$job_name" -o "$LOGDIR/%x_%j.out" --wrap "$wrap")

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    out="$("${cmd[@]}")"; echo "$out"
    jid="$(echo "$out" | grep -oE '[0-9]+' | tail -1)"
    echo "  -> $job_name on $node submitted as job $jid"
  fi
done
