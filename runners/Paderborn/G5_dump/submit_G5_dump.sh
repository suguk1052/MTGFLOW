#!/bin/bash
# ==============================================================================
# G-5 Step 1 — G-3a per-window score 캐시 dump (forward-only, 학습 없음)를 slurm에 제출.
# analysis/dump_G3a_window_scores.py 를 GPU 1장(V100-16, 기본 n17)에서 실행한다.
# train+test 쌍이 아니라 단일 analysis 명령이므로 slurm_run.sh 대신 이 스크립트로 직접 제출.
#
# 사용법 (MTGFLOW 기준 상대 경로):
#   bash runners/Paderborn/G5_dump/submit_G5_dump.sh              # seed 2026(스크리닝) 제출
#   SEEDS="2024 2025 2027 2028" bash runners/.../submit_G5_dump.sh # 5-seed 확장(4개 추가)
#   NODELIST=n16 bash runners/.../submit_G5_dump.sh               # n16에서 실행
#   DRY_RUN=1   bash runners/.../submit_G5_dump.sh               # 제출 없이 sbatch 명령만 출력
# ==============================================================================
set -euo pipefail

ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"

NODELIST="${NODELIST:-n17}"
SEEDS="${SEEDS:-2026}"                       # 기본 = 1-seed 스크리닝
SLURM_RES=(--partition=V100-16 --gres=gpu:V100-16:1 --nodelist="$NODELIST" --cpus-per-task=10)

mkdir -p "$LOGDIR"

for seed in $SEEDS; do
  job_name="g5_dump_s${seed}"
  inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && python analysis/dump_G3a_window_scores.py --seed $seed"
  wrap="singularity exec --nv $SIF bash -lc \"$inner\""
  cmd=(sbatch "${SLURM_RES[@]}" -J "$job_name" -o "$LOGDIR/%x_%j.out" --wrap "$wrap")

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    out="$("${cmd[@]}")"; echo "$out"
    echo "  -> $job_name submitted (seed=$seed, node=$NODELIST)"
  fi
done
