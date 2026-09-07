#!/bin/bash
# ==============================================================================
# P-1 per-window score dump 제출 (forward-only, 학습 없음). analysis/dump_G3a_window_scores.py 사용.
#  대상 N: 1(=G-1 체크포인트 재사용) · 2·3·4·8·12·24(P-1 재학습분). N=6은 g3a_window_scores 캐시 재사용 → 제외.
#  각 (N, seed) 호출 = 24 fold forward-only. seed 5개(2024~2028) × N 7개 = 35 호출.
#  ⚠️ 해당 N의 학습이 끝난 뒤(체크포인트 존재) 실행. dump 스크립트가 npz 이미 있으면 자동 skip.
#
#  사용:
#   bash runners/Paderborn/P1_band_count/submit_P1_dump.sh              # 기본 N·seed 전체
#   NS="1" bash ...                                                     # N=1만(g1 재사용, 즉시 가능)
#   NS="2 3" SEEDS="2026" bash ...                                      # 부분
#   DRY_RUN=1 bash ...                                                  # 미리보기
# ==============================================================================
set -euo pipefail

ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"

NODES=(n17 n16)
NS="${NS:-1 2 3 4 8 12 24}"
SEEDS="${SEEDS:-2024 2025 2026 2027 2028}"

mkdir -p "$LOGDIR"

i=0
submitted=0
for N in $NS; do
  # N별 입력 체크포인트 prefix/배치폴더, 출력 npz 폴더.
  if [[ "$N" == "1" ]]; then
    prefix="g1"; batch_dir="LONO_G1_s2026"; out_dir="results/Paderborn/g1_window_scores"
  else
    prefix="p1_n${N}"; batch_dir=""; out_dir="results/Paderborn/p1_n${N}_window_scores"
  fi
  for seed in $SEEDS; do
    node="${NODES[$((i % ${#NODES[@]}))]}"
    i=$((i+1))

    args="--seed $seed --g3a_prefix $prefix --out_dir $out_dir"
    [[ -n "$batch_dir" ]] && args="$args --g3a_batch_dir $batch_dir"
    inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && python analysis/dump_G3a_window_scores.py $args"
    wrap="singularity exec --nv $SIF bash -lc \"$inner\""
    job_name="p1_dump_n${N}_s${seed}"
    cmd=(sbatch --partition=V100-16 --gres=gpu:V100-16:1 --nodelist="$node" --cpus-per-task=10 -J "$job_name" -o "$LOGDIR/%x_%j.out" --wrap "$wrap")

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "[$node] $job_name  ($args)"
    else
      out="$("${cmd[@]}")"; jid="$(echo "$out" | grep -oE '[0-9]+' | tail -1)"
      echo "  -> $job_name on $node submitted as job $jid"
    fi
    submitted=$((submitted+1))
  done
done

echo "==== dump 제출 요약: submitted=$submitted ===="
