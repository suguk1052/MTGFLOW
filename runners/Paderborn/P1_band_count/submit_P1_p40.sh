#!/bin/bash
# ==============================================================================
# P-1 학습을 P40 노드(n8~n12)로 제출. 러너 .sh는 GPU 무관(CUDA_VISIBLE_DEVICES=0)이라 그대로 재사용.
#  용도: V100(n16·n17)만으로 느릴 때 idle P40으로 오프로드해 wall-time 단축.
#  주의: P40은 Pascal FP32(텐서코어 없음) → V100 대비 느릴 수 있음(배율 k는 로그로 확인).
#  인자: 공백구분 run_name 목록(예: p1_n24_123to0_LONO5). 각각 짝 test까지 train&&test 체인.
#  노드: 기본 n8 n9 (P40NODES 로 변경). resume-safe(s2028 metrics 존재 시 skip).
#
#  사용:
#   bash runners/Paderborn/P1_band_count/submit_P1_p40.sh p1_n24_123to0_LONO5 p1_n24_023to1_LONO1 ...
#   P40NODES="n8 n9 n10 n11" bash ... <run_names>
#   DRY_RUN=1 bash ... <run_names>
# ==============================================================================
set -euo pipefail

ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"
BASE="runners/Paderborn/P1_band_count"
RESULTS="$ROOT/results/Paderborn"

read -r -a NODES <<< "${P40NODES:-n8 n9}"
mkdir -p "$LOGDIR"

i=0; submitted=0; skipped=0
for run_name in "$@"; do
  # run_name = p1_n{N}_{split}_LONO{lono}  → N 추출로 러너 디렉토리 결정.
  N=$(echo "$run_name" | sed -E 's/^p1_n([0-9]+)_.*/\1/')
  DIR="$BASE/n${N}"
  train="$DIR/run_Paderborn_${run_name}_5seeds.sh"
  test_rel="$DIR/run_Paderborn_${run_name}_test_5seeds.sh"
  [[ -f "$ROOT/$train" && -f "$ROOT/$test_rel" ]] || { echo "러너 없음: $run_name" >&2; exit 1; }

  if [[ -f "$RESULTS/${run_name}_s2028/paderborn_per_bearing_metrics.json" ]]; then
    echo "[skip] $run_name (s2028 이미 존재)"; skipped=$((skipped+1)); continue
  fi

  node="${NODES[$((i % ${#NODES[@]}))]}"; i=$((i+1))
  res=(--partition=P40 --gres=gpu:P40:1 --nodelist="$node" --cpus-per-task=10)
  inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && bash '$train' && bash '$test_rel'"
  wrap="singularity exec --nv $SIF bash -lc \"$inner\""
  cmd=(sbatch "${res[@]}" -J "run_Paderborn_${run_name}_5seeds" -o "$LOGDIR/%x_%j.out" --wrap "$wrap")

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[$node/P40] $run_name"
  else
    out="$("${cmd[@]}")"; jid="$(echo "$out" | grep -oE '[0-9]+' | tail -1)"
    echo "  -> $run_name (P40) on $node submitted as job $jid"
  fi
  submitted=$((submitted+1))
done
echo "==== P40 제출 요약: submitted=$submitted, skipped=$skipped ===="
