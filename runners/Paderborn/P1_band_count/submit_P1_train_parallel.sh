#!/bin/bash
# ==============================================================================
# P-1(Band count N sweep) 학습 전량 병렬 제출.
#  대상: N ∈ {2,3,4,8,12,24} × 4 split × 6 LONO = 144 잡(각 잡 = 1 fold의 5-seed(2024~2028) train&&test).
#  N=1(=G-1)·N=6(=G-3a)은 재학습 없이 기존 체크포인트 재사용 → 여기 제외.
#  - node round-robin(n17,n16, 각 GPU 3장 → 최대 6 병렬). 독립 제출(afterok 체인 없음).
#  - 이미 완료된 fold(s2028 metrics 존재)는 자동 스킵(resume-safe).
#  선행: gen_p1_runners.py 로 n{N}/ 러너 생성 완료.
#  실제 제출: bash runners/Paderborn/P1_band_count/submit_P1_train_parallel.sh
#  미리보기:  DRY_RUN=1 bash runners/Paderborn/P1_band_count/submit_P1_train_parallel.sh
#  N 부분집합: NS="2 3" bash ...   (기본 = 2 3 4 8 12 24)
# ==============================================================================
set -euo pipefail

ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"
BASE="runners/Paderborn/P1_band_count"
RESULTS="$ROOT/results/Paderborn"

NODES=(n17 n16)
NS="${NS:-2 3 4 8 12 24}"
SPLITS=(123to0 023to1 013to2 012to3)
LONOS=(1 2 3 4 5 6)

mkdir -p "$LOGDIR"

i=0
submitted=0
skipped=0
for N in $NS; do
  DIR="$BASE/n${N}"
  for split in "${SPLITS[@]}"; do
    for lono in "${LONOS[@]}"; do
      run_name="p1_n${N}_${split}_LONO${lono}"
      train="$DIR/run_Paderborn_p1_n${N}_${split}_LONO${lono}_5seeds.sh"
      test_rel="$DIR/run_Paderborn_p1_n${N}_${split}_LONO${lono}_test_5seeds.sh"
      [[ -f "$ROOT/$train"    ]] || { echo "학습 스크립트 없음: $ROOT/$train (gen_p1_runners.py 먼저 실행)" >&2; exit 1; }
      [[ -f "$ROOT/$test_rel" ]] || { echo "test 스크립트 없음: $ROOT/$test_rel (gen_p1_runners.py 먼저 실행)" >&2; exit 1; }

      # 5 seed 중 마지막(2028) metrics 존재 시 스킵(resume-safe).
      if [[ -f "$RESULTS/${run_name}_s2028/paderborn_per_bearing_metrics.json" ]]; then
        echo "[skip] $run_name (s2028 metrics 이미 존재)"
        skipped=$((skipped+1))
        continue
      fi

      node="${NODES[$((i % ${#NODES[@]}))]}"
      i=$((i+1))

      res=(--partition=V100-16 --gres=gpu:V100-16:1 --nodelist="$node" --cpus-per-task=10)
      inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && bash '$train' && bash '$test_rel'"
      wrap="singularity exec --nv $SIF bash -lc \"$inner\""
      cmd=(sbatch "${res[@]}" -J "run_Paderborn_${run_name}_5seeds" -o "$LOGDIR/%x_%j.out" --wrap "$wrap")

      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[$node] $run_name"
      else
        out="$("${cmd[@]}")"; jid="$(echo "$out" | grep -oE '[0-9]+' | tail -1)"
        echo "  -> $run_name (train&&test, 5seeds) on $node submitted as job $jid"
      fi
      submitted=$((submitted+1))
    done
  done
done

echo "==== 제출 요약: submitted=$submitted, skipped=$skipped ===="
