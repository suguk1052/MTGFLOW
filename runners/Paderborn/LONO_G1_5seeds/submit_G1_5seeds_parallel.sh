#!/bin/bash
# ==============================================================================
# 작업 G-1(dual-branch disentangle) 5-seed(2024–2028) 병렬 제출 — 24 fold × 5 seed.
#  ⚠️ 1-seed(s2026) 게이트가 가능성을 보인 뒤에만 실행(사용자 승인 필수, GPU 자원).
#  - 각 잡 = 한 fold의 (학습(main.py, --seeds 2024..2028) && test) — 잡 내부에서 5 seed를 순차 학습.
#  - node round-robin(n17,n16). 이미 완료된 fold(summary_seeds.json 존재)는 자동 스킵.
#  실제 제출: bash runners/Paderborn/LONO_G1_5seeds/submit_G1_5seeds_parallel.sh
#  미리보기:  DRY_RUN=1 bash runners/Paderborn/LONO_G1_5seeds/submit_G1_5seeds_parallel.sh
# ==============================================================================
set -euo pipefail

ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"
DIR="runners/Paderborn/LONO_G1_5seeds"
RESULTS="$ROOT/results/Paderborn"

NODES=(n17 n16)
SPLITS=(023to1 012to3 013to2 123to0)
LONOS=(1 2 3 4 5 6)

mkdir -p "$LOGDIR"

i=0
submitted=0
skipped=0
for split in "${SPLITS[@]}"; do
  for lono in "${LONOS[@]}"; do
    run_name="g1_${split}_LONO${lono}"
    train="$DIR/run_Paderborn_g1_${split}_LONO${lono}_5seeds.sh"
    test_rel="$DIR/run_Paderborn_g1_${split}_LONO${lono}_test_5seeds.sh"
    [[ -f "$ROOT/$train"    ]] || { echo "학습 스크립트 없음: $ROOT/$train" >&2; exit 1; }
    [[ -f "$ROOT/$test_rel" ]] || { echo "test 스크립트 없음: $ROOT/$test_rel" >&2; exit 1; }

    # 5 seed 전체 완료(summary_seeds.json) 시 스킵
    if [[ -f "$RESULTS/${run_name}/summary_seeds.json" ]]; then
      echo "[skip] $run_name (summary_seeds.json 이미 존재)"
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
      echo "[$node] $run_name:"; printf '  %q ' "${cmd[@]}"; echo
    else
      out="$("${cmd[@]}")"; jid="$(echo "$out" | grep -oE '[0-9]+' | tail -1)"
      echo "  -> $run_name (train&&test, 5seeds) on $node submitted as job $jid"
    fi
    submitted=$((submitted+1))
  done
done

echo ""
echo "제출: $submitted, 스킵(완료): $skipped"
