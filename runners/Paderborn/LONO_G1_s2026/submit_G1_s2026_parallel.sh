#!/bin/bash
# ==============================================================================
# 작업 G-1(dual-branch shape/amplitude disentangle) 병렬 제출 — seed 2026, 24 fold 게이트.
#  - 24 fold 각각을 독립 잡(dependency 없음)으로 제출 → n16·n17 각 3 GPU에 packing(6 동시).
#  - node round-robin(n17,n16,n17,…). 각 잡 = (학습(main.py --amp_normalize --amp_branch) && test).
#  - 이미 완료된 fold(paderborn_per_bearing_metrics.json 존재)는 자동 스킵.
#  실제 제출: bash runners/Paderborn/LONO_G1_s2026/submit_G1_s2026_parallel.sh
#  미리보기:  DRY_RUN=1 bash runners/Paderborn/LONO_G1_s2026/submit_G1_s2026_parallel.sh
#  게이트 진단(전체 완료 후): conda run -n mtgflow python analysis/diagnose_G1_disentangle.py
# ==============================================================================
set -euo pipefail

ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"
DIR="runners/Paderborn/LONO_G1_s2026"
RESULTS="$ROOT/results/Paderborn"

# 사용 노드(사용자 승인: n16 + n17). round-robin.
NODES=(n17 n16)

# submit 순서(D와 동일: split 4 × LONO 6 = 24)
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

    # 이미 완료된 fold 스킵(재추론 낭비 방지)
    if [[ -f "$RESULTS/${run_name}_s2026/paderborn_per_bearing_metrics.json" ]]; then
      echo "[skip] $run_name (metrics 이미 존재)"
      skipped=$((skipped+1))
      continue
    fi

    node="${NODES[$((i % ${#NODES[@]}))]}"
    i=$((i+1))

    res=(--partition=V100-16 --gres=gpu:V100-16:1 --nodelist="$node" --cpus-per-task=10)
    inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && bash '$train' && bash '$test_rel'"
    wrap="singularity exec --nv $SIF bash -lc \"$inner\""
    cmd=(sbatch "${res[@]}" -J "run_Paderborn_${run_name}" -o "$LOGDIR/%x_%j.out" --wrap "$wrap")

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "[$node] $run_name:"; printf '  %q ' "${cmd[@]}"; echo
    else
      out="$("${cmd[@]}")"; jid="$(echo "$out" | grep -oE '[0-9]+' | tail -1)"
      echo "  -> $run_name (train&&test) on $node submitted as job $jid"
    fi
    submitted=$((submitted+1))
  done
done

echo ""
echo "제출: $submitted, 스킵(완료): $skipped"
