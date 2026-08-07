#!/bin/bash
# ==============================================================================
# 작업 F-2 gate 병렬 제출 — 직렬 afterok 체인 대체.
#   16개 (config,split,lono) = 8 fold × {all, mixed}. 각 fold/config는 독립이므로
#   train job과 test job을 별도 sbatch로 제출하고, 서로 다른 fold끼리는 의존성을 걸지 않는다.
#     - train job : GPU 1장(--gres=gpu:V100-16:1), n17/n16 round-robin.
#     - test  job : 동일 자원 + --dependency=afterok:<자기 train jid> 만.  (다른 fold 무관)
#   → fold 간 의존성 0 → Slurm이 n17·n16의 V100 6장에 최대 6 job 동시 실행.
#   GPU 개별 할당이라 동일 GPU에 두 job이 겹치지 않음(D 병렬 스크립트에서 검증된 패턴).
#   이미 완료된 fold(paderborn_per_bearing_metrics.json 존재)는 train·test 모두 skip.
#   모든 test 성공 후 F-2 진단(diagnose_F2_evaluation.py --lonos 1 2)을 afterok로 auto-chain
#   → reports/report_F2_evaluation.md 자동 생성 (WITH_DIAG=0으로 끌 수 있음).
#
#   실제 제출: bash runners/Paderborn/LONO_F2_c1c2vy_s2026/submit_F2_gate_parallel.sh
#   미리보기:  DRY_RUN=1 bash runners/Paderborn/LONO_F2_c1c2vy_s2026/submit_F2_gate_parallel.sh
#   n17 단독:  NODES="n17" bash ...
# ==============================================================================
set -euo pipefail

ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"
DIR="runners/Paderborn/LONO_F2_c1c2vy_s2026"
RESULTS="$ROOT/results/Paderborn"

# 사용 노드(사용자 승인 2026-08-07: n17 + n16, 각 V100 3장). round-robin. env로 축소 가능.
read -ra NODES <<< "${NODES:-n17 n16}"

CONFIGS=(all mixed)
SPLITS=(023to1 013to2 123to0 012to3)
LONOS=(1 2)

mkdir -p "$LOGDIR"

submit_or_echo() {  # stdout으로는 jid만 반환($(...) 캡처용). DRY_RUN 미리보기는 stderr로.
  local label="$1"; shift
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    { printf '  [%s] ' "$label"; printf '%q ' "$@"; echo; } >&2
    echo "JID_${label}"
  else
    local out; out="$("$@")"; echo "${out%%;*}"
  fi
}

i=0
submitted=0
skipped=0
test_jids=()
for cfg in "${CONFIGS[@]}"; do
  for split in "${SPLITS[@]}"; do
    for lono in "${LONOS[@]}"; do
      run_name="f2_${cfg}_${split}_LONO${lono}"
      train="$DIR/run_Paderborn_f2_${cfg}_${split}_LONO${lono}_5seeds.sh"
      test_rel="$DIR/run_Paderborn_f2_${cfg}_${split}_LONO${lono}_test_5seeds.sh"
      [[ -f "$ROOT/$train"    ]] || { echo "학습 스크립트 없음: $ROOT/$train" >&2; exit 1; }
      [[ -f "$ROOT/$test_rel" ]] || { echo "test 스크립트 없음: $ROOT/$test_rel" >&2; exit 1; }

      if [[ -f "$RESULTS/${run_name}_s2026/paderborn_per_bearing_metrics.json" ]]; then
        echo "[skip] $run_name (metrics 이미 존재)"; skipped=$((skipped+1)); continue
      fi

      node="${NODES[$((i % ${#NODES[@]}))]}"; i=$((i+1))
      res=(--partition=V100-16 --gres=gpu:V100-16:1 --nodes=1 --nodelist="$node" --cpus-per-task=10)

      inner_tr="source $CONDA_SH && conda activate mtgflow && cd $ROOT && bash '$train'"
      cmd_tr=(sbatch "${res[@]}" -J "f2tr_${run_name}" -o "$LOGDIR/%x_%j.out" --parsable \
              --wrap "singularity exec --nv $SIF bash -lc \"$inner_tr\"")
      tr_jid="$(submit_or_echo "train_${run_name}@${node}" "${cmd_tr[@]}")"

      inner_te="source $CONDA_SH && conda activate mtgflow && cd $ROOT && bash '$test_rel'"
      cmd_te=(sbatch "${res[@]}" --dependency=afterok:"$tr_jid" -J "f2te_${run_name}" -o "$LOGDIR/%x_%j.out" --parsable \
              --wrap "singularity exec --nv $SIF bash -lc \"$inner_te\"")
      te_jid="$(submit_or_echo "test_${run_name}@${node}(afterok:$tr_jid)" "${cmd_te[@]}")"

      [[ "${DRY_RUN:-0}" == "1" ]] || echo "  -> $run_name  train=$tr_jid  test=$te_jid  node=$node"
      test_jids+=("$te_jid")
      submitted=$((submitted+1))
    done
  done
done

# 진단 auto-chain: 모든 test 성공 시 F-2 진단 실행 → report_F2_evaluation.md
if [[ "${WITH_DIAG:-1}" == "1" && ${#test_jids[@]} -gt 0 ]]; then
  dep="$(IFS=:; echo "${test_jids[*]}")"
  node="${NODES[0]}"
  res=(--partition=V100-16 --gres=gpu:V100-16:1 --nodes=1 --nodelist="$node" --cpus-per-task=10)
  inner_dg="source $CONDA_SH && conda activate mtgflow && cd $ROOT && python -u analysis/diagnose_F2_evaluation.py --lonos 1 2"
  cmd_dg=(sbatch "${res[@]}" --dependency=afterok:"$dep" -J F2_eval -o "$LOGDIR/%x_%j.out" --parsable \
          --wrap "singularity exec --nv $SIF bash -lc \"$inner_dg\"")
  dg_jid="$(submit_or_echo "diag(afterok:${#test_jids[@]}tests)" "${cmd_dg[@]}")"
  [[ "${DRY_RUN:-0}" == "1" ]] || echo "  -> F2_eval(diag)=$dg_jid  afterok on ${#test_jids[@]} test jobs  node=$node"
fi

echo ""
echo "제출 fold: $submitted (train $submitted + test $submitted job), 스킵(완료): $skipped, 노드: ${NODES[*]}"
