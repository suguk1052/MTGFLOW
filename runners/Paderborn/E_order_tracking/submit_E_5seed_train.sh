#!/bin/bash
# ==============================================================================
# 작업 E-1 Phase B 5-seed 확장 — 023→1 LONO1/LONO2 나머지 4 seed 학습(main.py만).
#   seed2026은 이미 완료. 여기서 2024·2025·2027·2028 × {LONO1,LONO2} = 8잡을
#   n17·n16(각 3 GPU)에 교대로 독립 제출(병렬). 평가는 학습 후 diagnose_E_order_tracking_eval.py.
# 사용: bash runners/Paderborn/E_order_tracking/submit_E_5seed_train.sh
#       DRY_RUN=1 ... 미리보기.
# ==============================================================================
set -euo pipefail
ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"
mkdir -p "$LOGDIR"

SEEDS=(2024 2025 2027 2028)
NODES=(n17 n16)

# fold별 split (기존 B3와 동일)
declare -A TRAIN_IDS VAL_IDS TEST_IDS
TRAIN_IDS[LONO1]="K003 K004 K005 K006"; VAL_IDS[LONO1]="K002"; TEST_IDS[LONO1]="K001"
TRAIN_IDS[LONO2]="K001 K004 K005 K006"; VAL_IDS[LONO2]="K003"; TEST_IDS[LONO2]="K002"

i=0
for fold in LONO1 LONO2; do
  for s in "${SEEDS[@]}"; do
    node="${NODES[$(( (i/1) % 2 ))]}"   # n17,n16 교대 → 각 노드에 4잡(3 실행+1 대기)
    i=$((i+1))
    run_name="E_raw_023to1_${fold}"
    train_cmd="CUDA_VISIBLE_DEVICES=0 python3 main.py \
--name=paderborn --run_name=${run_name} --n_blocks=2 --batch_size=256 \
--window_size=2048 --stride_size=1024 --sampling_rate=64000 \
--train_load_setting N15_M07_F10 N15_M01_F10 N15_M07_F04 --test_load_setting N09_M07_F10 \
--train_ids ${TRAIN_IDS[$fold]} --val_ids ${VAL_IDS[$fold]} --test_norm_ids ${TEST_IDS[$fold]} \
--seeds ${s}"
    inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && ${train_cmd}"
    wrap="singularity exec --nv $SIF bash -lc \"$inner\""
    cmd=(sbatch --partition=V100-16 --gres=gpu:V100-16:1 --nodelist="$node" --cpus-per-task=10 \
         -J "E_${fold}_s${s}" -o "$LOGDIR/%x_%j.out" --wrap "$wrap")
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "[$node] E_${fold}_s${s}"; printf '%q ' "${cmd[@]}"; echo; echo
    else
      out="$("${cmd[@]}")"; echo "[$node] E_${fold}_s${s}: $out"
    fi
  done
done
