#!/bin/bash
# ==============================================================================
# 작업 E-3 — order tracking 적용범위 최종 ablation: raw+OT 전용 학습(main.py) 제출.
#
# 대상(신규 학습): non-speed 3 split × LONO1~6 × seed 2024~2028 = 90 job.
#   - 123→0(compositional) / 013→2(torque) / 012→3(force). target=1500rpm이라 test OT=identity,
#     raw+OT는 train의 900rpm source를 order 영역으로 재표현 → --order_track 켠 전용 모델이 필요.
#   - 023→1(speed)은 train OT가 identity라 신규 학습 없음(raw ckpt에 test-time OT 재추론; eval에서 처리).
#   - raw arm은 기존 B3 5-seed 체크포인트 재추론(학습 없음).
#
# 실행: n16+n17 V100-16 각 3 GPU = 최대 6 동시. 90 job을 n17/n16 교대 pin, dependency 없이 독립 제출
#   → Slurm 큐잉으로 노드당 3개씩 병렬. 전체 학습 성공(afterok) 후 eval 1잡 자동 제출.
#
# 사용:  bash runners/Paderborn/E_order_tracking/submit_E3_rawOT_train.sh
#   미리보기:  DRY_RUN=1 bash runners/Paderborn/E_order_tracking/submit_E3_rawOT_train.sh
# ==============================================================================
set -euo pipefail
ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"
mkdir -p "$LOGDIR"

SEEDS=(2024 2025 2026 2027 2028)
LONOS=(LONO1 LONO2 LONO3 LONO4 LONO5 LONO6)
SPLITS=(123to0 013to2 012to3)   # 023to1 제외(train OT identity → 재추론)
NODES=(n17 n16)

# split별 load_setting (B3 metadata 확증)
declare -A TRAIN_SET TEST_SET
TRAIN_SET[123to0]="N09_M07_F10 N15_M01_F10 N15_M07_F04"; TEST_SET[123to0]="N15_M07_F10"
TRAIN_SET[013to2]="N15_M07_F10 N09_M07_F10 N15_M07_F04"; TEST_SET[013to2]="N15_M01_F10"
TRAIN_SET[012to3]="N15_M07_F10 N09_M07_F10 N15_M01_F10"; TEST_SET[012to3]="N15_M07_F04"

# LONO별 bearing ID (4 split 공통, B3 metadata 확증)
declare -A TRAIN_IDS VAL_IDS TEST_IDS
TRAIN_IDS[LONO1]="K003 K004 K005 K006"; VAL_IDS[LONO1]="K002"; TEST_IDS[LONO1]="K001"
TRAIN_IDS[LONO2]="K001 K004 K005 K006"; VAL_IDS[LONO2]="K003"; TEST_IDS[LONO2]="K002"
TRAIN_IDS[LONO3]="K001 K002 K005 K006"; VAL_IDS[LONO3]="K004"; TEST_IDS[LONO3]="K003"
TRAIN_IDS[LONO4]="K001 K002 K003 K006"; VAL_IDS[LONO4]="K005"; TEST_IDS[LONO4]="K004"
TRAIN_IDS[LONO5]="K001 K002 K003 K004"; VAL_IDS[LONO5]="K006"; TEST_IDS[LONO5]="K005"
TRAIN_IDS[LONO6]="K002 K003 K004 K005"; VAL_IDS[LONO6]="K001"; TEST_IDS[LONO6]="K006"

JIDS=()
i=0
for split in "${SPLITS[@]}"; do
  for fold in "${LONOS[@]}"; do
    for s in "${SEEDS[@]}"; do
      node="${NODES[$(( i % 2 ))]}"   # n17/n16 교대 → 각 노드에 3개씩 병렬 실행
      i=$((i+1))
      run_name="E3_rawOT_${split}_${fold}"
      train_cmd="CUDA_VISIBLE_DEVICES=0 python3 main.py \
--name=paderborn --run_name=${run_name} --n_blocks=2 --batch_size=256 \
--window_size=2048 --stride_size=1024 --sampling_rate=64000 \
--train_load_setting ${TRAIN_SET[$split]} --test_load_setting ${TEST_SET[$split]} \
--train_ids ${TRAIN_IDS[$fold]} --val_ids ${VAL_IDS[$fold]} --test_norm_ids ${TEST_IDS[$fold]} \
--order_track --order_track_ref nominal --seeds ${s}"
      inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && ${train_cmd}"
      wrap="singularity exec --nv $SIF bash -lc \"$inner\""
      cmd=(sbatch --partition=V100-16 --gres=gpu:V100-16:1 --nodelist="$node" --cpus-per-task=10 \
           -J "E3_${split}_${fold}_s${s}" -o "$LOGDIR/%x_%j.out" --wrap "$wrap")
      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[$node] E3_${split}_${fold}_s${s}"; printf '%q ' "${cmd[@]}"; echo; echo
        JIDS+=("JID_${split}_${fold}_s${s}")
      else
        out="$("${cmd[@]}")"; echo "[$node] E3_${split}_${fold}_s${s}: $out"
        jid="$(echo "$out" | grep -oE '[0-9]+' | tail -1)"
        JIDS+=("$jid")
      fi
    done
  done
done

echo
echo "총 학습 제출: ${#JIDS[@]} job (기대 90)"

# ---- 전체 학습 afterok 후 eval 1잡 ----
dep="$(IFS=:; echo "${JIDS[*]}")"
eval_inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && python3 analysis/diagnose_E3_scope_eval.py"
eval_wrap="singularity exec --nv $SIF bash -lc \"$eval_inner\""
eval_cmd=(sbatch --partition=V100-16 --gres=gpu:V100-16:1 --nodelist="n17" --cpus-per-task=10 \
          --dependency=afterok:"$dep" -J "E3_eval" -o "$LOGDIR/%x_%j.out" --wrap "$eval_wrap")
echo
echo "=== eval 잡 (afterok: 위 ${#JIDS[@]} train 전부) ==="
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${eval_cmd[@]}"; echo
else
  out="$("${eval_cmd[@]}")"; echo "E3_eval: $out"
fi
