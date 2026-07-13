#!/bin/bash
# ==============================================================================
# 재현성 검증 드라이버 (TODO §0-(1) 검증). Slurm 잡 안에서 conda activate mtgflow
# 이후 실행되는 것을 전제로 한다 (submit_repro_check.sh 참고).
#
# 0단계: 기존 checkpoint를 test.py로 2회 평가해 evaluation 경로 자체의 결정성 확인
# 1단계: 동일 시드(2024)로 2회 독립 재학습 후 test set score 배열/AUROC 비교
# ==============================================================================
set -euo pipefail
cd /home/dayoon/DCP/MTGFLOW
LOGDIR="runners/repro_check"

BASE_ARGS=(--name=paderborn
    --n_blocks 2
    --load_setting N15_M07_F10 N09_M07_F10 N15_M01_F10 N15_M07_F04
    --train_ids K003 K004 K005 K006 --val_ids K002 --test_norm_ids K001
    --window_size 2048 --stride_size 1024 --sampling_rate 64000 --batch_size 256)

echo "=== [0단계] 기존 checkpoint(LONO_B2_5seeds/raw_vib_0123C_LONO1_5seeds_s2024) 재평가 결정성 확인 ==="
python3 test.py "${BASE_ARGS[@]}" --run_name=LONO_B2_5seeds/raw_vib_0123C_LONO1_5seeds --seeds 2024 \
    > "$LOGDIR/eval_run1.log" 2>&1
python3 test.py "${BASE_ARGS[@]}" --run_name=LONO_B2_5seeds/raw_vib_0123C_LONO1_5seeds --seeds 2024 \
    > "$LOGDIR/eval_run2.log" 2>&1
echo "--- eval run1 vs run2 'ROC score' 라인 비교 ---"
diff <(grep "ROC score" "$LOGDIR/eval_run1.log") <(grep "ROC score" "$LOGDIR/eval_run2.log") \
    && echo "[0단계] IDENTICAL" || echo "[0단계] DIFFERS (위 diff 참고)"

echo ""
echo "=== [1단계] 동일 시드(2024) 2회 독립 재학습 ==="
bash runners/repro_check/train_repro_check_A.sh
bash runners/repro_check/train_repro_check_B.sh

echo ""
echo "=== [1단계] test set score 배열 / AUROC 비교 (analysis/check_repro_scores.py) ==="
python3 analysis/check_repro_scores.py \
    --run_a repro_check_LONO1_A_s2024 \
    --run_b repro_check_LONO1_B_s2024
