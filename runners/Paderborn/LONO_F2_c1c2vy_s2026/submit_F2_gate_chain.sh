#!/bin/bash
# ==============================================================================
# 작업 F-2 gate — C1C2Vy 다채널(3노드) 대표 8 fold × 2 구성 = 16 학습잡, n17 V100-16 afterok 체인.
#   fold = 4 LOSO split × {LONO1(고진폭 K001), LONO2(저진폭 K002)}.
#   구성 = all(전 채널 rmsnorm) / mixed(Vy만 rmsnorm, 전류 raw).
#   각 실험 = (학습(main.py) && test(test.py, --rms_lambda 0)) 한 잡. slurm_run.sh가 _5seeds→_test_5seeds 짝매칭.
#   순서: split별로 두 LONO·두 구성을 인접 배치 → 앞 fold부터 both-config 비교가 일찍 나온다.
#   전류·조건 confound 정곡인 023to1(저속)을 맨 앞에 둔다.
#
#   실제 제출: bash runners/Paderborn/LONO_F2_c1c2vy_s2026/submit_F2_gate_chain.sh
#   미리보기:  DRY_RUN=1 bash runners/Paderborn/LONO_F2_c1c2vy_s2026/submit_F2_gate_chain.sh
#   gate 진단(체인 완료 후): bash runners/Paderborn/LONO_F2_c1c2vy_s2026/submit_F2_eval.sh
# ==============================================================================
set -euo pipefail
DIR="runners/Paderborn/LONO_F2_c1c2vy_s2026"

SPLITS=(023to1 013to2 123to0 012to3)
LONOS=(1 2)
CONFIGS=(all mixed)

scripts=()
for split in "${SPLITS[@]}"; do
  for lono in "${LONOS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
      scripts+=("$DIR/run_Paderborn_f2_${cfg}_${split}_LONO${lono}_5seeds.sh")
    done
  done
done

echo "F-2 gate: ${#scripts[@]} 학습잡(train&&test) afterok 체인 제출 → n17"
bash runners/slurm_run.sh "${scripts[@]}"
