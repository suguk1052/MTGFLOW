#!/bin/bash
# B-2 Deep SVDD 파일럿 (1 fold × 1 seed, GPU 1장). 실행 가능성 확인용:
#   fwd+bwd+opt GPU 메모리 · collapse 3지표 · per-window npz 덤프. 파일럿 AUROC로 하이퍼 변경 금지.
# 컨테이너(singularity --nv) 안에서 conda mtgflow 활성 상태로 호출된다.
set -euo pipefail
cd "$HOME/DCP/MTGFLOW"
python3 baselines/deep_svdd.py --split 012to3 --lono 1 --seed 2024 --overwrite
