#!/bin/bash
# ==============================================================================
# B-2 Deep SVDD 전량 — array task 1개 = (fold, seed) 1개 처리 (GPU 1장).
#   사용: b2_dispatch.sh <index 0..119>
#   index → fold=idx//5, seed=2024+idx%5, split=SPLITS[fold//6], lono=fold%6+1
#   (24 fold × 5 seed = 120). 파일럿 fold(012to3_LONO1_s2024)는 deep_svdd.py가 스킵.
# 컨테이너(singularity --nv) 안에서 conda mtgflow 활성 상태로 호출된다.
# ==============================================================================
set -euo pipefail
IDX="$1"
ROOT="$HOME/DCP/MTGFLOW"
cd "$ROOT"

SPLITS=(123to0 023to1 013to2 012to3)
fold=$(( IDX / 5 )); seed=$(( 2024 + IDX % 5 ))
split="${SPLITS[$(( fold / 6 ))]}"; lono=$(( fold % 6 + 1 ))
echo "[b2_dispatch] index=$IDX -> split=$split lono=$lono seed=$seed"
python3 baselines/deep_svdd.py --split "$split" --lono "$lono" --seed "$seed"
