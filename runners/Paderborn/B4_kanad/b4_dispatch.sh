#!/bin/bash
# B-4 KAN-AD 전량 — array task 1개 = (fold, seed) 1개 (GPU 1장).
#   index 0..119 → fold=idx//5, seed=2024+idx%5, split=SPLITS[fold//6], lono=fold%6+1.
#   파일럿 fold(012to3_LONO1_s2024)는 kanad.py가 스킵.
set -euo pipefail
IDX="$1"; ROOT="$HOME/DCP/MTGFLOW"; cd "$ROOT"
SPLITS=(123to0 023to1 013to2 012to3)
fold=$(( IDX / 5 )); seed=$(( 2024 + IDX % 5 ))
split="${SPLITS[$(( fold / 6 ))]}"; lono=$(( fold % 6 + 1 ))
echo "[b4_dispatch] index=$IDX -> split=$split lono=$lono seed=$seed"
python3 baselines/kanad.py --split "$split" --lono "$lono" --seed "$seed"
