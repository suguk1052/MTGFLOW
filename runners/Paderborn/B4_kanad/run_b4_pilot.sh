#!/bin/bash
# B-4 KAN-AD 파일럿 (1 fold × 1 seed, GPU 1장). 별도 어댑터(TSLib 래퍼 미사용).
#   확인: fwd+bwd+opt GPU 메모리 · score 덤프(b3 스키마) · val FPR≈0.05. 파일럿 AUROC로 하이퍼 변경 금지.
set -euo pipefail
cd "$HOME/DCP/MTGFLOW"
python3 baselines/kanad.py --split 012to3 --lono 1 --seed 2024 --overwrite
