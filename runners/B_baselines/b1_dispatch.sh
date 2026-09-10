#!/bin/bash
# ==============================================================================
# B-1 고전 baseline (IF·OC-SVM) — array task 1개 = fold/split 1개 처리 (CPU).
#   사용: b1_dispatch.sh <kind: pu|uods> <index>
#   PU  : index 0..23  -> split(4) × lono(6). dump_B1_pu_window_scores.py --split --lono
#   UODS: index 0..99  -> run_idx = 5 + index.  dump_B1_uods_window_scores.py --run_idx
# 컨테이너 안(singularity)에서 conda mtgflow 활성 상태로 호출된다(slurm_run_cpu.sh가 감쌈).
# ==============================================================================
set -euo pipefail
KIND="$1"; IDX="$2"
ROOT="$HOME/DCP/MTGFLOW"
cd "$ROOT"

# cpus-per-task=4 정합: BLAS/joblib 스레드 상한(오버구독 방지). IF는 n_jobs=4로 별도 병렬.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ "$KIND" == "pu" ]]; then
  SPLITS=(123to0 023to1 013to2 012to3)   # dump 스크립트 choices와 무관, 매핑 고정
  s=$(( IDX / 6 )); l=$(( IDX % 6 + 1 ))
  SPLIT="${SPLITS[$s]}"
  echo "[b1_dispatch] PU index=$IDX -> split=$SPLIT lono=$l"
  python3 analysis/dump_B1_pu_window_scores.py --split "$SPLIT" --lono "$l"
elif [[ "$KIND" == "uods" ]]; then
  RUN=$(( 5 + IDX ))
  echo "[b1_dispatch] UODS index=$IDX -> run_idx=$RUN"
  python3 analysis/dump_B1_uods_window_scores.py --run_idx "$RUN"
else
  echo "unknown kind: $KIND (pu|uods)" >&2; exit 1
fi
