#!/bin/bash
# ==============================================================================
# UODS 전량(eval 100 split × {proposed,raw} × 5 seed = 200 잡) 제출 orchestrator.
# 짝수 split→n17, 홀수 split→n16(각 100잡). 각 잡 = train(5 seed) && dump(5 seed).
# 노드당 3 GPU=3 동시. gen_eval_runners.py를 먼저 실행해 러너·노드 리스트가 있어야 함.
# 사용: bash runners/UODS/eval/submit_uods_eval.sh        (DRY_RUN=1 이면 미리보기)
# ==============================================================================
set -euo pipefail
ROOT="$HOME/DCP/MTGFLOW"
N17_LIST="$ROOT/runners/UODS/eval/n17_train_scripts.txt"
N16_LIST="$ROOT/runners/UODS/eval/n16_train_scripts.txt"
[[ -f "$N17_LIST" && -f "$N16_LIST" ]] || { echo "노드 리스트 없음 — gen_eval_runners.py 먼저 실행" >&2; exit 1; }

mapfile -t N17 < "$N17_LIST"
mapfile -t N16 < "$N16_LIST"
echo "n17: ${#N17[@]} 잡, n16: ${#N16[@]} 잡 (총 $(( ${#N17[@]} + ${#N16[@]} )))"

echo "== n17 제출 =="
DRY_RUN="${DRY_RUN:-0}" NODELIST=n17 PARALLEL=1 bash "$ROOT/runners/slurm_run.sh" "${N17[@]}"
echo "== n16 제출 =="
DRY_RUN="${DRY_RUN:-0}" NODELIST=n16 PARALLEL=1 bash "$ROOT/runners/slurm_run.sh" "${N16[@]}"
echo "제출 완료. squeue -u dyhwang 로 확인."
