#!/bin/bash
# ==============================================================================
# Slurm 제출 래퍼 (V100-16 1장; 노드 기본 n17, NODELIST 환경변수로 n16 등 오버라이드)
#  - 인자로 받은 각 "학습 스크립트"에 대해 (학습 && test)를 한 잡으로 제출
#  - test 스크립트는 파일명 규칙으로 자동 매칭: _5seeds.sh -> _test_5seeds.sh
#  - 여러 개면 --dependency=afterok 로 순차 체인 (앞이 성공해야 다음 시작)
#  - 비대화형: singularity exec --nv ... bash -lc '... conda activate mtgflow ...'
#  - 허락된 노드만 사용(기본 n17, NODELIST로 n16 지정 가능)
#
# 사용법:
#   bash runners/slurm_run.sh <train1.sh> [<train2.sh> ...]   # (경로는 MTGFLOW 기준 상대)
#   DRY_RUN=1 bash runners/slurm_run.sh ...   # 제출 안 하고 sbatch 명령만 출력
# ==============================================================================
set -euo pipefail

# 경로는 실행 사용자 HOME 기준(제출 시 로그인 노드에서 확장). 과거 /home/dayoon 하드코딩은
# 계정 이관 후 stale였음 → $HOME 기반으로 교체(현재 dyhwang, 공유 홈이면 이관에도 견고).
ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"

# V100-16 1장. 노드는 기본 n17, NODELIST 환경변수로 오버라이드(허락된 n16/n17만).
#   예: NODELIST=n16 bash runners/slurm_run.sh ...
NODELIST="${NODELIST:-n17}"
SLURM_RES=(--partition=V100-16 --gres=gpu:V100-16:1 --nodelist="$NODELIST" --cpus-per-task=10)

mkdir -p "$LOGDIR"

# 학습 스크립트명 -> test 스크립트명
to_test() { echo "$1" | sed 's/_5seeds\.sh$/_test_5seeds.sh/'; }

prev_jid=""
for train in "$@"; do
  test_rel="$(to_test "$train")"
  [[ -f "$ROOT/$train"    ]] || { echo "학습 스크립트 없음: $ROOT/$train" >&2; exit 1; }
  [[ -f "$ROOT/$test_rel" ]] || { echo "test 스크립트 없음: $ROOT/$test_rel" >&2; exit 1; }

  job_name="$(basename "$train" _5seeds.sh)"   # 예: run_Paderborn_0123_LONO3

  inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && bash '$train' && bash '$test_rel'"
  wrap="singularity exec --nv $SIF bash -lc \"$inner\""

  # PARALLEL=1이면 afterok 체인을 생략해 각 잡을 독립 제출(병렬). 기본은 직렬 체인(하위호환).
  dep=()
  [[ "${PARALLEL:-0}" != "1" && -n "$prev_jid" ]] && dep=(--dependency=afterok:"$prev_jid")

  cmd=(sbatch "${SLURM_RES[@]}" "${dep[@]}" -J "$job_name" -o "$LOGDIR/%x_%j.out" --wrap "$wrap")

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"; echo
    prev_jid="JID_${job_name}"
  else
    out="$("${cmd[@]}")"; echo "$out"
    prev_jid="$(echo "$out" | grep -oE '[0-9]+' | tail -1)"
    echo "  -> $job_name (train&&test) submitted as job $prev_jid"
  fi
done
