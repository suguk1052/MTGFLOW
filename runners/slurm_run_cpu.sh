#!/bin/bash
# ==============================================================================
# Slurm CPU 제출 래퍼 (B-1 고전 baseline 전용) — slurm_run.sh의 CPU 변형.
#  - 클러스터에 CPU 전용 파티션이 없어 GPU 파티션(V100-16, n16/n17) 노드에 --gres 없이 CPU 잡 제출.
#  - --cpus-per-task=4 (IF n_jobs=4와 정합). 노드당 동시 실행은 %CONC(기본 4잡=16코어 ≤ 절반 19코어)로 제한.
#  - 124잡을 두 노드(n17=앞 절반, n16=뒤 절반)로 나눠 array 2개 제출(노드당 코어 절반 이하 보장).
#  - 비대화형 singularity exec(--nv 없음: GPU 불필요) 안에서 conda activate mtgflow.
#
# 사용:
#   bash runners/slurm_run_cpu.sh pu     # PU 24 fold  (n17: idx0-11, n16: idx12-23)
#   bash runners/slurm_run_cpu.sh uods   # UODS 100 split (n17: idx0-49, n16: idx50-99)
#   DRY_RUN=1 bash runners/slurm_run_cpu.sh pu   # 제출 없이 sbatch 명령만 출력
#   CONC=4 MEM=16G CPUS=4 bash runners/slurm_run_cpu.sh pu
# ==============================================================================
set -euo pipefail
KIND="${1:?사용: slurm_run_cpu.sh <pu|uods>}"

ROOT="$HOME/DCP/MTGFLOW"
SIF="$HOME/containers/anomaly-base_cu118-u20.sif"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
LOGDIR="$ROOT/runners/slurm_logs"
DISPATCH="runners/B_baselines/b1_dispatch.sh"
mkdir -p "$LOGDIR"

CPUS="${CPUS:-4}"          # --cpus-per-task (IF n_jobs=4 정합)
CONC="${CONC:-4}"          # 노드당 동시 array task 상한 (4×4=16코어 ≤ 39/2)
MEM="${MEM:-20G}"          # 잡당 메모리(Slurm 파일럿 MaxRSS 12.5GB[IF n_jobs=4 워커 복제] → 20G 여유)

case "$KIND" in
  pu)   NTASK=24;  MIDN17=11;  START_N16=12;  END=23;;
  uods) NTASK=100; MIDN17=49;  START_N16=50;  END=99;;
  *) echo "unknown kind: $KIND (pu|uods)" >&2; exit 1;;
esac

# AFTER="<jobid>[:<jobid>...]" 이면 각 array를 그 잡들 전량 종료 후 시작(afterany 체인).
DEP=()
[[ -n "${AFTER:-}" ]] && DEP=(--dependency=afterany:"$AFTER")

submit() {  # $1=array_range  $2=node  → 성공 시 stdout에 job id만 출력
  local arr="$1" node="$2"
  local inner="source $CONDA_SH && conda activate mtgflow && cd $ROOT && bash $DISPATCH $KIND \$SLURM_ARRAY_TASK_ID"
  local wrap="singularity exec $SIF bash -lc \"$inner\""
  local cmd=(sbatch "${DEP[@]}" --partition=V100-16 --nodelist="$node" --cpus-per-task="$CPUS"
             --mem="$MEM" --array="$arr" -J "b1_${KIND}_${node}"
             -o "$LOGDIR/%x_%A_%a.out" --parsable --wrap "$wrap")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"   # --parsable → job id만 출력
  fi
}

echo "B-1 CPU 제출: kind=$KIND, tasks=$NTASK, cpus=$CPUS, conc=$CONC/node, mem=$MEM, after=${AFTER:-none}" >&2
jid1=$(submit "0-${MIDN17}%${CONC}"        n17)
jid2=$(submit "${START_N16}-${END}%${CONC}" n16)
echo "$jid1"; echo "$jid2"   # stdout = 제출된 두 array job id (호출부가 AFTER 체인에 사용)
