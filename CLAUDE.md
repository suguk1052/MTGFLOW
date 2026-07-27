# CLAUDE.md — DCP / MTGFlow Bearing Anomaly Detection (모델·실험)

> Claude Code가 이 프로젝트에서 작업할 때 참고하는 운영 매뉴얼.
> 공통 지침(응답 스타일, 작업 원칙)은 `../CLAUDE.md`, 데이터셋 구조는 `../Data/Paderborn/CLAUDE.md` 참고.
> **진행 현황·완료 실험·향후 계획은 `../TODO.md` 참조.**
> 상세 코드 구조는 직접 읽어서 파악할 것.

---

## 1. 프로젝트 한 줄 요약

MTGFlow(시계열 이상탐지 모델)로 **운행조건 적응형 window-level 이상탐지**를 연구한다.
핵심 질문: **운행조건 conditioning이 안 본 조건에서 one-class 이상탐지를 개선하는가.**

**메인 실험 무대가 ESTOGU로 이전됨(2026-07 기준).**
- **Paderborn(PU) — 일단락·참조 트랙.** MTGFlow를 PU bearing에 window-level OCC로 적용
  (K=normal, KA/KI/KB=anomaly). 진단 결과, PU는 축당 2조건뿐이라 **조건 보간 시험 불가**로 확정 →
  PU 실험 우선 중지, 진단 결과는 논문 동기로 활용. 아래 §3~§5는 이 PU 트랙의 설계 문서(참조용).
- **ESTOGU — 새 메인.** 전동기 데이터. 부하(payload 프록시)가 6단계·거의 등간격이라 **진짜 조건
  보간이 가능**(예: 000·222·444 학습 → 111·333·555 테스트). 과제 = 비지도 이진 OCC(N=정상만
  학습, {BB,BR,RB3,RB5,SW}=이상), 지표 AUROC. 우선 대상 **Without_Driver**(50 Hz 고정·부하 6단계).

> **진행 현황·Phase 계획·핵심 발견은 `../TODO.md`가 단일 출처.** ESTOGU 데이터 규칙·조건축
> 해석·주의사항은 `../Data/ESTOGU/CLAUDE.md`. 이 문서(§3 이후)는 주로 PU 트랙 운영 매뉴얼이며,
> ESTOGU 로더/실험 코드가 붙는 대로 이 문서에 ESTOGU 절을 확장한다.

---

## 2. 디렉토리 / 경로

- 작업 루트: `/home/dyhwang/DCP` (여기서 Claude Code 실행)
- 데이터:
  - `/home/dyhwang/DCP/Data/ESTOGU` — **현재 메인** (규칙은 그 폴더의 `CLAUDE.md`)
  - `/home/dyhwang/DCP/Data/Paderborn` — PU(참조) (상세는 그 폴더의 CLAUDE.md)
- 모델·실험: `/home/dyhwang/DCP/MTGFLOW`
- **실험 결과 `results/` — 데이터셋별로 묶는다.**
  - `results/Paderborn/` — PU 결과(참조). 완료 실험은 사용자가 수동으로 하위 폴더에 묶어 정리.
  - `results/ESTOGU/` — ESTOGU 결과(예정, 아직 없음).
  - 정리를 도울 땐 어떻게 묶을지 먼저 확인.
- **실행 스크립트 `runners/` — 데이터셋별로 묶는다 (2026-07 정리 완료).**
  - `runners/Paderborn/` — PU 실험 스크립트 전부(하위 배치 폴더 `LONO_*`, `LOSO_*`, `contrast_lossobs`,
    `repro_check`, `unused/` 등). PU 예시 `run_Paderborn_1{,_test}.sh`도 여기.
  - `runners/ESTOGU/` — ESTOGU 실험 스크립트(예정, 아직 없음).
  - `runners/slurm_run.sh`, `runners/slurm_logs/` — **데이터셋 무관 공용**이라 최상위 유지.
    `run_{MSL,PSM,SMD,SWaT,WADI}*.sh`는 원저자/타 데이터셋 예시(참고용).
  - `run_Paderborn_1.sh` 등은 **Codex가 만든 예시일 뿐**, 실제 실험 아님. 진짜 실험 조합은
    `runners/Paderborn/<배치폴더>/` 하위에 있음. 예:
    `runners/Paderborn/LONO_C2_measured/run_Paderborn_CA_012to3_LONO1_m_test.sh`
  - → **몇 개 열어보고 파일명 규칙과 인자 패턴을 대략 파악할 것.**
    (파일명에 setting split, LONO idx, measured 여부 등이 인코딩되어 있음)
  - ⚠️ runner `.sh` 상당수는 **로컬 실험 세팅이라 git 미추적**(의도적). 이미 추적 중인 스크립트만 유지.

### ESTOGU 로더/실험 (신설 예정 — TODO Phase 1)
- 로더: `Dataset/estogu.py`(가칭, 아직 없음) — `paderborn.py` 대응(CSV 로딩, 선택 채널 sliding window,
  normal-only OCC, train-normal에만 scaler fit, 파일경계 window 비겹침, 파일명 prefix→이진 라벨).
- 상세 설계·확정 파라미터(채널=진동 1축, window/stride=2048/1024, fs=34,482.76 Hz)는 `../TODO.md` 참조.

봐야 할 핵심 파일:
- `Dataset/paderborn.py` — loader (.mat 읽기, split 구성, sliding window, metadata 파싱)
- `main.py` — 학습 진입점 (Paderborn CLI args를 loader에 전달)
- `test.py` — 테스트/평가 (학습 때 쓴 split/scaler/checkpoint/threshold와 일관성 유지)
- `runners/Paderborn/` 하위 폴더의 실제 실험 `.sh` 파일들

---

## 3. MTGFlow 구조 ↔ 코드 매핑 (이해용)

- **H**: RNN/LSTM 기반 time encoding
- **A**: self-attention으로 만든 adjacency (동적 그래프)
- **C**: H와 A를 graph convolution한 결과 → flow 입력 condition
- **MAF**: novelty가 아니라 density estimation용 normalizing-flow 구조 (conditional MAF)

---

## 4. 실험 설계 전체 지도 (설계 참조) — **PU 트랙 (일단락/참조)**

### Baseline
- **B1. Single-setting**: 각 setting 하나씩 (`S0_A_vib` … `S3_A_vib`)
- **B2. Pooled multi-setting**: 4개 setting 통합 (`S0123_A_vib`)
- **B3. Leave-one-setting-out**: 3개 학습 / 1개 held-out (`S012_to_S3` 등)

### Context-aware (static metadata) — 파일명 setting을 고정값 조건으로
- **C1 (seen setting)**: `S0123_A_vib_context_setting`, B2와 비교
- **C2 (leave-one-setting-out)**: `S012_to_S3_A_vib_context_numeric`, B3와 비교

### Setting index ↔ operating condition
| idx | setting | 특징 |
| --- | --- | --- |
| 0 | N15_M07_F10 | 기준조건 (1500rpm, 0.7Nm, 1000N) |
| 1 | N09_M07_F10 | 저속 (900rpm) |
| 2 | N15_M01_F10 | 저토크 (0.1Nm) |
| 3 | N15_M07_F04 | 저 radial force (400N) |

### LONO / LOSO split
**LONO (Leave-One-Normal-Out)** — 정상 6개 K bearing 중 train 4 / val 1 / test normal 1, fault test는 결함 전체. 대표 6 split:

| Split | Train | Val | Test normal |
| --- | --- | --- | --- |
| LONO-1 | K003 K004 K005 K006 | K002 | K001 |
| LONO-2 | K001 K004 K005 K006 | K003 | K002 |
| LONO-3 | K001 K002 K005 K006 | K004 | K003 |
| LONO-4 | K001 K002 K003 K006 | K005 | K004 |
| LONO-5 | K001 K002 K003 K004 | K006 | K005 |
| LONO-6 | K002 K003 K004 K005 | K001 | K006 |

> K001/K003/K006 = high amplitude, K002/K004/K005 = low amplitude → threshold 안정성에 영향.

**LOSO (Leave-One-Setting-Out)** — source setting들로 학습, held-out target에서 테스트:

| Split | Train/Val source | Held-out target | 의미 |
| --- | --- | --- | --- |
| LOSO-0 / 123→0 | N09_M07_F10, N15_M01_F10, N15_M07_F04 | N15_M07_F10 | 기준조건 unseen |
| LOSO-1 / 023→1 | N15_M07_F10, N15_M01_F10, N15_M07_F04 | N09_M07_F10 | 저속 unseen |
| LOSO-2 / 013→2 | N15_M07_F10, N09_M07_F10, N15_M07_F04 | N15_M01_F10 | 저토크 unseen |
| LOSO-3 / 012→3 | N15_M07_F10, N09_M07_F10, N15_M01_F10 | N15_M07_F04 | 저 radial force unseen |

> 과거 ABCD split은 결과에 남아있으나 **더 이상 사용 안 함.**

---

## 5. Measured operational condition (설계 참조) — **PU 트랙 (일단락/참조)**

**핵심 아이디어:** 파일명에서 뽑은 고정값 `[rpm, torque, force]` 대신,
**실제 측정된 speed/torque/force 시계열**을 연속 운행 정보로 사용한다.
(measured signal 4kHz, vibration 64kHz — 데이터셋 CLAUDE.md 참고)

**구현 방향: "입력 채널 추가"가 아니라 "동적 condition 추가"**
speed/torque/force는 결함 신호가 아니라 운행 상태를 설명하는 외부 조건이므로,
vibration 같은 anomaly 대상 채널로 넣지 말고 **flow condition으로** 넣는다.

```
vibration window x → 기존 MTGFlow → 기존 condition C
speed/torque/force measured seq → window-level summary 또는 small encoder → C_op
C_total = concat(C, C_op) → conditional MAF
```

### 1차: window-level summary
vibration window(2048 ≈ 0.032s)에 대응하는 measured 구간 ≈ 128 samples를 가져와 요약:
`speed_mean, speed_std, torque_mean, torque_std, force_mean, force_std`
→ `Dataset/paderborn.py`에 mean/meanstd 둘 다 구현되어 있음.

### 2차: measured sequence encoder (요약 방식의 확장안)
```
speed/torque/force segment [128, 3] → 1D CNN / GRU / MLP pooling → op_emb [d_op]
C_total = concat(C, op_emb)
```

### metadata source 종류
| 실험 | metadata source | 형태 | 목적 |
| --- | --- | --- | --- |
| no-meta | 없음 | - | baseline |
| static-meta | 파일명 setting | 고정 3차원 | 기존 CA |
| measured-mean | 실측 speed/torque/force | window mean 3차원 | 실제 운행값 효과 |
| measured-meanstd | 실측 | window mean/std 6차원 | 변동성 정보 효과 |
| measured-seq | 실측 | sequence encoder | 고급 버전 (확장안) |

### Normalize
measured signal은 파일명 metadata처럼 임의 deviation이 아니라 **train set 기준 z-score**:
```
op_norm = (op - train_op_mean) / (train_op_std + 1e-8)
```
⚠️ mean/std는 **train split의 정상 데이터만으로** 계산. test setting을 포함하면
cross-domain 실험에서 정보가 섞임(누수).

---

## 6. 유지해야 할 원칙 (가드레일)

- **데이터 누수 방지**: train/val/test bearing ID 비겹침, scaler는 train normal에만 fit.
  이미 파이프라인에 구현됨 — 새 split에서도 이 원칙 유지.
- **평가**: AUROC만 보지 말고 setting별 FPR, recall, bearing ID별 미탐 패턴도 함께.
  (참고: 과거 KA15, KA22 미탐 이력)

---

## 7. 작업 방식 (프로젝트 고유)

- 실험은 **별도 `.sh` 파일로 관리하고 RUN_NAME을 하드코딩하는 방식을 선호.**
  자동 run-name 생성/setting compacting utility는 불필요. train/test setting 구분만 명확하면 됨.
- **분석/진단·보고서 작성용 스크립트는 `MTGFLOW/analysis/`에 모아서 관리한다.**
  (예: `analysis/diagnose_measured_zscore.py`, `analysis/diagnose_vibration_shift.py`, 공통 유틸은 `analysis/_common.py`)
  학습/평가 코드(`main.py`, `test.py`, `Dataset/`)와 분석 코드를 분리해 관리하기 위함. 앞으로 새 분석/리포트용 스크립트가 필요하면 이 폴더를 활용할 것.
- 만든 코드는 사용자가 논문 의도·실험 설계와 맞는지 직접 검토·병합 (무조건 병합 아님).
- 경로·파일명 규칙 하드코딩 금지 (root path + setting을 받아 구성).
- train/val/test bearing ID 겹침 여부에 assert 또는 warning 권장.
- **파이썬/실험 실행은 반드시 `conda run -n mtgflow <명령>` 접두사 사용** (mtgflow 관련 작업일 때만).
  예: `conda run -n mtgflow python main.py ...`, `conda run -n mtgflow python test.py ...`
  - 이유: Chat의 Bash는 매 명령마다 새 셸을 열어 base가 자동 활성화되고 셸 상태가 유지되지 않음.
    `settings.local.json`의 env PATH 주입 방식은 이 서버 프로필(ohpc lmod + conda init)이 매 셸마다 PATH를 재구성해 **실패**하므로 쓰지 말 것.
  - Bash 출력의 `id: command not found` / `uname: command not found` (ohpc lmod init) 경고는 기존 프로필 특성이라 **무해하니 무시**.
- (공통 원칙: 계획 먼저 승인, 임의 실행 금지, 브랜치 단위 작업 — `../CLAUDE.md` 참고)

### Slurm GPU 실행 (연속 실험 맡기기)

- **로그인 노드(piai-cluster)엔 GPU가 없다.** GPU 학습은 반드시 아래 래퍼로 잡을 제출한다.
  Claude는 대화형 셸(`srun --pty` + `singularity shell`)을 유지 못 하므로,
  `singularity exec --nv ... bash -lc '...'` 비대화형으로 접어 실행한다.
- **노드는 허락된 것만.** 기본은 **n17 V100-16 1장** (`--partition=V100-16
  --gres=gpu:V100-16:1 --nodelist=n17 --cpus-per-task=10`). 다른 노드는 사용자 허락 없이 쓰지 않는다.

**재사용 래퍼:** `runners/slurm_run.sh`
- 사용법(경로는 MTGFLOW 기준 상대, **여러 개 나열하면 순차 체인**):
  ```bash
  bash runners/slurm_run.sh <train1.sh> [<train2.sh> ...]
  DRY_RUN=1 bash runners/slurm_run.sh ...   # 제출 없이 나갈 sbatch 명령만 출력(미리보기)
  ```
- 동작:
  - 각 "학습 스크립트"에 대해 짝 test 스크립트를 파일명 규칙으로 자동 매칭
    (`_5seeds.sh` → `_test_5seeds.sh`).
  - 실험 하나 = **`학습(main.py) && test(test.py)` 한 잡** — 학습 성공해야 test 실행.
  - 실험들 사이 = **`--dependency=afterok` 체인** — 앞 잡이 exit 0이어야 다음 시작,
    중간 실패 시 뒤는 자동 취소.
  - 잡 안에서 `conda activate mtgflow` 후 실행 → 실험 `.sh`의 bare `python3`가 mtgflow env로 해석.
  - 로그: `runners/slurm_logs/<jobname>_<jobid>.out` (gitignore됨).

**배치 묶음:** 실험 조합·순서는 배치별 submit 스크립트로 하드코딩(위 §7 선호 방식).
예: `runners/Paderborn/LONO_B2_5seeds/submit_LONO_B2_5seeds_chain.sh` — 내부에서 `slurm_run.sh`에
순서대로 학습 스크립트를 넘긴다. 새 배치는 이 파일을 복사해 목록만 바꾼다.
(submit 스크립트 안 경로는 MTGFLOW 기준 상대이므로 데이터셋 폴더(`runners/Paderborn/…`) 기준으로 적는다.)

**모니터링/중단:** `squeue -u dyhwang` / `runners/slurm_logs/*.out` (Read) / `scancel <jobid>`.

**주의:** 실험 `.sh`는 **포그라운드**여야 한다(끝에 `&` 금지). 백그라운드면 잡이 먼저 끝나
학습이 죽는다. 실제 제출(GPU 사용)은 **사용자 승인 후** 실행한다(GPU 자원 정책).

### Git 워크플로우
- git 저장소는 `MTGFLOW/`에 있고 Claude Code는 상위 `DCP/`에서 실행된다.
  매 명령마다 새 셸이 열려 `cd`가 유지되지 않으므로, **모든 git 명령은 `cd` 없이
  `git -C MTGFLOW ...` 형태로** 쓴다.
- 브랜치 구조:
  - `main` = 원본 저자 레포. **절대 건드리지 않는다.**
  - `exp/pu` = 개인 연구 통합 브랜치. 모든 작업의 기준점.
  - **코드(.py/.sh) 수정이 포함될 때만** `exp/pu`에서 새 브랜치를 따서 작업하고,
    접두어는 `claude/`. 예: `git -C MTGFLOW checkout -b claude/<작업요약> exp/pu`
  - **`.md` 문서만 수정하는 작업은 새 브랜치를 만들지 말고 현재 브랜치에서 바로** 한다.
- 버전관리 대상:
  - 학습 산출물·대용량 파일(`results/`, `reports/`, `docs/`, `checkpoint/`,
    `*.pth`, `*.log`, `*.mat`, `__pycache__` 등)은 `.gitignore`로 **제외**한다.
  - 코드(`.py`)·실행 스크립트(`.sh`)·문서(`.md`)는 **추적 유지**한다.
- 승인 정책:
  - **커밋**은 의미 있는 단위로 자율적으로 해도 된다.
  - **`exp/pu`로의 merge, GitHub push, 브랜치 삭제**는 하지 말고 먼저 사용자에게 물어본다.
- 금지 사항:
  - `main` 관련 작업 일절 금지.
  - `rebase`, `reset --hard`, force push, `git clean` 등 히스토리·작업물을
    되돌릴 수 없게 날리는 명령 금지.
  - merge 충돌 시 임의 해결 금지 — **멈추고 사용자에게 알린다.**

---

## 8. 세션 시작 권장 순서

1. `../TODO.md`로 현재 트랙(ESTOGU 메인 / PU 참조)·Phase 확인.
2. `MTGFLOW/` 구조 훑기 — `main.py` / `test.py` / `Dataset/paderborn.py`(ESTOGU 로더는 미생성).
3. `runners/Paderborn/` 하위 폴더의 실제 실험 `.sh` 몇 개를 열어 파일명·인자 규칙 파악.
4. 필요 시 `results/Paderborn/C2_measured` 등 최근 결과 형식 확인(PU 참조).
5. 그 다음 요청 작업 시작.