# B-1 고전 baseline (IF · OC-SVM) — 제출 상태 (진행 중)

> 브랜치 `claude/b-external-baselines`. CPU 전용(GPU 미사용). **집계·리포트·TODO 갱신은 124잡 전량 완료 후 별도 세션.**
> 이 문서는 제출 시점 기록(재현·인수인계용).

## 제출 (2026-09-10 10:18:49 KST)

| 배치 | Slurm job id | 노드 / array | 동시성 | 자원 |
|---|---|---|---|---|
| PU (24 fold) | **6207** | n17 / idx 0–11 | %4 | `--cpus-per-task=4 --mem=20G`, `--gres` 없음 |
| PU (24 fold) | **6208** | n16 / idx 12–23 | %4 | 〃 |
| UODS (100 split) | **6209** | n17 / idx 0–49 | %4 | 〃 (`--dependency=afterany:6207:6208`) |
| UODS (100 split) | **6210** | n16 / idx 50–99 | %4 | 〃 (afterany 6207:6208) |

- **체인**: UODS(6209·6210)는 PU(6207·6208) **전량 종료 후 자동 시작**(afterany). PU·UODS 동시 미실행 → 노드당 ≤4잡(16코어 ≤ 절반 19) 유지.
- 파일럿 2 fold(PU `012to3_LONO1`, UODS `eval_run5`)는 이미 실제 results에 저장됨 → 배치에서 자동 스킵.
- index→fold 매핑: `runners/B_baselines/b1_dispatch.sh` (PU: SPLITS=[123to0,023to1,013to2,012to3]×LONO1–6, UODS: run_idx=5+idx).

## 파일럿 실측 (Slurm CPU, n17, cpus=4, IF n_jobs=4)

| 항목 | PU fold (012to3_LONO1) | UODS fold (run5) |
|---|---|---|
| 전체 시간 | 29.5분 (Elapsed 29:37) | 1.5분 (t_total 90s) |
| MaxRSS | 12.5 GB | ~8 GB |
| 지배 요소 | IF raw ~260s×5seed + OC-SVM raw 226s + OC-SVM band 162s (+load 53s, IF band 5s×5) | IF raw(작음)+OC-SVM |
| val FPR(전 combo) | 0.05 ✓ | 0.0501 ✓ |

- 로그인 노드에선 장시간 CPU 프로세스가 watchdog에 죽어(2회 확인) → **전량 Slurm 실행**.
- IF `n_jobs=4`는 결과 불변(순수 병렬), IF raw 730s→260s 가속. MaxRSS 12.5GB 반영해 `--mem=20G`.
- 예상 wallclock: PU 3웨이브×29.5분 ≈ 1.5h → 이후 UODS ~20분. **총 ~2h.**

## Band (b) 6-log-band z-vector 검증 (결정 1-A)

캐시 npz(p2_log_window_scores / UODS dump)에는 amp target 원값(rms_z) 미저장·S_amp 점수만 존재 →
**로더 rms_z 독립 재현**으로 제안 모델 amp target과의 동일성을 실증(`analysis/verify_B1_band_target.py`).

| fold | max\|Δ\| amp_normalize 불변성 | max\|Δ\| 독립 재현 | band edges |
|---|---|---|---|
| PU 012to3_LONO1 | **0.0** | **0.0** | [1,3,10,32,102,323,1025] ✓ |
| UODS eval_run5 | **0.0** | **0.0** | [1,3,10,32,102,323,1025] ✓ |

- **근거(코드)**: `Dataset/paderborn.py:540-545` band target `compute_band_rms(w[:,0])`를 amp_normalize 나눗셈 **이전**에 계산, z-score(667-685)는 amp_normalize 무관 train-normal 통계만 → `amp_normalize=False`의 `rms_z`가 학습(amp_normalize=True) target과 bit-identical.
- **로더 인자 diff (vs P-2 log 학습 러너 / UODS proposed 러너)**: window 2048·stride 1024·`--amp_n_bands 6`·`--amp_band_scheme log`·train/val/test ids 전부 동일. 차이 = ① `amp_normalize` (학습 True → B-1 False; rms_z 불변 실증·(a) raw 진폭 보존 위해 필수) ② `--amp_branch` (모델 전용 플래그, 로더 미소비).

## 규칙 (B-0 §2 계승, 동결)
- anomaly score = **−(IF score_samples / OC-SVM decision_function)**. threshold = val-normal 95pct.
- OC-SVM raw = fold당 min(N, n_train) 균등 비복원 **N=10,000, seed=2024**(결정론). band = 전량. IF = 전량.
- 하이퍼 고정: IF `n_estimators=100, random_state=seed`, OC-SVM `nu=0.05, gamma='scale'`. IF seed 2024~2028 5회, OC-SVM 1회.
- 파일럿 관찰: OC-SVM raw AUROC<0.5 fold 존재(012to3_LONO1=0.335) — 부호 버그 아님(동일 파이프라인의 IF/band는 정상 >0.5). raw 진폭 confound가 OC-SVM을 이기는 실제 실패(B 트랙 논지 부합). **하이퍼 불변 유지.**

## 산출물 (경로·스키마)
- PU: `results/Paderborn/b1_{if,ocsvm}_{raw,band}_window_scores/<split>_LONO<n>_s<seed>.npz` (288개). b3 스키마 동형(te_f/va_f/tr_f=anomaly score, te_lab/te_ids/te_rms/va_ids/va_rms/tr_ids/tr_rms/meta_json).
- UODS: `results/UODS/b1_{if,ocsvm}_{raw,band}_window_scores/eval_run<idx>_s<seed>.npz` (1200개). uods raw 스키마 동형(te_raw/va_raw/tr_raw + te_lab/te_ids/te_rms/te_fam/te_state/te_isball/va_ids/va_rms/tr_ids/meta_json).
- IF=seed당 파일, OC-SVM=s2024 단일.

## 재현 명령
```bash
# 제출 (파일럿 fold는 자동 스킵)
bash runners/slurm_run_cpu.sh pu                              # → PU 2 array job id 출력
AFTER=<puN17>:<puN16> bash runners/slurm_run_cpu.sh uods      # PU 전량 후 자동 시작
# 단일 fold 직접 (디버그)
conda run -n mtgflow python analysis/dump_B1_pu_window_scores.py --split 012to3 --lono 1
conda run -n mtgflow python analysis/dump_B1_uods_window_scores.py --run_idx 5
# band 검증
conda run -n mtgflow python analysis/verify_B1_band_target.py --pu_folds 012to3_LONO1 --uods_runs 5
# 집계·리포트 (전량 완료 후)
conda run -n mtgflow python analysis/report_B1_classical.py    # → reports/report_B1_classical.md
```

## 남은 작업 (별도 세션)
1. 124잡 전량 완료 확인(`squeue -u dyhwang`, npz 개수 PU 288 / UODS 1200).
2. `analysis/report_B1_classical.py` 실행 → `reports/report_B1_classical.md`(Table 4 전 열 + proposed/raw paired Wilcoxon·Holm(PU 24 fold)·paired(UODS 100 split)).
3. `TODO.md` B-1 상태 갱신, 최종 커밋(push는 승인 후).
