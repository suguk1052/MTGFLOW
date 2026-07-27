# LONO_FiLM_measured_5seeds — FiLM(A안) measured 조건부 주입 실험 큐

FiLM 방식(A안: condition **C** 변조) measured-mean 실험. concat 대신 C_op에서 Δγ,β를 만들어
`C_mod = (1+Δγ)⊙C + β`로 C를 변조(MAF 입력 차원 불변, 항등 초기화로 시작 시 no-meta와 동일).

- 플래그: `--use_meta --meta_source measured --measured_meta_stats mean --meta_inject film`
- 시드: `2024 2025 2026 2027 2028` (셀당 5 seed, 한 `.sh`)
- `--log_test_auroc` **off** (속도).
- 그리드: **C1(pooled) 6셀 + C2(LOSO) 24셀 = 30셀 × 5 seed = 150 run**.
- 셀 = (config × LONO)의 5 seed. 셀 원자성 유지(한 셀의 5 seed는 항상 같은 체인).

## 제출

```bash
# 미리보기(제출 안 함)
DRY_RUN=1 bash runners/LONO_FiLM_measured_5seeds/submit_all_6chains.sh
# ★ 제출 전 sinfo로 n16이 V100-16 파티션에 정상 스케줄되는지 확인할 것.
# 전체 6체인 제출 (체인1~3 n16, 체인4~6 n17)
bash runners/LONO_FiLM_measured_5seeds/submit_all_6chains.sh
# 개별 체인만
bash runners/LONO_FiLM_measured_5seeds/submit_chain2_n16.sh
```

각 체인 = `slurm_run.sh`로 (학습 && test) 한 잡 × 5셀을 `afterok` 순차 체인.
6장 병렬 = 6개 독립 체인. 노드는 각 체인 submit이 `NODELIST`로 지정.

## 6체인 ↔ 셀 ↔ RUN_NAME(=결과 폴더명) 매핑

RUN_NAME 규칙: C1 = `CA_raw_vib_0123C_LONO{n}_FiLM_measured_5seeds`,
C2 = `CA_raw_vib_{split}_LONO{n}_FiLM_measured_5seeds`.
결과는 `results/Paderborn/<RUN_NAME>_s{seed}/` 및 집계 `results/Paderborn/<RUN_NAME>/summary_seeds.json`.

| 체인 | 노드 | 셀 (config-LONO) | RUN_NAME |
| --- | --- | --- | --- |
| 1 | n16 | C1-L1 | CA_raw_vib_0123C_LONO1_FiLM_measured_5seeds |
| 1 | n16 | C1-L2 | CA_raw_vib_0123C_LONO2_FiLM_measured_5seeds |
| 1 | n16 | C1-L3 | CA_raw_vib_0123C_LONO3_FiLM_measured_5seeds |
| 1 | n16 | C1-L4 | CA_raw_vib_0123C_LONO4_FiLM_measured_5seeds |
| 1 | n16 | C1-L5 | CA_raw_vib_0123C_LONO5_FiLM_measured_5seeds |
| 2 | n16 | C1-L6 | CA_raw_vib_0123C_LONO6_FiLM_measured_5seeds |
| 2 | n16 | 123to0-L1 | CA_raw_vib_123to0_LONO1_FiLM_measured_5seeds |
| 2 | n16 | 123to0-L2 | CA_raw_vib_123to0_LONO2_FiLM_measured_5seeds |
| 2 | n16 | 123to0-L3 | CA_raw_vib_123to0_LONO3_FiLM_measured_5seeds |
| 2 | n16 | 123to0-L4 | CA_raw_vib_123to0_LONO4_FiLM_measured_5seeds |
| 3 | n16 | 123to0-L5 | CA_raw_vib_123to0_LONO5_FiLM_measured_5seeds |
| 3 | n16 | 123to0-L6 | CA_raw_vib_123to0_LONO6_FiLM_measured_5seeds |
| 3 | n16 | 012to3-L1 | CA_raw_vib_012to3_LONO1_FiLM_measured_5seeds |
| 3 | n16 | 012to3-L2 | CA_raw_vib_012to3_LONO2_FiLM_measured_5seeds |
| 3 | n16 | 012to3-L3 | CA_raw_vib_012to3_LONO3_FiLM_measured_5seeds |
| 4 | n17 | 012to3-L4 | CA_raw_vib_012to3_LONO4_FiLM_measured_5seeds |
| 4 | n17 | 012to3-L5 | CA_raw_vib_012to3_LONO5_FiLM_measured_5seeds |
| 4 | n17 | 012to3-L6 | CA_raw_vib_012to3_LONO6_FiLM_measured_5seeds |
| 4 | n17 | 023to1-L1 | CA_raw_vib_023to1_LONO1_FiLM_measured_5seeds |
| 4 | n17 | 023to1-L2 | CA_raw_vib_023to1_LONO2_FiLM_measured_5seeds |
| 5 | n17 | 023to1-L3 | CA_raw_vib_023to1_LONO3_FiLM_measured_5seeds |
| 5 | n17 | 023to1-L4 | CA_raw_vib_023to1_LONO4_FiLM_measured_5seeds |
| 5 | n17 | 023to1-L5 | CA_raw_vib_023to1_LONO5_FiLM_measured_5seeds |
| 5 | n17 | 023to1-L6 | CA_raw_vib_023to1_LONO6_FiLM_measured_5seeds |
| 5 | n17 | 013to2-L1 | CA_raw_vib_013to2_LONO1_FiLM_measured_5seeds |
| 6 | n17 | 013to2-L2 | CA_raw_vib_013to2_LONO2_FiLM_measured_5seeds |
| 6 | n17 | 013to2-L3 | CA_raw_vib_013to2_LONO3_FiLM_measured_5seeds |
| 6 | n17 | 013to2-L4 | CA_raw_vib_013to2_LONO4_FiLM_measured_5seeds |
| 6 | n17 | 013to2-L5 | CA_raw_vib_013to2_LONO5_FiLM_measured_5seeds |
| 6 | n17 | 013to2-L6 | CA_raw_vib_013to2_LONO6_FiLM_measured_5seeds |

각 체인 5셀=25 run. 총 30셀=150 run.

## LONO / LOSO 참조

- LONO(train/val/test_norm): L1 K003K004K005K006/K002/K001, L2 K001K004K005K006/K003/K002,
  L3 K001K002K005K006/K004/K003, L4 K001K002K003K006/K005/K004,
  L5 K001K002K003K004/K006/K005, L6 K002K003K004K005/K001/K006.
- C2 LOSO split(held-out target): 123to0=N15_M07_F10, 012to3=N15_M07_F04,
  023to1=N09_M07_F10, 013to2=N15_M01_F10.

## 결과 해석 주의

`--log_test_auroc` off이므로: FiLM 5-seed 평균이 baseline(no-meta/static/measured-concat, 기존 결과 재사용)보다
**오르면 신뢰**(val_loss 기반 checkpoint 선택 핸디캡을 안고도 이긴 것). **안 오르면** '방법 문제 vs
선택 손해'를 구분할 수 없어 결론 유보. baseline은 재실행하지 말 것(기존 5-seed 재사용).
