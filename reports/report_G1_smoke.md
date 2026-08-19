# 작업 G-1 스모크 로그 (파이프라인 동작·안정성 확인 — 성능 판정 아님)

> ⚠️ 이 문서는 **1 fold × 3 epoch 스모크**의 동작 검증 기록이다. 성능 판정·게이트 결론은 여기 없다.
> 24 fold × seed 2026 게이트 결과는 `report_G1_disentangle.md`(별도 파일)에 기록한다.

## 설정
- fold: `023to1 LONO1` (저속 N09 unseen = zero-support, target-normal K001=고진폭). seed 2026.
- 학습: `main.py --amp_normalize --amp_branch --epochs 3`, RUN_NAME `g1_smoke_023to1_LONO1`.
- 결과 폴더: `results/Paderborn/g1_smoke_023to1_LONO1_s2026/` (24-fold `g1_*`와 분리).
- slurm 잡 3473 COMPLETED (elapsed 00:04:06, V100-16 1장, n17).

## 확인 항목 (전부 통과)

**1. train/val/test end-to-end 완주 · loss 유한·감소** (`train_log.jsonl`)

| epoch | train_loss | val_loss(=best) |
|---|---|---|
| 0 | −2.067 | −3.437 |
| 1 | −4.061 | −4.050 |
| 2 | −4.262 | −5.181 |

→ joint NLL(`-(shape_NLL+amp_NLL)`) 정상 학습, NaN/inf 없음. checkpoint = 최소 val loss(epoch 2). test 74,850 window 추론 완주(cuda).

**2. L_shape / L_amp 분리 동작** — joint loss가 유한하게 수렴(위 표). log_sigma clamp[-7,7]로 σ 방어.

**3. amp-head μ·σ numerical stability** (val 1024 window 직접 추출)
- μ: min −0.0149 / max −0.0055 / mean −0.0108, 유한.
- σ: min 1.0071 / max 1.0080 / mean 1.0075, 유한, **전부 >0**(collapse·폭주 없음).
- S_amp: min 0.926 / max 1.193, 유한. → **STABILITY_OK**.
- (3 epoch라 amp-head가 아직 초기값 근방 N(0,1)에 가까움 — 정상. 40 epoch 게이트에서 conditioning 특화 관찰 예정.)

**4. S_shape / S_amp / S_total 분리 저장 + val-normal 표준화** (`paderborn_per_bearing_metrics.json`)
- `overall_auroc_shape=0.988`, `overall_auroc_amp=0.635`, `overall_auroc_total_std=0.773`(=primary `overall_auroc`), `overall_auroc_total_raw=0.681` — **4종 모두 유한값으로 따로 기록**.
- `val_norm_stats`: mu_shape −6.192 / sd_shape 0.0287, mu_amp 1.010 / **sd_amp 0.085(>0)** → val-normal 표준화 경로 동작.
- `amp_branch=true`, `amp_normalize=true`, `amp_branch_hidden=32` metadata 기록.

## 결론
파이프라인(학습 joint NLL → checkpoint → test 분리 채점 → val-normal 표준화 → JSON 기록)이
end-to-end로 동작하고 수치적으로 안정적임을 확인. **설계 변경 없이 원래 epoch(40)로 24 fold 게이트 진행.**
(스모크 수치는 3 epoch·단일 fold이므로 성능 해석 금지.)
