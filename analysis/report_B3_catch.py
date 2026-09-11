"""B-3 CATCH 표 편입 (재사용 검증 + per-window id/rms 복원, analysis-only, 모델 없음).

B-0에서 CATCH checkpoint = val-normal 재구성 loss 선택 확인(누수 0, 재추론 불필요).
overall/zero/comp = metrics.json `auroc` 집계(24 LOSO fold × 5 seed).

**정정(2026-09-10)**: CATCH LOSO fold의 `scores.npz`는 **window-level**(test_scores 길이 = n_test,
b3 캐시와 동일 74,895 등, ratio 1.0). (앞선 "point-level 299,521"은 B2_LONO 비-LOSO 오샘플.)
→ CATCH 로더를 동일 인자로 재실행(모델 없음)해 per-window **bearing id·RMS**를 복원하고,
   순서 일치(count + test_labels array_equal) 검증 후 amp/shape-sensitive·진폭군 FPR·ρ(RMS,score) 열을 채운다.
각주: CATCH scores.npz에 train score 없음 → 정상 pool = val + test-normal(제안/B-1은 tr 포함) 차이.

산출: reports/report_B3_catch.md
사용: conda run -n mtgflow python analysis/report_B3_catch.py
"""
import json
import os
import sys
import contextlib
import io

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))
sys.path.insert(0, os.path.normpath(os.path.join(PROJECT_ROOT, "..", "CATCH", "pu")))

from report_B1_classical import paired, holm, fmt, _mean, _ms, pu_proposed_folds, pu_raw_folds  # noqa: E402
from diagnose_G3a_amp_bands import (  # noqa: E402
    _score_block, SPLIT_INFO, AMP_SENSITIVE, SHAPE_SENSITIVE,
)
from paderborn_dataset import build_pu_loaders  # noqa: E402 (CATCH/pu 로더)

CATCH_ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "CATCH", "result_pu"))
SPLITS = ["123to0", "023to1", "013to2", "012to3"]
LONOS = [1, 2, 3, 4, 5, 6]
SEEDS = [2024, 2025, 2026, 2027, 2028]


def _run_dir(split, lono, seed):
    return os.path.join(CATCH_ROOT, f"LOSO_{split}_LONO{lono}_s{seed}")


def recover_meta(split, lono):
    """CATCH 로더 재실행(모델 없음)으로 per-window id·rms 복원. seed 무관(데이터 동일)."""
    cfg = json.load(open(os.path.join(_run_dir(split, lono, 2024), "config.json")))
    with contextlib.redirect_stdout(io.StringIO()):
        _, va, te, _ = build_pu_loaders(
            train_loads=cfg["train_load_setting"], test_loads=cfg["test_load_setting"],
            train_ids=cfg["train_ids"], val_ids=cfg["val_ids"], test_norm_ids=cfg["test_norm_ids"],
            exclude_ids=cfg.get("exclude_ids"), channels=cfg.get("channels", ["vibration_1"]),
            window_size=cfg["window_size"], stride_size=cfg["stride_size"],
            downsample=cfg.get("downsample", 1), batch_size=256, num_workers=0)

    def _rms(ds):
        w = np.asarray(ds.windows, dtype=np.float64)  # [N, seq_len, n_ch]
        return np.sqrt(np.mean(w[:, :, 0] ** 2, axis=1))
    te_ids = np.asarray(te.dataset.ids)
    va_ids = np.asarray(va.dataset.ids)
    return {"te_ids": te_ids, "te_rms": _rms(te.dataset), "te_lab": np.asarray(te.dataset.label, int),
            "va_ids": va_ids, "va_rms": _rms(va.dataset)}


def fold_full_metrics(split, lono):
    """복원 id/rms + CATCH per-seed score로 amp/shape·FPR·ρ 계산(seed 평균). 순서 검증 포함."""
    rec = recover_meta(split, lono)
    te_ids, te_rms, te_lab_rec = rec["te_ids"], rec["te_rms"], rec["te_lab"]
    va_ids, va_rms = rec["va_ids"], rec["va_rms"]
    blks, aurocs, verify = [], [], {"ok": 0, "fail": 0}
    for s in SEEDS:
        sp = os.path.join(_run_dir(split, lono, s), "scores.npz")
        if not os.path.exists(sp):
            continue
        z = np.load(sp, allow_pickle=True)
        te_sc, va_sc, te_lab = z["test_scores"], z["val_scores"], np.asarray(z["test_labels"], int)
        # 순서 일치 검증: count + label
        if len(te_sc) != len(te_ids) or len(va_sc) != len(va_ids) or not np.array_equal(te_lab, te_lab_rec):
            verify["fail"] += 1
            continue
        verify["ok"] += 1
        mj = json.load(open(os.path.join(_run_dir(split, lono, s), "metrics.json")))
        aurocs.append(float(mj["auroc"]))
        tgt = te_lab == 0
        norm_rms = np.concatenate([va_rms, te_rms[tgt]])   # train score 없음 → val+test-normal
        norm_ids = np.concatenate([va_ids, te_ids[tgt]])
        blk = _score_block("catch", te_sc, va_sc, te_lab, te_ids, te_rms,
                           np.array([]), va_sc, te_rms[tgt], norm_rms, norm_ids, 95.0)
        blks.append(blk)
    if not blks:
        return None
    # per-fault → amp/shape 군
    fa = {}
    for b in blks:
        for fid, d in b["per_fault_auroc"].items():
            fa.setdefault(fid, []).append(d["auroc"])
    fa = {fid: _mean(v) for fid, v in fa.items()}
    return {
        "auroc": _mean(aurocs),
        "amp_sensitive": _mean([fa[f] for f in fa if f in AMP_SENSITIVE]),
        "shape_sensitive": _mean([fa[f] for f in fa if f in SHAPE_SENSITIVE]),
        "high_fpr": _mean([b["high_amp_fpr"] for b in blks]),
        "low_fpr": _mean([b["low_amp_fpr"] for b in blks]),
        "rho": _mean([b["rho_rms_score"] for b in blks]),
        "verify": verify,
    }


def main():
    report = os.path.join(PROJECT_ROOT, "reports", "report_B3_catch.md")
    proposed = pu_proposed_folds(); raw = pu_raw_folds()

    folds, overall_auroc = {}, {}
    v_ok = v_fail = 0
    for split in SPLITS:
        for lono in LONOS:
            r = fold_full_metrics(split, lono)
            if r is None:
                continue
            folds[(split, lono)] = r
            overall_auroc[(split, lono)] = r["auroc"]
            v_ok += r["verify"]["ok"]; v_fail += r["verify"]["fail"]

    keys = list(overall_auroc)
    zero = [overall_auroc[k] for k in keys if SPLIT_INFO[k[0]][1] == "zero-support"]
    comp = [overall_auroc[k] for k in keys if SPLIT_INFO[k[0]][1] == "compositional"]
    # 전 fold 군 평균
    def col(name):
        return _ms([folds[k][name] for k in keys])
    amp_s = _mean([folds[k]["amp_sensitive"] for k in keys])
    shape_s = _mean([folds[k]["shape_sensitive"] for k in keys])
    pr = paired(overall_auroc, raw); pp = paired(overall_auroc, proposed)
    hr = holm([pr["p"]]); hp = holm([pp["p"]])

    L = ["# B-3. CATCH (재구성형 TSAD, ICLR'25) — 표 편입 + per-window 메타 복원", ""]
    L.append("B-0 확인: checkpoint=val-normal 재구성 loss(누수 0) → 재추론 불필요. metrics.json `auroc` 집계 + "
             "**CATCH 로더 재실행(모델 없음)으로 per-window id·rms 복원**해 subgroup 열을 채움.")
    L.append(f"**순서 일치 검증**: (count + test_labels array_equal) — 통과 {v_ok} / 실패 {v_fail} (fold×seed). "
             "patch 64/64, seq_len 2048, K=1 채널.")
    L.append("")
    L.append("## PU (24 LOSO fold) — Table 4 열 (seed 평균)")
    L.append("")
    L.append("| 모델 | overall | zero-support | compositional | amp-sensitive | shape-sensitive | 정상FPR고 | 정상FPR저 | ρ(RMS,score) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    L.append(f"| CATCH | {fmt(*_ms([overall_auroc[k] for k in keys]))} | {fmt(*_ms(zero))} | {fmt(*_ms(comp))} | "
             f"{fmt(amp_s)} | {fmt(shape_s)} | {fmt(*col('high_fpr'))} | {fmt(*col('low_fpr'))} | {fmt(*col('rho'))} |")
    L.append(f"| raw MTGFlow (참조) | {fmt(_mean(list(raw.values())))} | | | | | | | |")
    L.append(f"| 제안 N6+log+Fisher (참조) | {fmt(_mean(list(proposed.values())))} | | | | | | | |")
    L.append("")
    L.append("### paired (24 fold overall AUROC, Wilcoxon+Holm)")
    L.append("| 대상 | Δ(CATCH−대상) | p(Holm) | CATCH 우세/24 |")
    L.append("|---|---|---|---|")
    L.append(f"| vs raw MTGFlow | {pr['mean_diff']:+.3f} | {fmt(hr[0])} | {pr['n_pos']}/{pr['n']} |")
    L.append(f"| vs 제안(Fisher) | {pp['mean_diff']:+.3f} | {fmt(hp[0])} | {pp['n_pos']}/{pp['n']} |")
    L.append("")
    L.append("## 각주 (B-G1/B-G2)")
    L.append("- **정상 pool 차이**: CATCH scores.npz에 train score 없음 → 진폭군 정상 FPR·ρ의 정상 pool = **val + test-normal**"
             "(제안/B-1은 train+val+test-normal). amp/shape-sensitive는 test-normal을 음성으로 쓰므로 영향 없음.")
    L.append("- **정정**: CATCH LOSO scores.npz는 window-level(= n_test)로 확인 → id/rms 복원·정렬 검증 후 열 채움 완료.")
    L.append("- **프레이밍(B-G2)**: CATCH는 표준 재구성형 TSAD 참고 행. ① 시점 단위 탐지용 설계 ② K=1 채널이라 채널 융합 비활성 "
             "③ seq_len 2048 제약. '우세' 주장 대상 아님. 표값 = LOSO 0.539(B2 pooled 0.524는 표 미사용).")
    L.append("")
    L.append("> analysis-only. 재추론·모델 변경 없음. 복원은 CATCH 로더(동일 인자) 재실행으로 id/rms만 산출.")

    with open(report, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {report}\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
