"""UODS 전량(eval 100 split) 집계·검증·통계 리포트 (CPU, 전 잡 완료 후).

통계 기본 단위 = **seed 평균 낸 100 split**(500 cell을 독립 표본으로 검정하지 않음).
주지표 = 동결 **Fisher**. equal-z는 사후 참고. shape/amp = 보조.

절차:
 1. 파일 수·잡 상태 검증(ckpt 1000·dump 1000·sacct COMPLETED). 실패/누락 목록화.
 2. split별 5 seed 평균 → split당 대표값(method×subgroup+fpr).
 3. 100 split mean±std·분포.
 4. raw vs Fisher paired(100 split): mean diff·95% CI(t·bootstrap)·Wilcoxon·#개선/100·Cohen d_z.
 5. shape·amp·equal-z 보조.
 6. subgroup: developing/faulty·family(I/O/B/C)·ball/non-ball·정상 FPR(val-95).
 7. 재현성: fisher≡product-of-p max|Δ| 전 셀, 전 score finite.

산출: results/UODS/uods_full_eval_aggregate.json + reports/report_uods_full_eval.md
사용: conda run -n mtgflow python analysis/report_uods_full.py
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))

from diagnose_uods_fusion import evaluate_run  # noqa: E402

RESULTS_UODS = os.path.join(PROJECT_ROOT, "results", "UODS")
CACHE_DIR = os.path.join(RESULTS_UODS, "uods_window_scores")
DIAG_DIR = os.path.join(RESULTS_UODS, "diag_uods_full")
AGG_PATH = os.path.join(RESULTS_UODS, "uods_full_eval_aggregate.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "report_uods_full_eval.md")

RUNS = list(range(5, 105))
SEEDS = [2024, 2025, 2026, 2027, 2028]
METHODS = ["raw", "shape", "amp", "equal_z", "fisher"]
SUBGROUPS = ["overall", "developing", "faulty", "nonball_developing", "nonball_faulty",
             "fam_inner", "fam_outer", "fam_ball", "fam_cage", "ball", "nonball"]


def _ck(i, model, s):
    return os.path.join(RESULTS_UODS, f"uods_eval_run{i}_{model}_s{s}", "model.pth")


def _dp(i, model, s):
    return os.path.join(CACHE_DIR, f"uods_eval_run{i}_{model}_s{s}.npz")


def verify_files():
    ck_missing, dp_missing = [], []
    for i in RUNS:
        for model in ("proposed", "raw"):
            for s in SEEDS:
                if not os.path.exists(_ck(i, model, s)):
                    ck_missing.append((i, model, s))
                if not os.path.exists(_dp(i, model, s)):
                    dp_missing.append((i, model, s))
    n_ck = 1000 - len(ck_missing)
    n_dp = 1000 - len(dp_missing)
    return dict(checkpoints_found=n_ck, checkpoints_expected=1000, checkpoints_missing=ck_missing,
                dumps_found=n_dp, dumps_expected=1000, dumps_missing=dp_missing)


def verify_jobs():
    try:
        out = subprocess.run(
            ["sacct", "-u", os.environ.get("USER", "dyhwang"), "-X", "--noheader",
             "--starttime", "now-24hours", "--format=JobName%40,State"],
            capture_output=True, text=True, timeout=60).stdout
    except Exception as e:
        return dict(error=str(e))
    states = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("run_UODS_eval_"):
            st = parts[1]
            states[st] = states.get(st, 0) + 1
    return states


def per_split_values():
    """split별 5 seed 평균 → X[i][method][subgroup], fpr[i][method]. + 재현성·누락 수집."""
    X, FPR = {}, {}
    repro_max = 0.0
    finite_ok = True
    used_splits = []
    for i in RUNS:
        seed_res = []
        for s in SEEDS:
            p, r = _dp(i, "proposed", s), _dp(i, "raw", s)
            if not (os.path.exists(p) and os.path.exists(r)):
                continue
            res = evaluate_run(p, r)
            seed_res.append(res)
            repro_max = max(repro_max, res["sanity"]["fisher_vs_prodp_absdiff"])
            finite_ok = finite_ok and res["sanity"]["all_finite"]
        if not seed_res:
            continue
        used_splits.append(i)
        methods = [m for m in METHODS if m in seed_res[0]["methods"]]
        X[i] = {m: {} for m in methods}
        FPR[i] = {}
        for m in methods:
            for sg in SUBGROUPS:
                vals = [sr["methods"][m]["subgroup"].get(sg, np.nan) for sr in seed_res]
                X[i][m][sg] = float(np.nanmean(vals))
            FPR[i][m] = float(np.nanmean([sr["methods"][m]["fpr_val95"] for sr in seed_res]))
    return X, FPR, used_splits, repro_max, finite_ok


def agg_mean_std(X, splits, method, sg):
    v = np.array([X[i][method][sg] for i in splits if method in X[i]], dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, v


def paired_vs_fisher(X, splits, sg, ref="raw", test="fisher", n_boot=10000):
    a = np.array([X[i][test][sg] for i in splits], float)   # fisher
    b = np.array([X[i][ref][sg] for i in splits], float)    # raw
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    d = a - b
    n = len(d)
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 0 else float("nan")
    tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
    ci_t = (mean_d - tcrit * se, mean_d + tcrit * se)
    # bootstrap CI (고정 seed 재현)
    rng = np.random.default_rng(12345)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    ci_boot = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    try:
        w_stat, w_p = stats.wilcoxon(a, b) if n > 0 and np.any(d != 0) else (float("nan"), float("nan"))
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    d_z = mean_d / sd if sd > 0 else float("nan")
    return dict(n=n, mean_diff=mean_d, std_diff=sd, ci95_t=[float(ci_t[0]), float(ci_t[1])],
                ci95_boot=list(ci_boot), wilcoxon_p=float(w_p),
                improved=int(np.sum(d > 0)), worsened=int(np.sum(d < 0)), tied=int(np.sum(d == 0)),
                cohen_dz=float(d_z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_splits", type=int, default=1, help="집계 진행 최소 split 수")
    args = ap.parse_args()

    files = verify_files()
    jobs = verify_jobs()
    X, FPR, splits, repro_max, finite_ok = per_split_values()
    n_splits = len(splits)
    print(f"=== UODS 전량 eval 집계 ===")
    print(f"파일: ckpt {files['checkpoints_found']}/1000, dump {files['dumps_found']}/1000 | "
          f"집계 가능 split: {n_splits}/100")
    print(f"잡 상태(sacct): {jobs}")
    print(f"재현성 fisher≡prodp max|Δ|={repro_max:.2e}, finite={finite_ok}")
    if n_splits < args.min_splits:
        print("집계 가능한 split 부족 — 종료.")
        return 1

    methods = [m for m in METHODS if all(m in X[i] for i in splits)]

    # 2~3) 100 split mean±std
    agg = {m: {"subgroup": {}, "fpr_val95": None} for m in methods}
    for m in methods:
        for sg in SUBGROUPS:
            mu, sd, _ = agg_mean_std(X, splits, m, sg)
            agg[m]["subgroup"][sg] = {"mean": mu, "std": sd}
        fv = np.array([FPR[i][m] for i in splits], float)
        agg[m]["fpr_val95"] = {"mean": float(np.nanmean(fv)), "std": float(np.nanstd(fv, ddof=1))}

    # 4·6) raw vs fisher paired — subgroup별
    paired = {sg: paired_vs_fisher(X, splits, sg) for sg in SUBGROUPS}
    # equal-z 참고: fisher vs equal_z, raw vs equal_z(overall만 참고)
    aux_paired = {
        "equal_z_vs_raw_overall": paired_vs_fisher(X, splits, "overall", ref="raw", test="equal_z"),
        "fisher_vs_equal_z_overall": paired_vs_fisher(X, splits, "overall", ref="equal_z", test="fisher"),
    }

    out = dict(
        pilot="UODS full eval (100 split × 2 model × 5 seed)",
        statistical_unit="seed-averaged 100 splits (NOT 500 independent cells)",
        primary_metric="frozen Fisher (equal-z auxiliary only)",
        n_splits_aggregated=n_splits, splits=splits,
        file_verification=files, job_states=jobs,
        reproducibility=dict(fisher_vs_prodp_max_absdiff=repro_max, all_finite=finite_ok),
        aggregate=agg, paired_raw_vs_fisher=paired, auxiliary_paired=aux_paired,
        per_split_overall={m: {str(i): X[i][m]["overall"] for i in splits} for m in methods},
    )
    os.makedirs(RESULTS_UODS, exist_ok=True)
    with open(AGG_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    _print_console(agg, paired, methods, n_splits, repro_max, files, jobs)
    _write_md(out, agg, paired, methods, n_splits)
    print(f"\n-> {AGG_PATH}\n-> {REPORT_PATH}")
    return 0


def _print_console(agg, paired, methods, n_splits, repro_max, files, jobs):
    print(f"\n--- method × overall/developing/faulty (mean±std over {n_splits} split) ---")
    for m in methods:
        s = agg[m]["subgroup"]
        print(f"  {m:<9} overall {s['overall']['mean']:.3f}±{s['overall']['std']:.3f} | "
              f"dev {s['developing']['mean']:.3f} | fault {s['faulty']['mean']:.3f} | "
              f"ball {s['ball']['mean']:.3f} | nonball {s['nonball']['mean']:.3f} | "
              f"fpr95 {agg[m]['fpr_val95']['mean']:.3f}")
    p = paired["overall"]
    print(f"\n--- raw vs Fisher paired (overall, n={p['n']} split) ---")
    print(f"  mean diff(Fisher−raw) = {p['mean_diff']:+.4f}  95%CI_t [{p['ci95_t'][0]:+.4f}, {p['ci95_t'][1]:+.4f}]  "
          f"boot [{p['ci95_boot'][0]:+.4f}, {p['ci95_boot'][1]:+.4f}]")
    print(f"  Wilcoxon p={p['wilcoxon_p']:.2e} | 개선 {p['improved']}/{p['n']} | Cohen dz={p['cohen_dz']:.2f}")


def _write_md(out, agg, paired, methods, n_splits):
    L = ["# UODS 외부 검증 — 전량 eval (100 split × 2 model × 5 seed)\n"]
    L.append(f"> 통계 단위 = **seed 평균 낸 {n_splits} split**(500 cell 독립 검정 아님). 주지표 = **동결 Fisher**, equal-z 참고only.")
    L.append(f"> 파일: ckpt {out['file_verification']['checkpoints_found']}/1000, dump {out['file_verification']['dumps_found']}/1000. "
             f"잡: {out['job_states']}. 재현성 fisher≡prodp max|Δ|={out['reproducibility']['fisher_vs_prodp_max_absdiff']:.2e}.\n")
    key = ["overall", "developing", "faulty", "nonball_developing", "nonball_faulty", "ball", "nonball"]
    L.append("## 1) method × subgroup AUROC (mean±std, 100 split)\n")
    L.append("| method | " + " | ".join(key) + " | fpr95 |")
    L.append("|---|" + "---|" * (len(key) + 1))
    for m in methods:
        row = " | ".join(f"{agg[m]['subgroup'][sg]['mean']:.3f}±{agg[m]['subgroup'][sg]['std']:.3f}" for sg in key)
        L.append(f"| {m} | {row} | {agg[m]['fpr_val95']['mean']:.3f}±{agg[m]['fpr_val95']['std']:.3f} |")
    L.append("\n## 2) family별 AUROC (mean±std)\n")
    fam = ["fam_inner", "fam_outer", "fam_ball", "fam_cage"]
    L.append("| method | " + " | ".join(s.replace("fam_", "") for s in fam) + " |")
    L.append("|---|" + "---|" * len(fam))
    for m in methods:
        L.append(f"| {m} | " + " | ".join(f"{agg[m]['subgroup'][sg]['mean']:.3f}±{agg[m]['subgroup'][sg]['std']:.3f}" for sg in fam) + " |")
    L.append("\n## 3) raw vs Fisher — split-단위 paired (주분석)\n")
    L.append("| subgroup | Fisher−raw | 95% CI(t) | 95% CI(boot) | Wilcoxon p | 개선/n | Cohen dz |")
    L.append("|---|---|---|---|---|---|---|")
    for sg in key + fam:
        p = paired[sg]
        L.append(f"| {sg} | {p['mean_diff']:+.4f} | [{p['ci95_t'][0]:+.4f}, {p['ci95_t'][1]:+.4f}] | "
                 f"[{p['ci95_boot'][0]:+.4f}, {p['ci95_boot'][1]:+.4f}] | {p['wilcoxon_p']:.2e} | "
                 f"{p['improved']}/{p['n']} | {p['cohen_dz']:.2f} |")
    L.append("\n## 4) 보조 (equal-z 참고only)\n")
    az = out["auxiliary_paired"]["equal_z_vs_raw_overall"]
    fz = out["auxiliary_paired"]["fisher_vs_equal_z_overall"]
    L.append(f"- equal-z vs raw (overall): diff {az['mean_diff']:+.4f}, 개선 {az['improved']}/{az['n']}, Wilcoxon p={az['wilcoxon_p']:.2e}.")
    L.append(f"- fisher vs equal-z (overall, 참고): diff {fz['mean_diff']:+.4f}, 개선 {fz['improved']}/{fz['n']}, Wilcoxon p={fz['wilcoxon_p']:.2e}. "
             f"**동결 규칙상 fusion 재선택 안 함.**")
    L.append("\n## 5) 가드레일\n")
    L.append("- 구조·N·band·fusion 동결. 결과로 모델 재선택·UODS 폐기 없음. 외부 baseline·g-final 병합/push 미실시(별도 승인).")
    L.append("- test bearing은 어떤 fit/calibration에도 미사용. scaler·band=train-fit / Fisher·threshold=val. ball no-load confound는 ball/non-ball 분리로 격리.")
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
