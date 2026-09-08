"""UODS "1-split × 5-seed extended pilot" 집계 리포트 (CPU, 잡 완료 후).

seed별 proposed/raw dump(.npz)를 evaluate_run으로 평가해 method×subgroup mean±std를 낸다.
seed당 실측 학습시간은 slurm .out의 "[Seed <s>] Train wall-clock time: <sec>s"를 파싱.

산출: results/UODS/uods_extended_pilot_aggregate.json + reports/report_uods_extended_pilot.md
사용: conda run -n mtgflow python analysis/report_uods_pilot.py --seeds 2024 2025 2026 2027 2028
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))

from diagnose_uods_fusion import evaluate_run  # noqa: E402

CACHE_DIR = os.path.join(PROJECT_ROOT, "results", "UODS", "uods_window_scores")
DIAG_DIR = os.path.join(PROJECT_ROOT, "results", "UODS", "diag_uods_fusion")
SLURM_LOG_DIR = os.path.join(PROJECT_ROOT, "runners", "slurm_logs")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "report_uods_extended_pilot.md")
AGG_PATH = os.path.join(PROJECT_ROOT, "results", "UODS", "uods_extended_pilot_aggregate.json")

METHOD_ORDER = ["raw", "shape", "amp", "equal_z", "fisher"]
SUBGROUPS = ["overall", "developing", "faulty", "nonball_developing", "nonball_faulty",
             "fam_inner", "fam_outer", "fam_ball", "fam_cage", "ball", "nonball"]


def parse_train_times(job_name):
    """slurm .out에서 seed별 학습 wall-clock(sec) 파싱 → {seed: sec}. 최신 로그 사용."""
    times = {}
    logs = sorted(glob.glob(os.path.join(SLURM_LOG_DIR, f"{job_name}_*.out")))
    pat = re.compile(r"\[Seed (\d+)\] Train wall-clock time: ([\d.]+)s")
    for lg in logs:
        try:
            with open(lg, errors="ignore") as f:
                for line in f:
                    m = pat.search(line)
                    if m:
                        times[int(m.group(1))] = float(m.group(2))
        except OSError:
            pass
    return times


def mean_std(vals):
    a = np.asarray([v for v in vals if v == v], dtype=float)  # nan 제외
    if len(a) == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std()), len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026, 2027, 2028])
    ap.add_argument("--proposed_prefix", default="uods_pilot_proposed")
    ap.add_argument("--raw_prefix", default="uods_pilot_raw")
    ap.add_argument("--cache_dir", default=CACHE_DIR)
    ap.add_argument("--threshold_percentile", type=float, default=95.0)
    args = ap.parse_args()

    per_seed = {}
    missing = []
    os.makedirs(DIAG_DIR, exist_ok=True)
    for s in args.seeds:
        p = os.path.join(args.cache_dir, f"{args.proposed_prefix}_s{s}.npz")
        r = os.path.join(args.cache_dir, f"{args.raw_prefix}_s{s}.npz")
        if not os.path.exists(p) or not os.path.exists(r):
            missing.append((s, os.path.exists(p), os.path.exists(r)))
            continue
        res = evaluate_run(p, r, args.threshold_percentile)
        per_seed[s] = res
        with open(os.path.join(DIAG_DIR, f"{args.proposed_prefix}_s{s}.json"), "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    if missing:
        print("⚠️ 누락 seed(dump 없음):", missing)
    if not per_seed:
        print("집계할 seed 결과가 없습니다. dump 완료 여부 확인.")
        return 1

    seeds_ok = sorted(per_seed)
    methods_present = [m for m in METHOD_ORDER if m in next(iter(per_seed.values()))["methods"]]

    # 집계: method×subgroup mean±std, fpr mean±std
    agg = {}
    for m in methods_present:
        agg[m] = {"subgroup": {}, "fpr_val95": None}
        for sg in SUBGROUPS:
            vals = [per_seed[s]["methods"][m]["subgroup"].get(sg, float("nan")) for s in seeds_ok]
            mu, sd, n = mean_std(vals)
            agg[m]["subgroup"][sg] = {"mean": mu, "std": sd, "n": n,
                                      "per_seed": {str(s): per_seed[s]["methods"][m]["subgroup"].get(sg) for s in seeds_ok}}
        fmu, fsd, _ = mean_std([per_seed[s]["methods"][m]["fpr_val95"] for s in seeds_ok])
        agg[m]["fpr_val95"] = {"mean": fmu, "std": fsd}

    sanity = {str(s): per_seed[s]["sanity"] for s in seeds_ok}
    prop_times = parse_train_times("run_UODS_pilot_proposed")
    raw_times = parse_train_times("run_UODS_pilot_raw")

    meta0 = next(iter(per_seed.values()))
    out = dict(pilot="1-split x 5-seed extended pilot", split_source=meta0.get("source"),
               band_scheme=meta0.get("band_scheme"), band_edges=meta0.get("band_edges"),
               seeds=seeds_ok, methods_present=methods_present, threshold_percentile=args.threshold_percentile,
               aggregate=agg, per_seed_overall={m: {str(s): per_seed[s]["methods"][m]["subgroup"]["overall"]
                                                    for s in seeds_ok} for m in methods_present},
               sanity=sanity, train_time_sec=dict(proposed=prop_times, raw=raw_times), missing=missing)
    with open(AGG_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # --- 콘솔 표 ---
    print(f"\n=== UODS extended pilot (split={meta0.get('source')}, seeds={seeds_ok}) ===")
    hdr = f"{'method':<9}" + "".join(f"{sg[:10]:>12}" for sg in SUBGROUPS) + f"{'fpr95':>10}"
    print(hdr)
    for m in methods_present:
        row = f"{m:<9}"
        for sg in SUBGROUPS:
            c = agg[m]["subgroup"][sg]
            row += f"{c['mean']:>6.3f}±{c['std']:<5.3f}"
        f_ = agg[m]["fpr_val95"]
        row += f"{f_['mean']:>5.3f}±{f_['std']:<4.3f}"
        print(row)
    fisher_sanity_max = max(per_seed[s]["sanity"]["fisher_vs_prodp_absdiff"] for s in seeds_ok)
    print(f"\nFisher sanity(≡product-of-p) max|ΔAUROC| over seeds = {fisher_sanity_max:.2e}")
    if prop_times:
        pt = np.array(list(prop_times.values()))
        print(f"proposed 학습 seed당: mean {pt.mean():.1f}s (min {pt.min():.1f}/max {pt.max():.1f}), n={len(pt)}")
    if raw_times:
        rt = np.array(list(raw_times.values()))
        print(f"raw      학습 seed당: mean {rt.mean():.1f}s (min {rt.min():.1f}/max {rt.max():.1f}), n={len(rt)}")

    _write_markdown(out, agg, methods_present, seeds_ok, per_seed, prop_times, raw_times)
    print(f"\n-> {AGG_PATH}\n-> {REPORT_PATH}")
    return 0


def _write_markdown(out, agg, methods, seeds, per_seed, prop_times, raw_times):
    def ms(m, sg):
        c = agg[m]["subgroup"][sg]
        return f"{c['mean']:.3f}±{c['std']:.3f}"
    L = []
    L.append("# UODS 외부 검증 — 1-split × 5-seed extended pilot\n")
    L.append(f"> split=**{out.get('split_source')}**(Vieira tuning/run_0, 2/1/2), seeds={seeds}. "
             f"band=N=6 **{out.get('band_scheme')}** edges={out.get('band_edges')}, threshold={out['threshold_percentile']:.0f}pct(val-normal).\n")
    L.append("> 파이프라인 sanity·비용 측정 전용(모델 설정 변경·데이터셋 폐기 근거 아님). scaler·band=train-fit / Fisher·threshold=val, test bearing 미사용.\n")
    key_sg = ["overall", "developing", "faulty", "nonball_developing", "nonball_faulty", "ball", "nonball"]
    L.append("\n## 1) method × subgroup AUROC (mean±std, 5 seed)\n")
    L.append("| method | " + " | ".join(key_sg) + " | fpr95 |")
    L.append("|---|" + "---|" * (len(key_sg) + 1))
    for m in methods:
        f_ = agg[m]["fpr_val95"]
        L.append(f"| {m} | " + " | ".join(ms(m, sg) for sg in key_sg) + f" | {f_['mean']:.3f}±{f_['std']:.3f} |")
    L.append("\n## 2) family별 AUROC (mean±std)\n")
    fam_sg = ["fam_inner", "fam_outer", "fam_ball", "fam_cage"]
    L.append("| method | " + " | ".join(s.replace("fam_", "") for s in fam_sg) + " |")
    L.append("|---|" + "---|" * len(fam_sg))
    for m in methods:
        L.append(f"| {m} | " + " | ".join(ms(m, sg) for sg in fam_sg) + " |")
    L.append("\n## 3) seed별 overall AUROC\n")
    L.append("| method | " + " | ".join(str(s) for s in seeds) + " |")
    L.append("|---|" + "---|" * len(seeds))
    for m in methods:
        L.append(f"| {m} | " + " | ".join(f"{per_seed[s]['methods'][m]['subgroup']['overall']:.3f}" for s in seeds) + " |")
    L.append("\n## 4) sanity & 실측 시간\n")
    fmax = max(per_seed[s]["sanity"]["fisher_vs_prodp_absdiff"] for s in seeds)
    n0 = per_seed[seeds[0]]["sanity"]
    L.append(f"- Fisher ≡ product-of-p: seed 전체 max|ΔAUROC| = **{fmax:.2e}** (0 근처 = 정상).")
    L.append(f"- test 구성: normal {n0['n_norm']} / fault {n0['n_fault']} window. 전 score finite.")
    if prop_times:
        pt = np.array(list(prop_times.values()))
        L.append(f"- **proposed 학습 seed당**: mean **{pt.mean():.1f}s** (min {pt.min():.1f}/max {pt.max():.1f}), n={len(pt)}.")
    if raw_times:
        rt = np.array(list(raw_times.values()))
        L.append(f"- **raw 학습 seed당**: mean **{rt.mean():.1f}s** (min {rt.min():.1f}/max {rt.max():.1f}), n={len(rt)}.")
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
