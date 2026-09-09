"""UODS 전량 eval 결과 그림 3장 생성 (CPU). 입력 = results/UODS/uods_full_eval_aggregate.json.

Fig1 산점도(raw vs Fisher, 100 split, 대각선) / Fig2 subgroup Δ forest(95% CI) / Fig3 family×method 막대.
색: Okabe-Ito(colorblind-safe). 텍스트는 영문(matplotlib 기본 폰트 CJK 미지원). 출력 → reports/figs_uods/.
사용: conda run -n mtgflow python analysis/plot_uods_full.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGG = os.path.join(PROJECT_ROOT, "results", "UODS", "uods_full_eval_aggregate.json")
FIGDIR = os.path.join(PROJECT_ROOT, "reports", "figs_uods")

# Okabe-Ito (colorblind-safe). 엔티티 고정 배정.
C_RAW = "#999999"      # gray  (baseline)
C_SHAPE = "#56B4E9"    # sky blue
C_AMP = "#E69F00"      # orange
C_EQZ = "#009E73"      # bluish green
C_FISHER = "#0072B2"   # blue (primary)
C_WEAK = "#D55E00"     # vermillion (정직한 약점 강조: ball)
METHOD_COLOR = {"raw": C_RAW, "shape": C_SHAPE, "amp": C_AMP, "equal_z": C_EQZ, "fisher": C_FISHER}
METHOD_LABEL = {"raw": "raw MTGFlow", "shape": "shape", "amp": "amp", "equal_z": "equal-z (ref)", "fisher": "Fisher (proposed)"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.labelcolor": "#222222", "text.color": "#222222",
    "xtick.color": "#444444", "ytick.color": "#444444",
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.8,
})


def load():
    with open(AGG) as f:
        return json.load(f)


def fig1_scatter(d):
    ps = d["per_split_overall"]
    splits = sorted(ps["raw"], key=lambda k: int(k))
    x = np.array([ps["raw"][s] for s in splits])
    y = np.array([ps["fisher"][s] for s in splits])
    above = y > x
    n_above = int(above.sum())

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    lo, hi = 0.58, 1.0
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.5, color="#888888", zorder=1, label="y = x")
    ax.scatter(x[above], y[above], s=42, c=C_FISHER, alpha=0.80, edgecolors="white",
               linewidths=0.5, zorder=3, label=f"Fisher > raw  ({n_above}/100)")
    ax.scatter(x[~above], y[~above], s=60, c=C_WEAK, alpha=0.95, edgecolors="white",
               linewidths=0.6, marker="D", zorder=4, label=f"raw ≥ Fisher  ({100 - n_above}/100)")
    # 평균점(별)
    ax.scatter([x.mean()], [y.mean()], s=240, c="none", edgecolors="#222222", linewidths=1.6,
               marker="*", zorder=5)
    ax.annotate(f"mean\n({x.mean():.3f}, {y.mean():.3f})", (x.mean(), y.mean()),
                textcoords="offset points", xytext=(10, -28), fontsize=9, color="#222222")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("raw MTGFlow  overall AUROC")
    ax.set_ylabel("Fisher (proposed)  overall AUROC")
    ax.set_title("UODS eval: per-split raw vs Fisher (100 splits, seed-averaged)", fontsize=11.5)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.text(0.03, 0.97, "above diagonal = proposed better", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color="#555555")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig1_scatter_raw_vs_fisher.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig2_forest(d):
    pr = d["paired_raw_vs_fisher"]
    # 위→아래 순서(가독): 집계 subgroup → family. ball 계열은 각 그룹 끝(정직한 약점).
    rows = [
        ("overall", "overall"), ("developing", "developing"), ("faulty", "faulty"),
        ("nonball_developing", "non-ball developing"), ("nonball_faulty", "non-ball faulty"),
        ("nonball", "non-ball (all)"), ("ball", "ball (all)"),
        ("fam_inner", "family: inner"), ("fam_outer", "family: outer"),
        ("fam_cage", "family: cage"), ("fam_ball", "family: ball"),
    ]
    ys = np.arange(len(rows))[::-1]  # 첫 행이 위
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.axvline(0, ls="--", lw=1.4, color="#888888", zorder=1)
    for (key, _), yy in zip(rows, ys):
        r = pr[key]
        m = r["mean_diff"]; lo, hi = r["ci95_boot"]
        weak = "ball" in key
        col = C_WEAK if weak else C_FISHER
        ax.plot([lo, hi], [yy, yy], lw=2.2, color=col, alpha=0.85, zorder=2, solid_capstyle="round")
        ax.scatter([m], [yy], s=70, color=col, edgecolors="white", linewidths=0.6, zorder=3)
        ax.annotate(f"{m:+.3f}   {r['improved']}/100", (hi, yy), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=9, color="#333333")
    ax.set_yticks(ys); ax.set_yticklabels([lbl for _, lbl in rows], fontsize=10)
    ax.set_xlabel("Δ AUROC (Fisher − raw), split-paired, 95% CI (bootstrap)")
    ax.set_title("UODS eval: subgroup improvement over raw (n=100 splits)", fontsize=11.5)
    ax.set_xlim(-0.03, 0.30)
    ax.grid(axis="y", visible=False)
    ax.text(0.98, 0.02, "vermillion = ball (no-load confound, weakest)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=C_WEAK)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig2_forest_subgroup_delta.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig3_family_bars(d):
    fams = ["inner", "outer", "ball", "cage"]
    methods = ["raw", "shape", "amp", "equal_z", "fisher"]
    agg = d["aggregate"]
    x = np.arange(len(fams))
    w = 0.16
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for i, mth in enumerate(methods):
        means = [agg[mth]["subgroup"][f"fam_{f}"]["mean"] for f in fams]
        stds = [agg[mth]["subgroup"][f"fam_{f}"]["std"] for f in fams]
        off = (i - (len(methods) - 1) / 2) * w
        bars = ax.bar(x + off, means, w, yerr=stds, capsize=2.5,
                      color=METHOD_COLOR[mth], label=METHOD_LABEL[mth],
                      edgecolor="white", linewidth=0.6,
                      error_kw=dict(elinewidth=0.9, ecolor="#666666"))
        for b, mn in zip(bars, means):
            ax.annotate(f"{mn:.2f}", (b.get_x() + b.get_width() / 2, mn), textcoords="offset points",
                        xytext=(0, 2), ha="center", va="bottom", fontsize=7.0, color="#333333", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([f"{f}" for f in fams], fontsize=11)
    ax.set_ylabel("AUROC (mean ± std over 100 splits)")
    ax.set_ylim(0.55, 1.02)
    ax.set_title("UODS eval: family × method AUROC", fontsize=11.5)
    ax.legend(loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig3_family_method_bars.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    d = load()
    for fn in (fig1_scatter, fig2_forest, fig3_family_bars):
        p = fn(d)
        print(f"wrote {p}  ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    main()
