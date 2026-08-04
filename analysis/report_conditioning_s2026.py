"""MTGFLOW conditioning 4종(no-meta / static / measured-concat / FiLM) 비교표 — seed 2026 단독.

기존 report_*_5seeds.py는 5시드(2024~2028) 평균 ±std를 낸다. 이 스크립트는 그 표와 **같은 축**
(조건 × C1 pooled 6 LONO + C2 LOSO 4split×6=24셀)이되, **seed 2026 한 시드의 AUROC만** 뽑는다.
TODO ⭐"다음 무대"가 단일시드(2026)에서 C2(LOSO) 개선을 목표로 잡고 있어, 그 출발 기준선이 되는 표다.

값 출처: 각 조건 폴더의 `<run_name>_s2026/paderborn_per_bearing_metrics.json`의 `overall_auroc`.
(주의: base 폴더의 summary_seeds.json은 일부가 seeds=[2024]만 담겨 신뢰 불가 → 안 쓰고 raw를 직접 읽음.)
"""
import json
import os
import statistics

from _common import PROJECT_ROOT

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "Paderborn")

SEED = 2026

LOSO_SPLITS = [
    ("023to1", "저속(N09_M07_F10) unseen"),
    ("012to3", "저 radial force(N15_M07_F04) unseen"),
    ("013to2", "저토크(N15_M01_F10) unseen"),
    ("123to0", "기준조건(N15_M07_F10) unseen"),
]
LONO_TEST_NORM = {1: "K001", 2: "K002", 3: "K003", 4: "K004", 5: "K005", 6: "K006"}

CONDITIONS = ["no-meta", "static", "measured", "FiLM"]
COND_LABEL = {
    "no-meta": "no-meta",
    "static": "static",
    "measured": "measured-concat",
    "FiLM": "FiLM",
}


def load_auroc_s2026(base_dir, run_name):
    """<base_dir>/<run_name>_s2026/paderborn_per_bearing_metrics.json 의 overall_auroc(단일 시드)."""
    path = os.path.join(
        RESULTS_ROOT, base_dir, f"{run_name}_s{SEED}", "paderborn_per_bearing_metrics.json"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"seed {SEED} 결과 없음: {path}")
    with open(path) as f:
        data = json.load(f)
    return data["overall_auroc"]


def collect_pooled():
    """C1(pooled) split 0123C의 6 LONO 행 수집 (seed 2026)."""
    rows = []
    for lono in range(1, 7):
        vals = {
            "no-meta": load_auroc_s2026("LONO_B2_5seeds", f"raw_vib_0123C_LONO{lono}_5seeds"),
            "static": load_auroc_s2026("LONO_C1_5seeds", f"CA_raw_vib_0123C_LONO{lono}_5seeds"),
            "measured": load_auroc_s2026(
                "LONO_C1_measured_5seeds", f"CA_raw_vib_0123C_LONO{lono}_measured_5seeds"
            ),
            "FiLM": load_auroc_s2026(
                "LONO_FiLM_measured_5seeds", f"CA_raw_vib_0123C_LONO{lono}_FiLM_measured_5seeds"
            ),
        }
        best = max(vals, key=lambda k: vals[k])
        rows.append((lono, LONO_TEST_NORM[lono], vals, best))
    return rows


def collect_loso(split):
    rows = []
    for lono in range(1, 7):
        vals = {
            "no-meta": load_auroc_s2026("LONO_B3_5seeds", f"raw_vib_{split}_LONO{lono}"),
            "static": load_auroc_s2026("LONO_C2_5seeds", f"CA_raw_vib_{split}_LONO{lono}"),
            "measured": load_auroc_s2026(
                "LONO_C2_measured_5seeds", f"CA_raw_vib_{split}_LONO{lono}_measured"
            ),
            "FiLM": load_auroc_s2026(
                "LONO_FiLM_measured_5seeds", f"CA_raw_vib_{split}_LONO{lono}_FiLM_measured_5seeds"
            ),
        }
        best = max(vals, key=lambda k: vals[k])
        rows.append((lono, LONO_TEST_NORM[lono], vals, best))
    return rows


def fmt(auroc):
    return f"{auroc:.3f}"


def summarize(rows):
    """6 LONO 행의 조건별 mean/median + FiLM 기준 win-rate."""
    means = {c: [r[2][c] for r in rows] for c in CONDITIONS}
    summary = {}
    for c in CONDITIONS:
        vals = means[c]
        summary[c] = {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
        }
    win = {
        "vs_nometa": sum(1 for f, b in zip(means["FiLM"], means["no-meta"]) if f > b),
        "vs_static": sum(1 for f, b in zip(means["FiLM"], means["static"]) if f > b),
        "vs_measured": sum(1 for f, b in zip(means["FiLM"], means["measured"]) if f > b),
    }
    return summary, win


def bold_if(text, cond):
    """cond가 True면 마크다운 볼드로 감싼다 (행별 최고값 강조용)."""
    return f"**{text}**" if cond else text


def render_detail_table(rows):
    lines = [
        "| LONO split | Test normal | no-meta | static | measured-concat | FiLM | best |",
        "|---|---|---|---|---|---|---|",
    ]
    for lono, test_norm, vals, best in rows:
        # best = 4종 중 최고 AUROC 조건 → 그 셀만 볼드 (비교 편의)
        cells = " | ".join(bold_if(fmt(vals[c]), c == best) for c in CONDITIONS)
        lines.append(f"| LONO-{lono} | {test_norm} | {cells} | {COND_LABEL[best]} |")
    return lines


def render_summary_table(summary, win):
    lines = [
        "| Model | Mean AUROC | Median AUROC |",
        "|---|---|---|",
    ]
    best_mean = max(CONDITIONS, key=lambda c: summary[c]["mean"])  # 4종 중 최고 mean 조건
    for c in CONDITIONS:
        s = summary[c]
        mean_cell = bold_if(f"{s['mean']:.3f}", c == best_mean)
        lines.append(f"| {COND_LABEL[c]} | {mean_cell} | {s['median']:.3f} |")
    d_nm = summary["FiLM"]["mean"] - summary["no-meta"]["mean"]
    d_ms = summary["FiLM"]["mean"] - summary["measured"]["mean"]
    lines.append(f"| Δ(FiLM−no-meta) | {d_nm:+.3f} | - |")
    lines.append(f"| Δ(FiLM−measured-concat) | {d_ms:+.3f} | - |")
    lines.append("")
    lines.append(
        f"(FiLM이 이기는 LONO: no-meta 대비 {win['vs_nometa']}/6, static 대비 {win['vs_static']}/6, "
        f"measured-concat 대비 {win['vs_measured']}/6)"
    )
    return lines


def main():
    lines = []
    lines.append("# MTGFLOW conditioning 비교표 — seed 2026 단독 (Paderborn)")
    lines.append("")
    lines.append(
        "no-meta / static / measured-concat / FiLM 4종 conditioning의 window-level AUROC를 "
        "C1(pooled, in-distribution) 6 LONO와 C2(LOSO 4 split: 012to3/013to2/023to1/123to0) × LONO(1~6)에서 "
        "**seed 2026 한 시드**로 비교한다. 기존 5시드 리포트(report_*_5seeds.md)와 축은 같고 값만 단일 시드다."
    )
    lines.append("")
    lines.append(
        f"- 각 셀 값 = 해당 run의 `_s{SEED}/paderborn_per_bearing_metrics.json`의 `overall_auroc` "
        "(seed 2026 단일값, ±std 없음)."
    )
    lines.append(
        "- \"요약 표\"의 Mean/Median은 6개 LONO의 값들을 집계한 것이다."
    )
    lines.append(
        "- 이 표는 TODO ⭐\"다음 무대\"(단일시드 2026에서 C2 개선)의 출발 기준선이다."
    )
    lines.append("")

    # --- C1 (pooled) ---
    pooled_rows = collect_pooled()
    pooled_summary, pooled_win = summarize(pooled_rows)
    lines.append("## C1 (pooled, in-distribution)")
    lines.append("")
    lines.extend(render_detail_table(pooled_rows))
    lines.append("")
    lines.extend(render_summary_table(pooled_summary, pooled_win))
    lines.append("")

    # --- C2 (LOSO) split별 상세 ---
    lines.append("## C2 (LOSO, cross-setting) — split별 상세")
    lines.append("")
    loso_all = {c: [] for c in CONDITIONS}
    overview_rows = []
    for split, desc in LOSO_SPLITS:
        rows = collect_loso(split)
        summary, win = summarize(rows)
        lines.append(f"### LOSO {split} ({desc})")
        lines.append("")
        lines.extend(render_detail_table(rows))
        lines.append("")
        lines.extend(render_summary_table(summary, win))
        lines.append("")
        for c in CONDITIONS:
            loso_all[c].extend(r[2][c] for r in rows)
        overview_rows.append((split, desc, summary, win))

    # --- LOSO 전체 개요 ---
    lines.append("## C2(LOSO) 전체 개요 (4 split × 6 LONO = 24 셀)")
    lines.append("")
    lines.append("| LOSO split | 설명 | no-meta | static | measured-concat | FiLM | FiLM win vs no-meta |")
    lines.append("|---|---|---|---|---|---|---|")
    for split, desc, summary, win in overview_rows:
        best_c = max(CONDITIONS, key=lambda c: summary[c]["mean"])  # 행별 최고 mean 조건
        cells = " | ".join(bold_if(f"{summary[c]['mean']:.3f}", c == best_c) for c in CONDITIONS)
        lines.append(f"| {split} | {desc} | {cells} | {win['vs_nometa']}/6 |")
    lines.append("")

    overall_mean = {c: statistics.mean(loso_all[c]) for c in CONDITIONS}
    win_nm = sum(1 for f, b in zip(loso_all["FiLM"], loso_all["no-meta"]) if f > b)
    win_st = sum(1 for f, b in zip(loso_all["FiLM"], loso_all["static"]) if f > b)
    win_ms = sum(1 for f, b in zip(loso_all["FiLM"], loso_all["measured"]) if f > b)
    lines.append("| Model | 전체(24 셀) Mean AUROC |")
    lines.append("|---|---|")
    best_overall = max(CONDITIONS, key=lambda c: overall_mean[c])  # 24셀 최고 mean 조건
    for c in CONDITIONS:
        lines.append(f"| {COND_LABEL[c]} | {bold_if(f'{overall_mean[c]:.3f}', c == best_overall)} |")
    lines.append(f"| Δ(FiLM−no-meta) | {overall_mean['FiLM']-overall_mean['no-meta']:+.3f} |")
    lines.append(f"| Δ(FiLM−measured-concat) | {overall_mean['FiLM']-overall_mean['measured']:+.3f} |")
    lines.append("")
    lines.append(
        f"(FiLM이 이기는 LONO×split: no-meta 대비 {win_nm}/24, static 대비 {win_st}/24, "
        f"measured-concat 대비 {win_ms}/24)"
    )
    lines.append("")

    out_path = os.path.join(PROJECT_ROOT, "reports", "report_conditioning_s2026.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
