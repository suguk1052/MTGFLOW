"""B3(no-meta) vs C2(static-meta) vs C2_measured(실측 meta) 5시드 비교 리포트 생성.

results/Paderborn/LONO_B3_5seeds, LONO_C2_5seeds, LONO_C2_measured_5seeds의
summary_seeds.json(5시드 auroc_mean/auroc_std)을 읽어 LOSO split별 상세 표 +
전체 요약 표를 markdown으로 만든다. report_C1_measured_comparison_5seeds.md와
동일한 형식/집계 방식(상세 표 std=시드 간 population std, 요약 표 std=split 간 sample std)을 따른다.
"""
import json
import os
import statistics

from _common import PROJECT_ROOT

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "Paderborn")

SPLITS = [
    ("023to1", "저속(N09_M07_F10) unseen"),
    ("012to3", "저 radial force(N15_M07_F04) unseen"),
    ("013to2", "저토크(N15_M01_F10) unseen"),
    ("123to0", "기준조건(N15_M07_F10) unseen"),
]
LONO_TEST_NORM = {1: "K001", 2: "K002", 3: "K003", 4: "K004", 5: "K005", 6: "K006"}

CONDITIONS = ["no-meta", "static", "measured"]


def load_summary(base_dir, run_name):
    path = os.path.join(RESULTS_ROOT, base_dir, run_name, "summary_seeds.json")
    with open(path) as f:
        data = json.load(f)
    return data["auroc_mean"], data["auroc_std"]


def collect(split):
    rows = []
    for lono in range(1, 7):
        no_meta = load_summary("LONO_B3_5seeds", f"raw_vib_{split}_LONO{lono}")
        static = load_summary("LONO_C2_5seeds", f"CA_raw_vib_{split}_LONO{lono}")
        measured = load_summary("LONO_C2_measured_5seeds", f"CA_raw_vib_{split}_LONO{lono}_measured")
        vals = {"no-meta": no_meta, "static": static, "measured": measured}
        best = max(vals, key=lambda k: vals[k][0])
        rows.append((lono, LONO_TEST_NORM[lono], vals, best))
    return rows


def fmt(mean_std):
    mean, std = mean_std
    return f"{mean:.3f} ± {std:.3f}"


def summarize(rows):
    means = {c: [rows_i[2][c][0] for rows_i in rows] for c in CONDITIONS}
    summary = {}
    for c in CONDITIONS:
        vals = means[c]
        summary[c] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "median": statistics.median(vals),
        }
    improved_static = sum(1 for a, b in zip(means["static"], means["no-meta"]) if a > b)
    improved_measured_vs_nometa = sum(1 for a, b in zip(means["measured"], means["no-meta"]) if a > b)
    improved_measured_vs_static = sum(1 for a, b in zip(means["measured"], means["static"]) if a > b)
    return summary, improved_static, improved_measured_vs_nometa, improved_measured_vs_static


def render_split_section(split, desc, rows):
    lines = [f"### LOSO {split} ({desc})", ""]
    lines.append("| LONO split | Test normal | no-meta (B3) | static (C2) | measured (C2_measured) | best |")
    lines.append("|---|---|---|---|---|---|")
    for lono, test_norm, vals, best in rows:
        lines.append(
            f"| LONO-{lono} | {test_norm} | {fmt(vals['no-meta'])} | {fmt(vals['static'])} | "
            f"{fmt(vals['measured'])} | {best} |"
        )
    lines.append("")

    summary, imp_static, imp_meas_vs_nometa, imp_meas_vs_static = summarize(rows)
    lines.append("| Model | Mean AUROC | Std AUROC | Median AUROC | Improved splits (vs no-meta) |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| no-meta (baseline) | {summary['no-meta']['mean']:.3f} | {summary['no-meta']['std']:.3f} | "
        f"{summary['no-meta']['median']:.3f} | - |"
    )
    lines.append(
        f"| static | {summary['static']['mean']:.3f} | {summary['static']['std']:.3f} | "
        f"{summary['static']['median']:.3f} | {imp_static}/6 |"
    )
    lines.append(
        f"| measured | {summary['measured']['mean']:.3f} | {summary['measured']['std']:.3f} | "
        f"{summary['measured']['median']:.3f} | {imp_meas_vs_nometa}/6 |"
    )
    delta_meas_nometa = summary["measured"]["mean"] - summary["no-meta"]["mean"]
    delta_meas_static = summary["measured"]["mean"] - summary["static"]["mean"]
    lines.append(f"| Δ(measured−no-meta) | {delta_meas_nometa:+.3f} | - | - | - |")
    lines.append(f"| Δ(measured−static) | {delta_meas_static:+.3f} | - | - | - |")
    lines.append("")
    lines.append(f"(measured가 static보다 나은 split: {imp_meas_vs_static}/6)")
    lines.append("")
    return "\n".join(lines), summary, imp_static, imp_meas_vs_nometa, imp_meas_vs_static


def main():
    per_split = {}
    all_rows = {c: [] for c in CONDITIONS}
    sections = []
    overview_rows = []

    for split, desc in SPLITS:
        rows = collect(split)
        per_split[split] = rows
        section_md, summary, imp_static, imp_meas_vs_nometa, imp_meas_vs_static = render_split_section(
            split, desc, rows
        )
        sections.append(section_md)
        for c in CONDITIONS:
            all_rows[c].extend(r[2][c][0] for r in rows)
        overview_rows.append(
            (
                split,
                desc,
                summary["no-meta"]["mean"],
                summary["static"]["mean"],
                summary["measured"]["mean"],
                imp_static,
                imp_meas_vs_nometa,
                imp_meas_vs_static,
            )
        )

    overall_mean = {c: statistics.mean(all_rows[c]) for c in CONDITIONS}
    overall_std = {c: statistics.stdev(all_rows[c]) for c in CONDITIONS}
    total_improved_static = sum(r[5] for r in overview_rows)
    total_improved_meas_nometa = sum(r[6] for r in overview_rows)
    total_improved_meas_static = sum(r[7] for r in overview_rows)

    lines = []
    lines.append("# B3 vs C2 vs C2_measured 5시드 비교 보고서 (Paderborn LOSO)")
    lines.append("")
    lines.append(
        "LOSO(Leave-One-Setting-Out) 실험에서 실측 운행값 메타(`measured`)가 baseline(`no-meta`=B3)과 "
        "기존 static meta(`static`=C2) 대비 이상탐지 AUROC를 개선하는지, 4개 LOSO split(012to3/013to2/023to1/123to0) "
        "전체 × LONO(i=1~6)에 대해 5시드(2024~2028)로 확인한다. 이전에는 023to1(저속 unseen) 1개 split만 "
        "1시드(2026)로 돌려 `measured가 크게 개선된다`는 case-study 관찰이 있었는데, 이번에는 4개 split 전체를 "
        "B2 vs C1과 동일한 방식(5-seed)으로 재검증한다."
    )
    lines.append("")
    lines.append(
        "- 각 LONO split 셀의 `mean ± std`는 해당 폴더의 `summary_seeds.json`에 기록된 5시드 "
        "`auroc_mean`/`auroc_std`를 그대로 읽은 값이다(5개 시드에 대한 population std, ddof=0)."
    )
    lines.append(
        "- \"요약 표\"의 Mean/Std/Median은 6개 LONO split의 split-level mean 값들을 다시 집계한 것이며, "
        "Std는 표본표준편차(ddof=1)다(상세 표의 std와 대상이 다름 — report_C1_measured_comparison_5seeds.md와 동일 관례)."
    )
    lines.append("- **no-meta**: baseline (=B3, LOSO no-meta)")
    lines.append("- **static**: 기존 static meta (C2)")
    lines.append("- **measured**: 신규 실측 meta (C2_measured)")
    lines.append("")
    lines.append("## Split별 상세")
    lines.append("")
    lines.extend(sections)

    lines.append("## 전체 개요 (4 split 통합)")
    lines.append("")
    lines.append("| LOSO split | 설명 | no-meta mean | static mean | measured mean | static 개선 | measured vs no-meta 개선 | measured vs static 개선 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for split, desc, nm, st, ms, imp_st, imp_mn, imp_ms in overview_rows:
        lines.append(
            f"| {split} | {desc} | {nm:.3f} | {st:.3f} | {ms:.3f} | {imp_st}/6 | {imp_mn}/6 | {imp_ms}/6 |"
        )
    lines.append("")
    lines.append("| Model | 전체(24 LONO×split) Mean AUROC | Std AUROC | Improved LONO×split (vs no-meta) |")
    lines.append("|---|---|---|---|")
    lines.append(f"| no-meta (baseline) | {overall_mean['no-meta']:.3f} | {overall_std['no-meta']:.3f} | - |")
    lines.append(f"| static | {overall_mean['static']:.3f} | {overall_std['static']:.3f} | {total_improved_static}/24 |")
    lines.append(f"| measured | {overall_mean['measured']:.3f} | {overall_std['measured']:.3f} | {total_improved_meas_nometa}/24 |")
    lines.append(f"| Δ(measured−no-meta) | {overall_mean['measured']-overall_mean['no-meta']:+.3f} | - | - |")
    lines.append(f"| Δ(measured−static) | {overall_mean['measured']-overall_mean['static']:+.3f} | - | - |")
    lines.append("")
    lines.append(f"(measured가 static보다 나은 LONO×split: {total_improved_meas_static}/24)")
    lines.append("")

    lines.append("## 핵심 관찰")
    lines.append("")
    loso1_row = [r for r in overview_rows if r[0] == "023to1"][0]
    lines.append(
        f"- **023to1(저속 unseen)**: measured mean {loso1_row[4]:.3f} vs no-meta {loso1_row[2]:.3f}, "
        f"static {loso1_row[3]:.3f} — measured가 no-meta를 이기는 LONO는 {loso1_row[6]}/6, static을 "
        f"이기는 LONO는 {loso1_row[7]}/6. 5-seed로 재확인한 결과이므로 기존 1-seed case-study 관찰과 "
        "직접 비교해 판단할 것."
    )
    lines.append(
        f"- 4개 split 전체(24 LONO×split)에서 measured가 no-meta를 이기는 비율은 "
        f"{total_improved_meas_nometa}/24, static을 이기는 비율은 {total_improved_meas_static}/24."
    )
    lines.append(
        "- B2 vs C1 5-seed 재검증에서는 1-seed 때 보였던 개선이 5-seed 평균에서 사라졌다(3/6 split만 개선). "
        "이번 B3 vs C2/C2_measured 결과가 같은 패턴을 보이는지, 아니면 LOSO 환경에서는 measured 메타가 "
        "실제로 강건한 이득을 주는지 위 수치로 판단한다."
    )
    lines.append("")

    out_path = os.path.join(PROJECT_ROOT, "reports", "report_B3_vs_C2_5seeds.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
