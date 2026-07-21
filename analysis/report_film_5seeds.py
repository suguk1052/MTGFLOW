"""FiLM(A안) measured 5시드 실험을 no-meta / static / measured-concat baseline과 비교하는 리포트 생성.

FiLM A안 = concat 대신 C_op에서 작은 MLP로 Δγ,β를 만들어 C_mod=(1+Δγ)⊙C+β로 기존 condition C를
변조(항등 초기화로 시작 시 no-meta와 동일). 이 스크립트는 C1(pooled)과 C2(LOSO 4-split) 양쪽에서
FiLM 5시드 평균 AUROC를 baseline 3종과 나란히 놓고, FiLM이 baseline을 넘는지(특히 no-meta) 판정한다.

집계 관례는 report_b3_vs_c2_5seeds.py와 동일:
- 상세 표의 셀 std = 각 폴더 summary_seeds.json의 auroc_std (5시드 population std, ddof=0)
- 요약 표의 Std = split-level mean 값들의 표본표준편차(ddof=1)

판정 규칙(--log_test_auroc off): FiLM 5시드 평균이 baseline보다 오르면 신뢰, 안 오르면 방법/선택손해
미구분으로 결론 유보. baseline은 재실행 없이 기존 5시드 결과 재사용.
"""
import json
import os
import statistics

from _common import PROJECT_ROOT

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "Paderborn")

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


def load_summary(base_dir, run_name):
    path = os.path.join(RESULTS_ROOT, base_dir, run_name, "summary_seeds.json")
    with open(path) as f:
        data = json.load(f)
    return data["auroc_mean"], data["auroc_std"]


def collect_pooled():
    """C1(pooled) split 0123C의 6 LONO 행 수집."""
    rows = []
    for lono in range(1, 7):
        vals = {
            "no-meta": load_summary("LONO_B2_5seeds", f"raw_vib_0123C_LONO{lono}_5seeds"),
            "static": load_summary("LONO_C1_5seeds", f"CA_raw_vib_0123C_LONO{lono}_5seeds"),
            "measured": load_summary(
                "LONO_C1_measured_5seeds", f"CA_raw_vib_0123C_LONO{lono}_measured_5seeds"
            ),
            "FiLM": load_summary(
                "LONO_FiLM_measured_5seeds", f"CA_raw_vib_0123C_LONO{lono}_FiLM_measured_5seeds"
            ),
        }
        best = max(vals, key=lambda k: vals[k][0])
        rows.append((lono, LONO_TEST_NORM[lono], vals, best))
    return rows


def collect_loso(split):
    rows = []
    for lono in range(1, 7):
        vals = {
            "no-meta": load_summary("LONO_B3_5seeds", f"raw_vib_{split}_LONO{lono}"),
            "static": load_summary("LONO_C2_5seeds", f"CA_raw_vib_{split}_LONO{lono}"),
            "measured": load_summary(
                "LONO_C2_measured_5seeds", f"CA_raw_vib_{split}_LONO{lono}_measured"
            ),
            "FiLM": load_summary(
                "LONO_FiLM_measured_5seeds", f"CA_raw_vib_{split}_LONO{lono}_FiLM_measured_5seeds"
            ),
        }
        best = max(vals, key=lambda k: vals[k][0])
        rows.append((lono, LONO_TEST_NORM[lono], vals, best))
    return rows


def fmt(mean_std):
    mean, std = mean_std
    return f"{mean:.3f} ± {std:.3f}"


def summarize(rows):
    """6 LONO 행의 split-level 요약(FiLM 기준 win-rate 포함)."""
    means = {c: [r[2][c][0] for r in rows] for c in CONDITIONS}
    summary = {}
    for c in CONDITIONS:
        vals = means[c]
        summary[c] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "median": statistics.median(vals),
        }
    win = {
        "vs_nometa": sum(1 for f, b in zip(means["FiLM"], means["no-meta"]) if f > b),
        "vs_static": sum(1 for f, b in zip(means["FiLM"], means["static"]) if f > b),
        "vs_measured": sum(1 for f, b in zip(means["FiLM"], means["measured"]) if f > b),
    }
    return summary, win


def render_detail_table(rows):
    lines = [
        "| LONO split | Test normal | no-meta | static | measured-concat | **FiLM** | best |",
        "|---|---|---|---|---|---|---|",
    ]
    for lono, test_norm, vals, best in rows:
        lines.append(
            f"| LONO-{lono} | {test_norm} | {fmt(vals['no-meta'])} | {fmt(vals['static'])} | "
            f"{fmt(vals['measured'])} | **{fmt(vals['FiLM'])}** | {COND_LABEL[best]} |"
        )
    return lines


def render_summary_table(summary, win):
    lines = [
        "| Model | Mean AUROC | Std AUROC | Median AUROC |",
        "|---|---|---|---|",
    ]
    for c in CONDITIONS:
        s = summary[c]
        lines.append(f"| {COND_LABEL[c]} | {s['mean']:.3f} | {s['std']:.3f} | {s['median']:.3f} |")
    d_nm = summary["FiLM"]["mean"] - summary["no-meta"]["mean"]
    d_ms = summary["FiLM"]["mean"] - summary["measured"]["mean"]
    lines.append(f"| Δ(FiLM−no-meta) | {d_nm:+.3f} | - | - |")
    lines.append(f"| Δ(FiLM−measured-concat) | {d_ms:+.3f} | - | - |")
    lines.append("")
    lines.append(
        f"(FiLM이 이기는 LONO: no-meta 대비 {win['vs_nometa']}/6, static 대비 {win['vs_static']}/6, "
        f"measured-concat 대비 {win['vs_measured']}/6)"
    )
    return lines


def main():
    lines = []
    lines.append("# FiLM(A안) measured 5시드 비교 보고서 (Paderborn)")
    lines.append("")
    lines.append(
        "첫 ours 시도인 **FiLM A안**(condition C 변조, measured-mean 3차원, 항등 초기화)이 "
        "baseline(`no-meta`, `static`, 기존 `measured`=단순 concat) 대비 이상탐지 AUROC를 개선하는지, "
        "C1(pooled, in-distribution) 6 LONO와 C2(LOSO 4 split: 012to3/013to2/023to1/123to0) × LONO(1~6)에 대해 "
        "5시드(2024~2028)로 검증한다. FiLM은 concat 대신 C_op에서 Δγ,β를 만들어 `C_mod=(1+Δγ)⊙C+β`로 기존 "
        "condition C를 변조(MAF 입력 차원 불변)하고, Δγ,β 최종 선형층을 0으로 항등 초기화해 시작 시 no-meta와 "
        "동일(do-no-harm)하도록 설계됐다."
    )
    lines.append("")
    lines.append(
        "- 각 셀의 `mean ± std`는 해당 폴더 `summary_seeds.json`의 5시드 `auroc_mean`/`auroc_std`를 그대로 읽은 "
        "값이다(5시드 population std, ddof=0)."
    )
    lines.append(
        "- \"요약 표\"의 Mean/Std/Median은 6개 LONO의 mean 값들을 다시 집계한 것이며 Std는 표본표준편차(ddof=1)다 "
        "(report_C1_measured_comparison_5seeds.md / report_B3_vs_C2_5seeds.md와 동일 관례)."
    )
    lines.append(
        "- **판정 규칙**(`--log_test_auroc` off): FiLM 5시드 평균이 baseline보다 **오르면 신뢰**(val_loss 기반 "
        "checkpoint 선택 핸디캡을 안고도 이긴 것), **안 오르면** '방법 문제 vs 선택 손해'를 구분할 수 없어 결론 "
        "유보. baseline은 재실행 없이 기존 5시드 결과를 재사용했다."
    )
    lines.append("")

    # --- C1 (pooled) ---
    pooled_rows = collect_pooled()
    pooled_summary, pooled_win = summarize(pooled_rows)
    lines.append("## C1 (pooled, in-distribution)")
    lines.append("")
    lines.append("do-no-harm 확인 구간 — FiLM이 C1을 깨지 않는지(no-meta 수준 유지) 본다.")
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
            loso_all[c].extend(r[2][c][0] for r in rows)
        overview_rows.append((split, desc, summary, win))

    # --- LOSO 전체 개요 ---
    lines.append("## C2(LOSO) 전체 개요 (4 split × 6 LONO = 24 셀)")
    lines.append("")
    lines.append("| LOSO split | 설명 | no-meta | static | measured-concat | FiLM | FiLM win vs no-meta |")
    lines.append("|---|---|---|---|---|---|---|")
    for split, desc, summary, win in overview_rows:
        lines.append(
            f"| {split} | {desc} | {summary['no-meta']['mean']:.3f} | {summary['static']['mean']:.3f} | "
            f"{summary['measured']['mean']:.3f} | {summary['FiLM']['mean']:.3f} | {win['vs_nometa']}/6 |"
        )
    lines.append("")

    overall_mean = {c: statistics.mean(loso_all[c]) for c in CONDITIONS}
    overall_std = {c: statistics.stdev(loso_all[c]) for c in CONDITIONS}
    win_nm = sum(1 for f, b in zip(loso_all["FiLM"], loso_all["no-meta"]) if f > b)
    win_st = sum(1 for f, b in zip(loso_all["FiLM"], loso_all["static"]) if f > b)
    win_ms = sum(1 for f, b in zip(loso_all["FiLM"], loso_all["measured"]) if f > b)
    lines.append("| Model | 전체(24 셀) Mean AUROC | Std AUROC |")
    lines.append("|---|---|---|")
    for c in CONDITIONS:
        lines.append(f"| {COND_LABEL[c]} | {overall_mean[c]:.3f} | {overall_std[c]:.3f} |")
    lines.append(f"| Δ(FiLM−no-meta) | {overall_mean['FiLM']-overall_mean['no-meta']:+.3f} | - |")
    lines.append(f"| Δ(FiLM−measured-concat) | {overall_mean['FiLM']-overall_mean['measured']:+.3f} | - |")
    lines.append("")
    lines.append(
        f"(FiLM이 이기는 LONO×split: no-meta 대비 {win_nm}/24, static 대비 {win_st}/24, "
        f"measured-concat 대비 {win_ms}/24)"
    )
    lines.append("")

    # --- 핵심 관찰 / 판정 ---
    lines.append("## 핵심 관찰 및 판정")
    lines.append("")
    lines.append(
        f"- **C1(pooled)**: FiLM {pooled_summary['FiLM']['mean']:.3f} vs no-meta "
        f"{pooled_summary['no-meta']['mean']:.3f} (Δ{pooled_summary['FiLM']['mean']-pooled_summary['no-meta']['mean']:+.3f}). "
        f"static {pooled_summary['static']['mean']:.3f} / measured-concat {pooled_summary['measured']['mean']:.3f}보다는 "
        "높다. FiLM은 C1을 깨지 않고 **do-no-harm을 대체로 유지**한다(항등 초기화 설계대로)."
    )
    lines.append(
        f"- **C2(LOSO) 전체**: FiLM {overall_mean['FiLM']:.3f} < no-meta {overall_mean['no-meta']:.3f} "
        f"(Δ{overall_mean['FiLM']-overall_mean['no-meta']:+.3f}), FiLM이 no-meta를 이기는 셀은 {win_nm}/24에 그친다. "
        f"단, 단순 concat measured({overall_mean['measured']:.3f})는 근소하게 넘는다"
        f"(Δ{overall_mean['FiLM']-overall_mean['measured']:+.3f}, {win_ms}/24)."
    )
    ms_split = {s: (summ, w) for s, _, summ, w in overview_rows}
    recovered = [
        s for s in ("012to3", "023to1", "013to2")
        if ms_split[s][0]["FiLM"]["mean"] > ms_split[s][0]["measured"]["mean"]
    ]
    lines.append(
        "- **FiLM의 이득은 measured-concat이 무너졌던 split을 부분 회복하는 데 국한**된다"
        f"({', '.join(recovered)}에서 FiLM > measured-concat). 반대로 measured가 이미 강했던 "
        f"123to0(measured {ms_split['123to0'][0]['measured']['mean']:.3f} → FiLM "
        f"{ms_split['123to0'][0]['FiLM']['mean']:.3f})에서는 오히려 내려간다. 즉 FiLM은 concat의 "
        "불안정을 다소 완화할 뿐, no-meta를 넘는 cross-setting 이득을 만들지는 못한다."
    )
    lines.append(
        "- **판정**: FiLM 5시드 평균이 LOSO에서 no-meta baseline을 넘지 못했다 → 판정 규칙상 "
        "**'개선 근거 없음'**(방법 자체 한계인지 val_loss 기반 checkpoint 선택 손해인지는 이 실험만으로 구분 불가). "
        "measured 운행값을 FiLM으로 주입하는 것만으로는 unseen 운행조건 일반화 목표(TODO ⭐§2)를 달성하지 못한다."
    )
    lines.append("")

    out_path = os.path.join(PROJECT_ROOT, "reports", "report_FiLM_measured_5seeds.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
