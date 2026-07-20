"""§1′ 진단 — meta가 왜 LOSO를 망치나: OOD-meta 외삽 가설 검증 (새 학습 無, GPU 無).

가설(미확정): LOSO held-out 세팅의 운행값은 train 분포 밖(OOD)이라 train 기준 z-score가
극단값이 되고, 주입기(FiLM Δγ,β MLP / concat 임베딩)가 못 본 극단 입력을 '외삽'하며
예측 불가한 변조를 뱉는다 → (a) 평균 하락 + (b) 시드 요동을 동시에 유발.

검증: split별 held-out 운행값의 OOD 정도( |z-score| / coverage ) ↔ 주입 방식의 성능
하락폭(no-meta − 주입) 및 seed-std 의 방향 일관성을, concat·FiLM **두 주입 방식에서 함께**
확인한다. 같은 OOD 신호가 두 방식에서 동시에 나오면 '특정 주입 버그'가 아니라
'meta-OOD 공통 원인'이라는 증거가 된다.

⚠️ 판정 원칙(n 반영): OOD는 held-out 세팅으로 결정 → **split 단위로 사실상 상수**다.
24셀이지만 유효 OOD 수준은 4덩어리(4 split)뿐이므로, 1차 판정은 24셀 상관계수가 아니라
**split 단위 방향 일관성**(OOD 큰 split에서 하락폭·seed-std가 OOD 작은 split보다 체계적으로
큰가)으로 한다. 상관계수는 보조 숫자로만 기재한다. n=4라 인과 단정 금지(정황 증거).

재사용(재구현 금지):
- OOD 지표: results/Paderborn/diag_measured_gate/zscore_stats.json
  (diagnose_measured_zscore.py 산출물)
- 성능(AUROC mean/seed-std): report_film_5seeds.py와 동일한 run_name 규칙으로 로드
  no-meta=LONO_B3_5seeds, FiLM=LONO_FiLM_measured_5seeds, concat=LONO_C2_measured_5seeds

산출물:
- results/Paderborn/diag_ood_meta_film/ood_vs_perf.json (24셀 조인 테이블 + 요약)
- reports/report_ood_meta_film.md
"""
import json
import os

import numpy as np

from _common import PROJECT_ROOT

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "Paderborn")
ZSCORE_PATH = os.path.join(RESULTS_ROOT, "diag_measured_gate", "zscore_stats.json")
OUT_DIR = os.path.join(RESULTS_ROOT, "diag_ood_meta_film")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "report_ood_meta_film.md")

# LOSO split ↔ 설명 (report_film_5seeds.py와 동일). held-out으로 빠지는 물리축을 함께 기록.
LOSO_SPLITS = [
    ("023to1", "저속(N09) unseen", "speed_mean"),
    ("013to2", "저토크(M01) unseen", "torque_mean"),
    ("012to3", "저 radial force(F04) unseen", "force_mean"),
    ("123to0", "기준조건(N15_M07_F10) unseen", None),  # held-out이 train 분포 내부 → OOD 아님
]
LONOS = list(range(1, 7))

# FiLM measured가 실제 주입하는 3개 mean 차원. OOD는 '주입되는 입력' 기준으로 계산한다.
# (held-out 세팅이 이동시키는 물리량이 이 mean 축들이며, concat에서도 지배적 OOD 성분이다.)
MEAN_DIMS = ["speed_mean", "torque_mean", "force_mean"]

# no-meta baseline이 이 값 이하로 chance(0.5)에 붙으면 '주입으로 인한 하락'을 측정할 수 없는
# 교란 셀로 본다(TODO §5의 023to1 threshold collapse). drop 기반 판정에서 제외하되 표엔 남긴다.
NOMETA_CHANCE_MAX = 0.55


def load_summary(base_dir, run_name):
    """report_film_5seeds.py의 load_summary와 동일 규칙 — (auroc_mean, auroc_std) 반환."""
    path = os.path.join(RESULTS_ROOT, base_dir, run_name, "summary_seeds.json")
    with open(path) as f:
        data = json.load(f)
    return data["auroc_mean"], data["auroc_std"]


def load_zscore_index():
    """zscore_stats.json을 (loso_split, lono_idx) → 엔트리 dict 로 인덱싱."""
    with open(ZSCORE_PATH) as f:
        entries = json.load(f)
    idx = {}
    for e in entries:
        idx[(e["loso_split"], e["lono_idx"])] = e
    return idx


def ood_scalars(entry):
    """3개 mean 차원 기준 OOD 스칼라 3종.

    - ood_maxz : max |z_abs_mean|  (held-out 축이 지배 → 대표 지표)
    - ood_meanz: mean |z_abs_mean|
    - ood_cov  : mean coverage(%) of |z|>2  (0~100)
    """
    zmean = [entry["z_abs_mean"][d] for d in MEAN_DIMS]
    cov = [entry["coverage_pct_z_gt_2.0"][d] for d in MEAN_DIMS]
    return {
        "ood_maxz": float(max(zmean)),
        "ood_meanz": float(sum(zmean) / len(zmean)),
        "ood_cov": float(sum(cov) / len(cov)),
    }


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    """순위 상관 = 순위에 대한 Pearson. scale 폭발(023to1 등)에 강건."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)


def build_cells(zidx):
    """24셀 조인 테이블 구성."""
    cells = []
    for split, desc, held_axis in LOSO_SPLITS:
        for lono in LONOS:
            entry = zidx[(split, lono)]
            ood = ood_scalars(entry)
            nm_mean, nm_std = load_summary("LONO_B3_5seeds", f"raw_vib_{split}_LONO{lono}")
            fm_mean, fm_std = load_summary(
                "LONO_FiLM_measured_5seeds",
                f"CA_raw_vib_{split}_LONO{lono}_FiLM_measured_5seeds",
            )
            cc_mean, cc_std = load_summary(
                "LONO_C2_measured_5seeds", f"CA_raw_vib_{split}_LONO{lono}_measured"
            )
            cells.append(
                {
                    "loso_split": split,
                    "desc": desc,
                    "held_axis": held_axis,
                    "lono_idx": lono,
                    **ood,
                    "auroc_nometa": nm_mean,
                    "auroc_film": fm_mean,
                    "auroc_concat": cc_mean,
                    "seedstd_nometa": nm_std,
                    "seedstd_film": fm_std,
                    "seedstd_concat": cc_std,
                    # 하락폭: no-meta 대비 주입이 얼마나 깎였나 (양수 = 주입이 해로움)
                    "drop_film": nm_mean - fm_mean,
                    "drop_concat": nm_mean - cc_mean,
                }
            )
    return cells


def agg_by_split(cells):
    """split 단위 평균 — 1차 판정의 기준(유효 n=4)."""
    out = []
    for split, desc, held_axis in LOSO_SPLITS:
        rows = [c for c in cells if c["loso_split"] == split]
        def m(k):
            return float(sum(r[k] for r in rows) / len(rows))
        out.append(
            {
                "loso_split": split,
                "desc": desc,
                "held_axis": held_axis,
                "ood_maxz": m("ood_maxz"),
                "ood_meanz": m("ood_meanz"),
                "ood_cov": m("ood_cov"),
                "drop_film": m("drop_film"),
                "drop_concat": m("drop_concat"),
                "seedstd_film": m("seedstd_film"),
                "seedstd_concat": m("seedstd_concat"),
                "auroc_nometa": m("auroc_nometa"),
                "auroc_film": m("auroc_film"),
                "auroc_concat": m("auroc_concat"),
                # baseline이 chance 이하로 붕괴 → drop 측정 불가(교란). 표엔 남기고 drop 판정선 제외.
                "nometa_degenerate": m("auroc_nometa") < NOMETA_CHANCE_MAX,
            }
        )
    # OOD(maxz) 내림차순 정렬 → 방향 일관성을 눈으로 확인하기 쉽게
    out.sort(key=lambda r: r["ood_maxz"], reverse=True)
    return out


def correlations(cells, split_rows):
    """보조 상관계수 — cell-level(24, 실질 4덩어리) + split-level(4)."""
    def corrset(rows, ood_key):
        ood = [r[ood_key] for r in rows]
        return {
            "drop_film": {"pearson": pearson(ood, [r["drop_film"] for r in rows]),
                          "spearman": spearman(ood, [r["drop_film"] for r in rows])},
            "drop_concat": {"pearson": pearson(ood, [r["drop_concat"] for r in rows]),
                            "spearman": spearman(ood, [r["drop_concat"] for r in rows])},
            "seedstd_film": {"pearson": pearson(ood, [r["seedstd_film"] for r in rows]),
                             "spearman": spearman(ood, [r["seedstd_film"] for r in rows])},
            "seedstd_concat": {"pearson": pearson(ood, [r["seedstd_concat"] for r in rows]),
                               "spearman": spearman(ood, [r["seedstd_concat"] for r in rows])},
        }
    return {
        "cell_level_n24_maxz": corrset(cells, "ood_maxz"),
        "cell_level_n24_cov": corrset(cells, "ood_cov"),
        "split_level_n4_maxz": corrset(split_rows, "ood_maxz"),
        "split_level_n4_cov": corrset(split_rows, "ood_cov"),
    }


def direction_verdict(split_rows):
    """1차 판정 = split 단위 방향 일관성.

    OOD 큰 split(held-out 축 폭발)의 하락폭·seed-std가 OOD 작은 기준 split(123to0)보다
    체계적으로 큰가를, concat·FiLM 양쪽에서 확인. 두 방식 모두에서 성립해야 '공통 원인' 지지.

    ⚠️ drop 판정에서는 baseline이 chance 이하로 붕괴한 교란 셀(nometa_degenerate)을 제외한다
    (no-meta가 이미 망가진 곳은 '주입으로 인한 하락'을 측정할 수 없어 OOD 크기와 무관하게 하락폭이
    작게 나옴 → 단조성을 깨뜨리는 착시). seed-std는 baseline 수준과 무관하므로 전부 사용한다.
    """
    ref = next(r for r in split_rows if r["loso_split"] == "123to0")  # OOD 최소(기준조건)
    ood_splits = [r for r in split_rows if r["loso_split"] != "123to0"]
    # drop 판정용: 교란 셀 제외한 '해석 가능한' OOD split
    ood_interp = [r for r in ood_splits if not r["nometa_degenerate"]]
    ood_degen = [r for r in ood_splits if r["nometa_degenerate"]]

    def mean(rows, k):
        return sum(r[k] for r in rows) / len(rows) if rows else float("nan")

    checks = {}
    for method in ("film", "concat"):
        dk, sk = f"drop_{method}", f"seedstd_{method}"
        checks[method] = {
            # drop 판정: 해석 가능한 OOD split만 사용
            "drop_ood_interp_mean": mean(ood_interp, dk),
            "drop_ref_split": ref[dk],
            "drop_larger_in_ood": mean(ood_interp, dk) > ref[dk] if ood_interp else False,
            "drop_each_gt_ref": sum(1 for r in ood_interp if r[dk] > ref[dk]),
            "drop_n_interp": len(ood_interp),
            # seed-std 판정: baseline 수준 무관 → 전체 OOD split 사용
            "seedstd_ood_splits_mean": mean(ood_splits, sk),
            "seedstd_ref_split": ref[sk],
            "seedstd_larger_in_ood": mean(ood_splits, sk) > ref[sk],
            "seedstd_each_gt_ref": sum(1 for r in ood_splits if r[sk] > ref[sk]),
        }
    both_drop = checks["film"]["drop_larger_in_ood"] and checks["concat"]["drop_larger_in_ood"]
    both_std = checks["film"]["seedstd_larger_in_ood"] and checks["concat"]["seedstd_larger_in_ood"]
    supported = both_drop and both_std
    return {
        "checks": checks,
        "degenerate_splits": [r["loso_split"] for r in ood_degen],
        "both_methods_drop_larger_in_ood": both_drop,
        "both_methods_seedstd_larger_in_ood": both_std,
        "hypothesis_supported": supported,
    }


def fmt(v, p=3):
    return f"{v:.{p}f}"


def build_report(cells, split_rows, corr, verdict):
    L = []
    L.append("# §1′ 진단 보고서 — OOD-meta 외삽 가설 (Paderborn, 새 학습 無)")
    L.append("")
    L.append(
        "concat·FiLM 두 주입 방식이 연속으로 LOSO(cross-setting)에서 no-meta를 넘지 못했다. "
        "이 실패가 **meta가 학습분포 밖(OOD)으로 나가며 외삽하기 때문**인지 검증한다. "
        "새 학습 없이 기존 산출물만 재사용: OOD 지표는 `diag_measured_gate/zscore_stats.json`, "
        "성능은 `LONO_B3_5seeds`(no-meta)·`LONO_FiLM_measured_5seeds`(FiLM)·"
        "`LONO_C2_measured_5seeds`(measured-concat)의 5시드 집계."
    )
    L.append("")
    L.append("**OOD 정의**: FiLM measured가 실제 주입하는 3개 mean 차원"
             "(speed_mean/torque_mean/force_mean) 기준. `ood_maxz`=max |z|, `ood_cov`=mean coverage(|z|>2, %). "
             "held-out 세팅이 이동시키는 물리량이 이 축들이며 concat에서도 지배적 OOD 성분이다.")
    L.append("")
    L.append("**하락폭 정의**: `drop = AUROC(no-meta) − AUROC(주입)`. 양수 = 주입이 해로움.")
    L.append("")
    L.append(
        "> ⚠️ **판정 원칙(n 반영)**: OOD는 held-out 세팅으로 결정되어 **split 단위로 사실상 상수**다. "
        "24셀이지만 유효 OOD 수준은 **4덩어리(4 split)뿐**이므로, 1차 판정은 24셀 상관계수가 아니라 "
        "**split 단위 방향 일관성**으로 한다. 24셀 상관계수는 통계적 힘을 부풀리지 않도록 보조 숫자로만 본다. "
        "n=4라 인과가 아니라 정황 증거다."
    )
    L.append("")

    # --- 1차 판정: split 단위 (OOD 내림차순) ---
    L.append("## 1) split 단위 (1차 판정 기준, OOD 내림차순, 유효 n=4)")
    L.append("")
    L.append("| LOSO | held-out 축 | ood_maxz | ood_cov(%) | no-meta | FiLM | concat | drop_FiLM | drop_concat | seedstd_FiLM | seedstd_concat | 비고 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in split_rows:
        axis = r["held_axis"] or "(OOD 아님)"
        note = "⚠️baseline≤chance(교란)" if r["nometa_degenerate"] else ""
        L.append(
            f"| {r['loso_split']} | {axis} | {fmt(r['ood_maxz'],2)} | {fmt(r['ood_cov'],1)} | "
            f"{fmt(r['auroc_nometa'])} | {fmt(r['auroc_film'])} | {fmt(r['auroc_concat'])} | "
            f"{fmt(r['drop_film'])} | {fmt(r['drop_concat'])} | "
            f"{fmt(r['seedstd_film'])} | {fmt(r['seedstd_concat'])} | {note} |"
        )
    L.append("")
    if verdict["degenerate_splits"]:
        L.append(
            f"> **교란 셀 처리**: {', '.join(verdict['degenerate_splits'])}은 no-meta AUROC가 "
            f"chance(0.5)~{NOMETA_CHANCE_MAX} 이하로 baseline 자체가 붕괴(TODO §5 threshold collapse). "
            "no-meta가 이미 망가진 곳은 '주입으로 인한 하락'을 잴 수 없어 OOD가 아무리 커도 하락폭이 작게 "
            "나온다(단조성을 깨는 착시). → **drop 기반 판정에서 제외**하고, baseline 수준과 무관한 "
            "seed-std 판정에는 포함한다."
        )
        L.append("")

    # --- 방향 일관성 판정 ---
    v = verdict
    fc, cc = v["checks"]["film"], v["checks"]["concat"]
    n_interp = fc["drop_n_interp"]
    L.append("### 방향 일관성 체크 (OOD 큰 split vs 기준 split 123to0)")
    L.append("")
    L.append(f"drop은 교란 셀 제외 후 **해석 가능한 OOD split {n_interp}개** 기준, seed-std는 전체 OOD 3개 기준.")
    L.append("")
    L.append("| 방식 | 지표 | OOD-split 평균 | 기준(123to0) | OOD에서 더 큼? | 개별 초과 |")
    L.append("|---|---|---|---|---|---|")
    for name, ck in (("FiLM", fc), ("concat", cc)):
        L.append(
            f"| {name} | drop (교란 제외 {n_interp}개) | {fmt(ck['drop_ood_interp_mean'])} | {fmt(ck['drop_ref_split'])} | "
            f"{'예' if ck['drop_larger_in_ood'] else '아니오'} | {ck['drop_each_gt_ref']}/{n_interp} |"
        )
        L.append(
            f"| {name} | seed-std (전체 3개) | {fmt(ck['seedstd_ood_splits_mean'])} | {fmt(ck['seedstd_ref_split'])} | "
            f"{'예' if ck['seedstd_larger_in_ood'] else '아니오'} | {ck['seedstd_each_gt_ref']}/3 |"
        )
    L.append("")

    # --- 최종 판정 ---
    L.append("## 2) 판정")
    L.append("")
    if v["hypothesis_supported"]:
        L.append(
            "- **가설 지지(정황)**: baseline이 건강한 OOD split에서 하락폭이 크고(주입 시 ~chance로 붕괴), "
            "seed-std도 기준 split보다 크다. 이 방향이 **concat·FiLM 두 주입 방식 모두에서** 함께 나타나므로 "
            "특정 주입 버그가 아니라 **meta-OOD 공통 원인**이라는 정황 증거다."
        )
        L.append(
            "- **단, OOD 크기와 하락폭은 단조롭지 않다**: OOD가 가장 극단인 023to1(speed z≈수백)은 오히려 "
            "하락폭이 최소인데, 이는 no-meta baseline이 이미 chance 이하(교란 셀)라 잴 게 없기 때문이지 "
            "'극단 OOD는 안전'하다는 뜻이 아니다. seed-std는 023to1에서도 크다(주입이 불안정을 키움). "
            "즉 '주입 가능한(baseline이 성립하는) 영역에서 OOD가 클수록 해롭다'로 읽어야 한다."
        )
        L.append(
            "- **제언(확정 아님)**: OOD-meta 외삽이 LOSO 실패의 **유력 원인** → §2에서는 무조건 주입"
            "(concat/FiLM/B안)보다 **OOD-감쇠형(gating 등, OOD면 meta를 끄는) 방향을 우선 검토**한다. "
            "다만 gating이 실제로 unseen 이득을 내는지는 학습이 필요하므로 이 진단만으로 확정하지 않는다."
        )
    else:
        L.append(
            "- **가설 반박/불명확**: OOD 큰 split에서 하락폭·seed-std가 체계적으로 크지 않거나, "
            "concat·FiLM 두 방식에서 방향이 일치하지 않는다 → LOSO 실패가 meta-OOD 외삽만으로는 설명되지 않는다."
        )
        L.append(
            "- **제언**: §0 checkpoint-selection gap 등 **대안 원인을 재검토**한다. "
            "gating 우선 근거가 약해지므로 §2 방향을 다시 정한다."
        )
    L.append("")

    # --- 보조: 셀 단위 표 ---
    L.append("## 3) (보조) 24셀 상세")
    L.append("")
    L.append("| LOSO | LONO | ood_maxz | ood_cov(%) | no-meta | FiLM | concat | drop_FiLM | drop_concat | seedstd_FiLM |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        L.append(
            f"| {c['loso_split']} | {c['lono_idx']} | {fmt(c['ood_maxz'],2)} | {fmt(c['ood_cov'],1)} | "
            f"{fmt(c['auroc_nometa'])} | {fmt(c['auroc_film'])} | {fmt(c['auroc_concat'])} | "
            f"{fmt(c['drop_film'])} | {fmt(c['drop_concat'])} | {fmt(c['seedstd_film'])} |"
        )
    L.append("")

    # --- 보조: 상관계수 ---
    L.append("## 4) (보조) 상관계수 — 통계적 힘 과대해석 금지")
    L.append("")
    L.append(
        "24셀 상관은 OOD가 split 단위 상수라 **실질 4덩어리**임을 유의(같은 OOD값이 6번 반복 → n을 부풀림). "
        "scale 폭발(023to1의 speed z ≈ 수백)이 있어 **Spearman(순위)** 이 Pearson보다 신뢰도 높다."
    )
    L.append("")
    L.append(
        "> **maxz vs cov 읽는 법**: `maxz` 기반 drop 상관이 음수로 나오는 건 023to1(극단 z + 붕괴 baseline, "
        "교란 셀)이 상관을 뒤집기 때문이다(가설 반증 아님). scale에 강건한 `cov`(0~100) 기반은 4split에서 "
        "drop·seed-std 모두 양의 상관(concat drop Spearman 1.00, seed-std 1.00)으로, 위 방향 일관성 판정과 일치한다. "
        "단 cov는 세 OOD split 모두 33.3%(1/3 축 OOD)·기준 0%로 사실상 'OOD 축 유무'의 이분값이라 상관값의 "
        "정밀도에 의미를 두지 말 것."
    )
    L.append("")
    L.append("| 스코프 | OOD | vs | Pearson | Spearman |")
    L.append("|---|---|---|---|---|")
    label = {
        "cell_level_n24_maxz": ("24셀", "maxz"),
        "cell_level_n24_cov": ("24셀", "cov"),
        "split_level_n4_maxz": ("4split", "maxz"),
        "split_level_n4_cov": ("4split", "cov"),
    }
    for scope_key, targets in corr.items():
        sc, ood_name = label[scope_key]
        for tgt in ("drop_film", "drop_concat", "seedstd_film", "seedstd_concat"):
            p = targets[tgt]
            L.append(f"| {sc} | {ood_name} | {tgt} | {fmt(p['pearson'])} | {fmt(p['spearman'])} |")
    L.append("")

    L.append("## 주의 / 한계")
    L.append("")
    L.append("- n=4 split(유효 OOD 수준 4개)이라 상관은 **정황 증거**이며 인과가 아니다. 개입 재학습 없음.")
    L.append("- §0 checkpoint-selection gap 진단과 교란 가능(OOD 입력이 val_loss 기반 selection gap도 키울 수 있음). "
             "두 원인이 별개인지 같은 뿌리인지는 이 진단만으로 분리 불가.")
    L.append("- gating이 실제 이득인지는 학습 필요 → 본 진단은 '방향 제언'까지만.")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    zidx = load_zscore_index()
    cells = build_cells(zidx)
    split_rows = agg_by_split(cells)
    corr = correlations(cells, split_rows)
    verdict = direction_verdict(split_rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    json_out = {
        "ood_definition": {"mean_dims": MEAN_DIMS, "z_threshold": 2.0},
        "cells": cells,
        "split_level": split_rows,
        "correlations": corr,
        "verdict": verdict,
    }
    json_path = os.path.join(OUT_DIR, "ood_vs_perf.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    report = build_report(cells, split_rows, corr, verdict)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    # 콘솔 요약
    print(f"wrote {json_path}")
    print(f"wrote {REPORT_PATH}")
    print("\n[split-level, OOD 내림차순]")
    for r in split_rows:
        print(
            f"  {r['loso_split']:>7} ood_maxz={r['ood_maxz']:8.2f} "
            f"drop_FiLM={r['drop_film']:+.3f} drop_concat={r['drop_concat']:+.3f} "
            f"seedstd_FiLM={r['seedstd_film']:.3f} seedstd_concat={r['seedstd_concat']:.3f}"
        )
    print(f"\n[verdict] hypothesis_supported={verdict['hypothesis_supported']} "
          f"(both_drop={verdict['both_methods_drop_larger_in_ood']}, "
          f"both_seedstd={verdict['both_methods_seedstd_larger_in_ood']})")


if __name__ == "__main__":
    main()
