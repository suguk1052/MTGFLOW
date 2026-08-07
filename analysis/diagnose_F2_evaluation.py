"""작업 F-2 평가 — Vy-rmsnorm 기준 C1+C2(전류) 다변량 확장(채널-as-노드) 검증.

핵심 질문: Vy-rmsnorm(F-1 기준선)에 전류 C1+C2를 dynamic-graph 노드로 추가하면
  (i) 독립적 결함 정보를 보완하는가, 아니면 (ii) 운행조건·진폭 confound를 재주입하는가.

두 입력 구성을 나란히 비교(둘 다 채널셋 C1C2Vy = 3노드, no-meta, window 2048, seed 2026):
  - all   : Vy-rmsnorm + C1-rmsnorm + C2-rmsnorm (전 채널 shape-only)
  - mixed : Vy-rmsnorm + C1-raw + C2-raw       (전류 절대 진폭 보존)
기준선 Vy(단일채널 rmsnorm)는 재학습하지 않고 F-1 결과(diag_F1_evaluation/*.json의 'rmsnorm')를 재사용.

점수 = 순수 shape flow_NLL(rms_lambda=0), F-1과 동일 정의라 paired 비교 성립.

fold별 산출(각 구성 + Vy 기준선):
  1) 분리 AUROC: target-normal(0) vs fault 전체(1).
  2) per-fault AUROC: KA15·KA22 명시.
  3) ρ(RMS_Vy, flow) + ρ(RMS_current, flow): 정상 pooled(train+val+target-normal), Spearman.
     → 전류 진폭 confound 재주입 정량화(F-2 신규 축).
  4) 정상 FPR: threshold=val-normal flow 95p, 고/저진폭 그룹 평균.
  5) 저진폭 target(LONO2/4/5) 거동.
  6) mean adjacency A(3×3): 전류 노드가 진동 노드 density에 정보를 주는지 격리(F-2 신규 축).

산출:
- results/Paderborn/diag_F2_evaluation/<config>_<split>_LONO<n>_s<seed>.json (fold·구성별)
- results/Paderborn/diag_F2_evaluation/aggregate_s<seed>.json
- reports/report_F2_evaluation.md
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.MTGFLOW import MTGFLOW  # noqa: E402
from Dataset.paderborn import loader_Paderborn_OCC  # noqa: E402

DATA_ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "Data", "Paderborn"))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "Paderborn")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "report_F2_evaluation.md")

HIGH_AMP = {"K001", "K003", "K006"}

SPLIT_INFO = {
    "123to0": ("기준조건(N15_M07_F10) unseen", "compositional"),
    "023to1": ("저속(N09) unseen", "zero-support"),
    "013to2": ("저토크(M01) unseen", "zero-support"),
    "012to3": ("저 radial force(F04) unseen", "zero-support"),
}
LONO_TARGET = {1: "K001", 2: "K002", 3: "K003", 4: "K004", 5: "K005", 6: "K006"}

# F-2 구성 정의(채널셋 C1C2Vy 고정, amp_normalize_channels로 구성 구분).
CONFIGS = {
    "all":   {"prefix": "f2_all",   "desc": "전 채널 rmsnorm(shape-only)"},
    "mixed": {"prefix": "f2_mixed", "desc": "Vy-rmsnorm + 전류 raw(진폭 보존)"},
}


# ---------------------------------------------------------------------------
# 모델·로더 복원
# ---------------------------------------------------------------------------
def build_model(ckpt, n_sensor, device):
    a = ckpt.get("args", {})
    meta = ckpt["paderborn_metadata"]
    model = MTGFLOW(
        n_blocks=int(a.get("n_blocks", 2)),
        input_size=int(a.get("input_size", 1)),
        hidden_size=int(a.get("hidden_size", 32)),
        n_hidden=int(a.get("n_hidden", 1)),
        window_size=int(meta["window_size"]),
        n_sensor=n_sensor,
        dropout=0.0,
        model=a.get("model", "MAF"),
        batch_norm=bool(a.get("batch_norm", False)),
        use_meta=bool(meta["use_meta"]),
        meta_input_dim=int(meta.get("meta_input_dim", 3)),
        meta_emb_dim=int(meta.get("meta_emb_dim", 8)),
        meta_inject=meta.get("meta_inject", "concat"),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def build_loader(meta, batch_size):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return loader_Paderborn_OCC(
            root=DATA_ROOT,
            train_loads=meta["train_load_setting"],
            test_loads=meta["test_load_setting"],
            train_ids=meta["train_ids"],
            val_ids=meta["val_ids"],
            test_norm_ids=meta["test_norm_ids"],
            exclude_ids=meta.get("exclude_ids", []),
            window_size=int(meta["window_size"]),
            stride_size=int(meta["stride_size"]),
            sensor_mode=meta["sensor_mode"],
            meta_source=meta.get("meta_source", "static"),
            measured_meta_stats=meta.get("measured_meta_stats", "meanstd"),
            amp_normalize=bool(meta.get("amp_normalize", False)),
            amp_normalize_channels=meta.get("amp_normalize_channels", "all"),
            rms_eps=float(meta.get("rms_eps", 1e-8)),
            batch_size=batch_size,
        )


def _windows_to_batch(windows_slice):
    """ds.windows 조각 → 모델 입력 [b, K, L, 1]. 단일채널(N,win)·다채널(N,win,C) 모두 처리."""
    a = np.asarray(windows_slice, dtype=np.float32)
    if a.ndim == 2:                       # (b, win) 단일채널
        b, win = a.shape
        return a.reshape(b, win, 1, 1).transpose(0, 2, 1, 3)   # (b,1,win,1)
    b, win, c = a.shape                   # (b, win, C) 다채널
    return np.transpose(a, (0, 2, 1))[..., None]               # (b,C,win,1)


@torch.no_grad()
def flow_nll_and_graph(model, windows, device, batch_size=256, collect_graph=False):
    """flow_NLL(=-log p) per window [N]. collect_graph면 mean adjacency A(K,K)도 반환."""
    n = len(windows)
    if n == 0:
        return np.empty(0, dtype=np.float64), None
    out = np.empty(n, dtype=np.float64)
    g_sum, g_cnt = None, 0
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        xb = torch.as_tensor(_windows_to_batch(windows[s:e]), dtype=torch.float32, device=device)
        out[s:e] = (-model.test(xb, None)).cpu().numpy()
        if collect_graph:
            g = getattr(model, "graph", None)   # test()가 self.graph=(N,K,K) 저장
            if g is not None:
                gs = g.detach().cpu().numpy().sum(axis=0)   # (K,K)
                g_sum = gs if g_sum is None else g_sum + gs
                g_cnt += g.shape[0]
    mean_graph = (g_sum / g_cnt).tolist() if (collect_graph and g_cnt) else None
    return out, mean_graph


# ---------------------------------------------------------------------------
# 통계 유틸
# ---------------------------------------------------------------------------
def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def safe_auroc(labels, scores):
    labels = np.asarray(labels, int)
    if len(labels) == 0 or labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def fmt(v, p=3):
    if v is None:
        return "—"
    if isinstance(v, float) and v != v:
        return "nan"
    return f"{v:.{p}f}"


# ---------------------------------------------------------------------------
# 한 구성 checkpoint 재추론 → 지표 dict
# ---------------------------------------------------------------------------
def eval_config(batch_dir, prefix, split, lono, seed, device, batch_size, thr_pct):
    run_name = f"{prefix}_{split}_LONO{lono}_s{seed}"
    ckpt_path = os.path.join(RESULTS_ROOT, run_name, "model.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(RESULTS_ROOT, batch_dir, run_name, "model.pth")
    if not os.path.exists(ckpt_path):
        print(f"    [skip] checkpoint 없음: {run_name}")
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["paderborn_metadata"]
    assert not bool(meta["use_meta"]), f"no-meta 전용 평가인데 use_meta=True: {run_name}"

    train_loader, val_loader, test_loader, n_sensor = build_loader(meta, batch_size)
    model = build_model(ckpt, n_sensor, device)

    def pull(ds):
        rms = np.asarray(ds.rms_raw, dtype=np.float64)
        return (np.asarray(ds.windows, dtype=np.float32), rms,
                np.asarray(ds.ids), np.asarray(ds.label, dtype=int))

    tr_w, tr_rms, tr_ids, _ = pull(train_loader.dataset)
    va_w, va_rms, va_ids, _ = pull(val_loader.dataset)
    te_w, te_rms, te_ids, te_lab = pull(test_loader.dataset)

    tr_f, _ = flow_nll_and_graph(model, tr_w, device, batch_size)
    va_f, _ = flow_nll_and_graph(model, va_w, device, batch_size)
    te_f, _ = flow_nll_and_graph(model, te_w, device, batch_size)

    tgt = te_lab == 0
    flt = te_lab == 1

    # adjacency는 target-normal window 기준 평균(전류→진동 정보 흐름 격리)
    _, mean_graph = flow_nll_and_graph(model, te_w[tgt], device, batch_size, collect_graph=True)

    auroc = safe_auroc(te_lab, te_f)

    tgt_f = te_f[tgt]
    per_fault = {}
    for fid in sorted(set(te_ids[flt].tolist())):
        m = flt & (te_ids == fid)
        f_scores = te_f[m]
        labels = np.concatenate([np.zeros(len(tgt_f), int), np.ones(len(f_scores), int)])
        scores = np.concatenate([tgt_f, f_scores])
        per_fault[fid] = {"n": int(m.sum()), "auroc": safe_auroc(labels, scores)}

    # ρ(RMS_Vy, flow) + ρ(RMS_current, flow) — 정상 pooled
    def _col(rms, kind):
        if rms.ndim == 1:
            return rms
        if kind == "vib":
            return rms[:, 0]
        return rms[:, 1:].mean(axis=1)   # 전류 채널 평균 RMS
    norm_rms = np.concatenate([tr_rms, va_rms, te_rms[tgt]], axis=0)
    norm_flow = np.concatenate([tr_f, va_f, te_f[tgt]])
    norm_ids = np.concatenate([tr_ids, va_ids, te_ids[tgt]])
    rho_vib = spearman(_col(norm_rms, "vib"), norm_flow)
    rho_cur = spearman(_col(norm_rms, "cur"), norm_flow) if norm_rms.ndim == 2 else float("nan")

    thr = float(np.percentile(va_f, thr_pct)) if len(va_f) else float("inf")
    per_bearing = {}
    norm_rms_vib = _col(norm_rms, "vib")
    for bid in sorted(set(norm_ids.tolist())):
        m = norm_ids == bid
        per_bearing[bid] = {
            "n": int(m.sum()),
            "group": "high" if bid in HIGH_AMP else "low",
            "mean_rms_vib": float(np.mean(norm_rms_vib[m])),
            "mean_flow_nll": float(np.mean(norm_flow[m])),
            "fpr": float(np.mean(norm_flow[m] >= thr)),
        }

    def _grp_fpr(grp):
        vals = [v["fpr"] for v in per_bearing.values() if v["group"] == grp]
        return float(np.mean(vals)) if vals else float("nan")

    tgt_bid = LONO_TARGET.get(lono)
    return {
        "run_name": run_name,
        "n_sensor": int(n_sensor),
        "amp_normalize_channels": meta.get("amp_normalize_channels", "all"),
        "auroc": auroc,
        "per_fault_auroc": per_fault,
        "rho_rms_vib_flow": rho_vib,
        "rho_rms_cur_flow": rho_cur,
        "threshold_val95": thr,
        "high_amp_fpr": _grp_fpr("high"),
        "low_amp_fpr": _grp_fpr("low"),
        "mean_adjacency": mean_graph,
        "per_bearing_normal": per_bearing,
        "target_norm_ids": sorted(set(te_ids[tgt].tolist())),
        "target_amp_group": "high" if (tgt_bid in HIGH_AMP) else "low",
        "n": {"train": int(len(tr_w)), "val": int(len(va_w)),
              "target_norm": int(tgt.sum()), "fault": int(flt.sum())},
    }


# ---------------------------------------------------------------------------
# Vy 기준선(F-1 rmsnorm) 로드
# ---------------------------------------------------------------------------
def load_vy_baseline(f1_dir, split, lono, seed):
    path = os.path.join(f1_dir, f"{split}_LONO{lono}_s{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    v = d.get("rmsnorm")
    if not v:
        return None
    return {
        "auroc": v.get("auroc"),
        "per_fault_auroc": v.get("per_fault_auroc", {}),
        "rho_rms_vib_flow": v.get("rho_rms_flow"),
        "high_amp_fpr": v.get("high_amp_fpr"),
        "low_amp_fpr": v.get("low_amp_fpr"),
        "target_amp_group": d.get("target_amp_group"),
        "target_norm_ids": d.get("target_norm_ids"),
    }


# ---------------------------------------------------------------------------
# fold 하나 = 각 구성 + Vy 기준선
# ---------------------------------------------------------------------------
def process_fold(cfg, split, lono, seed, device):
    results = {}
    for ckey in cfg["configs"]:
        results[ckey] = eval_config(cfg["batch_dir"], CONFIGS[ckey]["prefix"], split, lono,
                                    seed, device, cfg["batch_size"], cfg["thr_pct"])
    vy = load_vy_baseline(cfg["f1_dir"], split, lono, seed)
    if all(v is None for v in results.values()) and vy is None:
        return None
    ref = next((v for v in results.values() if v), None)
    tgt_amp = ref["target_amp_group"] if ref else (vy["target_amp_group"] if vy else None)
    tgt_ids = ref["target_norm_ids"] if ref else (vy["target_norm_ids"] if vy else [])
    return {
        "split": split, "lono": lono, "seed": seed,
        "fold_type": SPLIT_INFO[split][1], "desc": SPLIT_INFO[split][0],
        "target_norm_ids": tgt_ids, "target_amp_group": tgt_amp,
        "vy_baseline": vy, "configs": results,
    }


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------
def _mean(vals):
    vals = [v for v in vals if v is not None and v == v]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(folds, config_keys):
    ok = [f for f in folds if f]

    def cfg_auroc(f, ck):
        v = f["configs"].get(ck)
        return v["auroc"] if v else float("nan")

    def vy_auroc(f):
        return f["vy_baseline"]["auroc"] if f["vy_baseline"] else float("nan")

    by_type = {}
    for ftype in ("zero-support", "compositional"):
        rows = [f for f in ok if f["fold_type"] == ftype]
        if not rows:
            continue
        entry = {"n_folds": len(rows), "auroc_vy": _mean([vy_auroc(r) for r in rows])}
        for ck in config_keys:
            entry[f"auroc_{ck}"] = _mean([cfg_auroc(r, ck) for r in rows])
            entry[f"delta_{ck}_vs_vy"] = _mean(
                [cfg_auroc(r, ck) - vy_auroc(r) for r in rows
                 if r["configs"].get(ck) and r["vy_baseline"]])
            entry[f"rho_vib_{ck}"] = _mean([r["configs"][ck]["rho_rms_vib_flow"] for r in rows if r["configs"].get(ck)])
            entry[f"rho_cur_{ck}"] = _mean([r["configs"][ck]["rho_rms_cur_flow"] for r in rows if r["configs"].get(ck)])
            entry[f"high_fpr_{ck}"] = _mean([r["configs"][ck]["high_amp_fpr"] for r in rows if r["configs"].get(ck)])
            entry[f"low_fpr_{ck}"] = _mean([r["configs"][ck]["low_amp_fpr"] for r in rows if r["configs"].get(ck)])
        by_type[ftype] = entry

    by_amp = {}
    for grp in ("high", "low"):
        rows = [f for f in ok if f["target_amp_group"] == grp]
        if not rows:
            continue
        entry = {"n_folds": len(rows), "auroc_vy": _mean([vy_auroc(r) for r in rows]),
                 "n_reversed_vy": int(sum(1 for r in rows if r["vy_baseline"] and r["vy_baseline"]["auroc"] is not None and r["vy_baseline"]["auroc"] < 0.5))}
        for ck in config_keys:
            entry[f"auroc_{ck}"] = _mean([cfg_auroc(r, ck) for r in rows])
            entry[f"n_reversed_{ck}"] = int(sum(1 for r in rows if r["configs"].get(ck) and r["configs"][ck]["auroc"] < 0.5))
        by_amp[grp] = entry

    # per-fault (KA15/KA22 강조)
    fault_agg = {}
    for f in ok:
        for fid, d in (f["vy_baseline"]["per_fault_auroc"].items() if f["vy_baseline"] else []):
            fault_agg.setdefault(fid, {}).setdefault("vy", []).append(
                d["auroc"] if isinstance(d, dict) else d)
        for ck in config_keys:
            v = f["configs"].get(ck)
            if not v:
                continue
            for fid, d in v["per_fault_auroc"].items():
                fault_agg.setdefault(fid, {}).setdefault(ck, []).append(d["auroc"])
    fault_summary = {}
    for fid, e in sorted(fault_agg.items()):
        row = {"auroc_vy": _mean(e.get("vy", []))}
        for ck in config_keys:
            row[f"auroc_{ck}"] = _mean(e.get(ck, []))
        fault_summary[fid] = row

    return {"n_folds": len(ok), "config_keys": config_keys,
            "by_fold_type": by_type, "by_target_amp": by_amp, "per_fault": fault_summary}


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------
def build_report(folds, agg, seed, config_keys, gate):
    ok = [f for f in folds if f]
    cks = config_keys
    L = []
    L.append("# 작업 F-2 평가 — Vy-rmsnorm 기준 C1+C2(전류) 다변량 확장 (채널-as-노드)")
    L.append("")
    scope = "대표 gate fold" if gate else "전체"
    L.append(f"seed {seed}, window 2048, no-meta, 채널셋 C1C2Vy(=3노드), {scope}. 점수=순수 shape flow_NLL. "
             "기준선 Vy(단일채널 rmsnorm)=F-1 결과 재사용(재학습 없음).")
    L.append("")
    L.append("- **all**: 전 채널 rmsnorm(shape-only) — 전류 파형 shape가 독립 fault 정보를 주는지 순수 검증.")
    L.append("- **mixed**: Vy-rmsnorm + 전류 raw(진폭 보존) — 전류 절대 진폭의 조건/진폭 confound 재주입 노출.")
    L.append("")

    zt = agg["by_fold_type"].get("zero-support", {})
    ct = agg["by_fold_type"].get("compositional", {})
    hi = agg["by_target_amp"].get("high", {})
    lo = agg["by_target_amp"].get("low", {})

    L.append("## 핵심 요약 (F-2 검증 목표별)")
    L.append("")
    def _cfg_line(a):
        segs = []
        for ck in cks:
            segs.append(f"{ck} {fmt(a.get(f'auroc_{ck}'))}(Δ{fmt(a.get(f'delta_{ck}_vs_vy'))})")
        return ", ".join(segs)
    L.append(f"- **분리 AUROC (Vy 기준선 대비)**: zero-support Vy {fmt(zt.get('auroc_vy'))} → {_cfg_line(zt)}; "
             f"compositional Vy {fmt(ct.get('auroc_vy'))} → {_cfg_line(ct)}.")
    L.append(f"- **ρ(RMS_current, flow) [confound 재주입 지표]**: "
             + "; ".join(f"{ck} zero {fmt(zt.get(f'rho_cur_{ck}'))}/comp {fmt(ct.get(f'rho_cur_{ck}'))}" for ck in cks)
             + " (0 근방=전류 진폭 무상관, 큰 양수=confound 재주입).")
    L.append(f"- **ρ(RMS_Vy, flow)**: "
             + "; ".join(f"{ck} zero {fmt(zt.get(f'rho_vib_{ck}'))}/comp {fmt(ct.get(f'rho_vib_{ck}'))}" for ck in cks)
             + " (F-1 Vy 기준선 ≈ -0.36/-0.27).")
    lo_line = ", ".join(f"{ck} {fmt(lo.get(f'auroc_{ck}'))}(역전 {lo.get(f'n_reversed_{ck}')}/{lo.get('n_folds')})" for ck in cks)
    L.append(f"- **저진폭 target(LONO2/4/5)**: Vy {fmt(lo.get('auroc_vy'))}(역전 {lo.get('n_reversed_vy')}/{lo.get('n_folds')}) → {lo_line}.")
    L.append(f"- **정상 FPR**: 고진폭 "
             + "; ".join(f"{ck} zero {fmt(zt.get(f'high_fpr_{ck}'))}/comp {fmt(ct.get(f'high_fpr_{ck}'))}" for ck in cks)
             + " / 저진폭 "
             + "; ".join(f"{ck} zero {fmt(zt.get(f'low_fpr_{ck}'))}/comp {fmt(ct.get(f'low_fpr_{ck}'))}" for ck in cks) + ".")
    L.append("")

    # 1) fold 유형별
    L.append("## 1) fold 유형별 집계 (Vy 기준선 vs 구성)")
    L.append("")
    hdr = "| fold 유형 | n | AUROC(Vy) | " + " | ".join(f"AUROC({ck})/Δ" for ck in cks) \
          + " | " + " | ".join(f"ρ_cur({ck})" for ck in cks) + " | " + " | ".join(f"ρ_vib({ck})" for ck in cks) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (2 + 1 + len(cks) * 3))
    for ftype, a in agg["by_fold_type"].items():
        cells = [ftype, str(a["n_folds"]), fmt(a.get("auroc_vy"))]
        cells += [f"{fmt(a.get(f'auroc_{ck}'))}/{fmt(a.get(f'delta_{ck}_vs_vy'))}" for ck in cks]
        cells += [fmt(a.get(f"rho_cur_{ck}")) for ck in cks]
        cells += [fmt(a.get(f"rho_vib_{ck}")) for ck in cks]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # 2) target 진폭군별
    L.append("## 2) target-normal 진폭군별 분리 AUROC (저진폭 역전 진단)")
    L.append("")
    L.append("| target 진폭군 | n | AUROC(Vy) | 역전(Vy) | " + " | ".join(f"AUROC({ck})/역전" for ck in cks) + " |")
    L.append("|" + "---|" * (4 + len(cks)))
    for grp, a in agg["by_target_amp"].items():
        gname = "고진폭(K001/K003/K006)" if grp == "high" else "저진폭(K002/K004/K005)"
        cells = [gname, str(a["n_folds"]), fmt(a.get("auroc_vy")), str(a.get("n_reversed_vy"))]
        cells += [f"{fmt(a.get(f'auroc_{ck}'))}/{a.get(f'n_reversed_{ck}')}" for ck in cks]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # 3) per-fault (KA15/KA22 강조)
    L.append("## 3) per-fault AUROC 집계 (fault id별 fold 평균) — KA15·KA22 강조")
    L.append("")
    L.append("| fault id | AUROC(Vy) | " + " | ".join(f"AUROC({ck})" for ck in cks) + " |")
    L.append("|" + "---|" * (2 + len(cks)))
    for fid, d in agg["per_fault"].items():
        star = " ⚠" if fid in ("KA15", "KA22") else ""
        cells = [f"{fid}{star}", fmt(d.get("auroc_vy"))] + [fmt(d.get(f"auroc_{ck}")) for ck in cks]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append("> ⚠ = 작업 B/F-0/F-1에서 불가시(AUROC≈0.5) 이력. 전류 추가로 상승하면 독립 정보 보완 신호.")
    L.append("")

    # 4) fold 상세 + adjacency
    L.append("## 4) fold 상세 (AUROC + mean adjacency A[Vy행: 전류→진동 기여])")
    L.append("")
    L.append("| split | LONO | 유형 | target | AUROC(Vy) | " + " | ".join(f"AUROC({ck})" for ck in cks)
             + " | " + " | ".join(f"A[Vy←C1,C2]({ck})" for ck in cks) + " |")
    L.append("|" + "---|" * (5 + len(cks) * 2))
    for f in ok:
        vy = f["vy_baseline"]
        cells = [f["split"], str(f["lono"]), f["fold_type"], ",".join(f["target_norm_ids"]),
                 fmt(vy["auroc"]) if vy else "—"]
        cells += [fmt(f["configs"][ck]["auroc"]) if f["configs"].get(ck) else "—" for ck in cks]
        for ck in cks:
            v = f["configs"].get(ck)
            g = v.get("mean_adjacency") if v else None
            if g and len(g) == 3:
                # A[i,j]=노드 i가 노드 j를 참조(softmax dim=1). Vy=노드0 행에서 C1(1),C2(2) 기여.
                cells.append(f"{g[0][1]:.2f},{g[0][2]:.2f}")
            else:
                cells.append("—")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 판정 가이드 (go/no-go)")
    L.append("")
    L.append("- **독립 fault 정보 보완**: 분리 AUROC↑(Δ>0) & 저진폭 target 개선·역전 감소 & KA15/KA22 상승 & "
             "ρ(RMS_current) 무상관 & adjacency가 전류→진동 비자명 기여.")
    L.append("- **confound 재주입**: ρ(RMS_current)↑(양수) & 저진폭 target 악화 & 고진폭 정상 FPR↑ "
             "(특히 mixed·zero-support 023to1/013to2).")
    L.append("- all은 개선·mixed는 악화면 → 전류 shape=독립 정보, 절대 진폭=confound로 분리 결론.")
    L.append("")
    L.append("## 주의 / 한계")
    L.append("")
    L.append("- 무학습 재추론(학습된 F-2 체크포인트), seed 2026 단일. target-normal이 setting+bearing 이중 "
             "held-out → 개체차 confound 잔존(F-0/B/F-1과 정합).")
    L.append("- adjacency는 softmax(dim=1) 참조 가중; 절대적 인과 아님(정보 흐름 대리 지표).")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--batch_dir", type=str, default="LONO_F2_c1c2vy_s2026",
                    help="F-2 체크포인트 배치 폴더(results/Paderborn/<batch_dir>/). run은 배치 폴더 또는 루트에서 탐색.")
    ap.add_argument("--configs", nargs="+", default=["all", "mixed"], choices=list(CONFIGS.keys()))
    ap.add_argument("--f1_dir", type=str, default=os.path.join(RESULTS_ROOT, "diag_F1_evaluation"),
                    help="Vy 기준선(F-1 rmsnorm) fold JSON 폴더.")
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_INFO.keys()), choices=list(SPLIT_INFO.keys()))
    ap.add_argument("--lonos", type=int, nargs="+", default=[1, 2],
                    help="기본 gate=LONO 1(고진폭 K001)·2(저진폭 K002). 전체 확장 시 1..6.")
    ap.add_argument("--threshold_percentile", type=float, default=95.0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_dir", type=str, default=os.path.join(RESULTS_ROOT, "diag_F2_evaluation"))
    ap.add_argument("--no_report", action="store_true")
    args = ap.parse_args()

    gate = (sorted(args.lonos) == [1, 2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, configs={args.configs}, splits={args.splits}, lonos={args.lonos}")
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = {"batch_dir": args.batch_dir, "configs": args.configs, "f1_dir": args.f1_dir,
           "batch_size": args.batch_size, "thr_pct": args.threshold_percentile}

    folds = []
    for split in args.splits:
        for lono in args.lonos:
            print(f"[{split} LONO{lono}]")
            res = process_fold(cfg, split, lono, args.seed, device)
            if res is None:
                continue
            folds.append(res)
            with open(os.path.join(args.out_dir, f"{split}_LONO{lono}_s{args.seed}.json"), "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            vy = res["vy_baseline"]
            msg = f"    Vy={fmt(vy['auroc']) if vy else '—'}"
            for ck in args.configs:
                v = res["configs"].get(ck)
                msg += f" | {ck}={fmt(v['auroc']) if v else '—'}"
                if v:
                    msg += f"(ρcur={fmt(v['rho_rms_cur_flow'])})"
            print(msg)

    if args.no_report or not folds:
        print(f"\nfold JSON {len(folds)}개 기록. (집계/리포트 생략)")
        return

    agg = aggregate(folds, args.configs)
    with open(os.path.join(args.out_dir, f"aggregate_s{args.seed}.json"), "w") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    with open(REPORT_PATH, "w") as f:
        f.write(build_report(folds, agg, args.seed, args.configs, gate))
    print(f"\nwrote {REPORT_PATH}")
    for ftype, a in agg["by_fold_type"].items():
        seg = " ".join(f"{ck}={fmt(a.get(f'auroc_{ck}'))}(Δ{fmt(a.get(f'delta_{ck}_vs_vy'))})" for ck in args.configs)
        print(f"  {ftype:>13} n={a['n_folds']} Vy={fmt(a.get('auroc_vy'))} {seg}")


if __name__ == "__main__":
    main()
