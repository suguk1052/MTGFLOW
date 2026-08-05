"""작업 D 게이트 진단 — 진폭 confound 교정(Route 2) 전처리의 성공 게이트 판정.

전제: 진폭 정규화(amp_normalize) no-meta flow가 학습돼 배치 폴더(기본 LONO_D_ampnorm_s2026)에
체크포인트로 존재. 이 스크립트는 **재추론만**(새 학습 없음)으로 게이트 (a)(b)와 AUROC를 낸다.

anomaly score = flow_NLL(shape) + λ·0.5·penalty(z_rms)
  - flow_NLL(shape): per-window 정규화된 window의 flow NLL(-log p). 진폭 제거됨.
  - z_rms: 떼어낸 log-RMS의 train-normal z-score(로더가 batch/dataset로 노출).
  - penalty: one-sided=max(0,z)² / two-sided=z² (체크포인트 metadata 값).
  - λ: **결정 3 — test 금지, val-normal 분산정합으로 fold별 사전 고정.**
      λ* = Var(flow_NLL | val-normal) / Var(0.5·penalty(z_rms) | val-normal).

게이트(작업 B 0.96·AUROC보다 우선):
  (a) 정상 window에서 **정규화 *이전* 원 RMS**와 **flow_NLL(shape) 단독**의 Spearman ρ이 0.96에서
      유의미 하락. (정규화 후 window RMS≈1이라 결합 score로 재면 무의미 → 원 RMS·flow 단독으로만.)
  (b) 고진폭 정상(K001/K003/K006) 오탐 감소. threshold=val-normal 결합 score 95p. 저진폭 정상
      오탐이 새로 늘지 않는지 함께(one-sided 페널티 타당성).
부수: flow-branch 정상-결함 분리 AUROC(충격형 결함이 RMS 정규화로 약화되는지), 결합 AUROC,
      원 RMS의 fault vs normal 분포 겹침.

산출:
- results/Paderborn/diag_amp_correction/<split>_LONO<n>_s<seed>.json (fold별)
- results/Paderborn/diag_amp_correction/aggregate_s<seed>.json (집계)
- reports/report_amp_correction.md
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
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "report_amp_correction.md")

HIGH_AMP = {"K001", "K003", "K006"}  # 고진폭 정상(작업 B). 나머지 K00x = 저진폭.

# LOSO split ↔ (설명, fold 유형). 작업 B와 동일 규약(뭉뚱그리지 않음).
SPLIT_INFO = {
    "123to0": ("기준조건(N15_M07_F10) unseen", "compositional"),
    "023to1": ("저속(N09) unseen", "zero-support"),
    "013to2": ("저토크(M01) unseen", "zero-support"),
    "012to3": ("저 radial force(F04) unseen", "zero-support"),
}


# ---------------------------------------------------------------------------
# 모델·로더 복원 (diagnose_nll_decomposition.py 규약 준용, 자립형)
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
        meta_input_dim=int(meta["meta_input_dim"]),
        meta_emb_dim=int(meta["meta_emb_dim"]),
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
            meta_source=meta["meta_source"],
            measured_meta_stats=meta["measured_meta_stats"],
            amp_normalize=bool(meta.get("amp_normalize", False)),
            rms_eps=float(meta.get("rms_eps", 1e-8)),
            batch_size=batch_size,
        )


@torch.no_grad()
def flow_nll(model, windows, device, batch_size=256):
    """windows: [N, win](정규화된 shape) → flow_NLL(=-log p_x) per window [N]."""
    n = len(windows)
    win = windows.shape[1]
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        xb = torch.as_tensor(windows[s:e], dtype=torch.float32, device=device)
        xb = xb.reshape(e - s, win, 1, 1).transpose(1, 2).contiguous()  # [b,K=1,L,D=1]
        out[s:e] = (-model.test(xb, None)).cpu().numpy()
    return out


def penalty(z, kind):
    z = np.asarray(z, dtype=np.float64)
    if kind == "two-sided":
        return 0.5 * z ** 2
    return 0.5 * np.maximum(0.0, z) ** 2


# ---------------------------------------------------------------------------
# 통계 유틸 (작업 B와 동일 정의)
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
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def fmt(v, p=3):
    if v is None:
        return "—"
    if isinstance(v, float) and v != v:
        return "nan"
    return f"{v:.{p}f}"


# ---------------------------------------------------------------------------
# fold 하나 처리
# ---------------------------------------------------------------------------
def process_fold(batch_dir, run_prefix, split, lono, seed, device, batch_size):
    run_name = f"{run_prefix}_{split}_LONO{lono}_s{seed}"
    ckpt_path = os.path.join(RESULTS_ROOT, batch_dir, run_name, "model.pth")
    if not os.path.exists(ckpt_path):
        print(f"  [skip] checkpoint 없음: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["paderborn_metadata"]
    assert not bool(meta["use_meta"]), "no-meta 전용 진단인데 use_meta=True 체크포인트"
    assert bool(meta.get("amp_normalize", False)), "amp_normalize=False 체크포인트 — 작업 D 대상 아님"
    rms_kind = meta.get("rms_penalty", "one-sided")

    train_loader, val_loader, test_loader, n_sensor = build_loader(meta, batch_size)
    model = build_model(ckpt, n_sensor, device)

    def pull(ds):
        return (np.asarray(ds.windows, dtype=np.float32),
                np.asarray(ds.rms_raw, dtype=np.float64),
                np.asarray(ds.rms_z, dtype=np.float64).reshape(-1),
                np.asarray(ds.ids), np.asarray(ds.label, dtype=int))

    tr_w, tr_rms, tr_z, tr_ids, _ = pull(train_loader.dataset)
    va_w, va_rms, va_z, va_ids, _ = pull(val_loader.dataset)
    te_w, te_rms, te_z, te_ids, te_lab = pull(test_loader.dataset)

    # flow_NLL(shape)
    tr_f = flow_nll(model, tr_w, device, batch_size)
    va_f = flow_nll(model, va_w, device, batch_size)
    te_f = flow_nll(model, te_w, device, batch_size)

    # penalty
    va_pen = penalty(va_z, rms_kind)
    te_pen = penalty(te_z, rms_kind)

    # ---- λ* : val-normal 분산정합 (test 미사용, 사전 고정) ----
    var_f = float(np.var(va_f)) if len(va_f) else 0.0
    var_pen = float(np.var(va_pen)) if len(va_pen) else 0.0
    lam = var_f / (var_pen + 1e-12) if var_pen > 0 else 0.0

    # 결합 score
    va_score = va_f + lam * va_pen
    te_score = te_f + lam * te_pen

    tgt = te_lab == 0
    flt = te_lab == 1

    # ---- 정상 pooled (source+target normal) ----
    norm_rms = np.concatenate([tr_rms, va_rms, te_rms[tgt]])
    norm_flow = np.concatenate([tr_f, va_f, te_f[tgt]])
    norm_pen = np.concatenate([penalty(tr_z, rms_kind), va_pen, te_pen[tgt]])
    norm_score = norm_flow + lam * norm_pen
    norm_ids = np.concatenate([tr_ids, va_ids, te_ids[tgt]])

    # ---- 게이트 (a): 원 RMS vs flow_NLL(shape) 단독 ----
    rho_flow = spearman(norm_rms, norm_flow)            # 게이트 (a) 지표
    rho_combined = spearman(norm_rms, norm_score)       # 참고(결합은 재상승 예상 — 게이트 아님)

    # ---- 게이트 (b): threshold = val-normal 결합 score 95p, per-bearing FPR ----
    thr = float(np.percentile(va_score, 95)) if len(va_score) else float("inf")
    thr_flow = float(np.percentile(va_f, 95)) if len(va_f) else float("inf")
    per_bearing = {}
    for bid in sorted(set(norm_ids.tolist())):
        m = norm_ids == bid
        per_bearing[bid] = {
            "n": int(m.sum()),
            "group": "high" if bid in HIGH_AMP else "low",
            "mean_rms": float(np.mean(norm_rms[m])),
            "mean_flow_nll": float(np.mean(norm_flow[m])),
            "fpr_combined": float(np.mean(norm_score[m] >= thr)),
            "fpr_flow_only": float(np.mean(norm_flow[m] >= thr_flow)),
        }
    high_fpr = float(np.mean([v["fpr_combined"] for k, v in per_bearing.items() if v["group"] == "high"])) \
        if any(v["group"] == "high" for v in per_bearing.values()) else float("nan")
    low_fpr = float(np.mean([v["fpr_combined"] for k, v in per_bearing.items() if v["group"] == "low"])) \
        if any(v["group"] == "low" for v in per_bearing.values()) else float("nan")

    # ---- AUROC (target-normal 0 vs fault 1) ----
    auroc_combined = safe_auroc(te_lab, te_score)
    auroc_flow = safe_auroc(te_lab, te_f)

    # ---- 원 RMS 분포 겹침(진폭형 결함이 정상 고진폭 대역에 묻히는지) ----
    def q(a):
        a = np.asarray(a, float)
        return {} if not len(a) else {"n": int(len(a)), "median": float(np.median(a)),
                                       "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75))}
    rms_dist = {"normal": q(norm_rms), "fault": q(te_rms[flt])}

    return {
        "split": split, "lono": lono, "seed": seed,
        "fold_type": SPLIT_INFO[split][1], "desc": SPLIT_INFO[split][0],
        "target_norm_ids": sorted(set(te_ids[tgt].tolist())),
        "rms_penalty": rms_kind,
        "lambda_star": lam,
        "val_var_flow": var_f, "val_var_penalty": var_pen,
        "gate_a": {"rho_rms_flow": rho_flow, "rho_rms_combined_ref": rho_combined},
        "gate_b": {"threshold_combined": thr, "high_amp_fpr": high_fpr, "low_amp_fpr": low_fpr},
        "auroc": {"combined": auroc_combined, "flow_only": auroc_flow},
        "flow_branch_separation_auroc": auroc_flow,  # 충격형 결함 약화 점검(정상 vs fault, flow 단독)
        "per_bearing_normal": per_bearing,
        "rms_distribution": rms_dist,
    }


# ---------------------------------------------------------------------------
# 집계 + 리포트
# ---------------------------------------------------------------------------
def aggregate(folds):
    ok = [f for f in folds if f]
    by_type = {}
    for ftype in ("zero-support", "compositional"):
        rows = [f for f in ok if f["fold_type"] == ftype]
        if not rows:
            continue

        def mean_of(fn):
            vals = [fn(r) for r in rows]
            vals = [v for v in vals if v == v]
            return float(np.mean(vals)) if vals else float("nan")

        by_type[ftype] = {
            "n_folds": len(rows),
            "rho_rms_flow": mean_of(lambda r: r["gate_a"]["rho_rms_flow"]),
            "high_amp_fpr": mean_of(lambda r: r["gate_b"]["high_amp_fpr"]),
            "low_amp_fpr": mean_of(lambda r: r["gate_b"]["low_amp_fpr"]),
            "auroc_combined": mean_of(lambda r: r["auroc"]["combined"]),
            "auroc_flow_only": mean_of(lambda r: r["auroc"]["flow_only"]),
            "lambda_star": mean_of(lambda r: r["lambda_star"]),
        }
    return {"n_folds": len(ok), "by_fold_type": by_type}


def build_report(folds, agg, seed, b_rho=0.96):
    L = []
    L.append("# 작업 D 게이트 진단 — 진폭 confound 교정(Route 2, Paderborn no-meta, 무학습)")
    L.append("")
    L.append(f"seed {seed}, amp_normalize no-meta LOSO, 4 split × 6 LONO. 학습된 진폭정규화 "
             "체크포인트 재추론만. anomaly score = flow_NLL(shape) + λ·0.5·penalty(z_rms).")
    L.append("")
    L.append("**λ 규약(결정 3):** test 미사용. fold별 **val-normal 분산정합** "
             "λ\\* = Var(flow_NLL|val)/Var(0.5·penalty|val)로 사전 고정. 아래 수치는 이 λ\\* 적용.")
    L.append("")
    L.append("**게이트 (a):** 정상 window에서 **정규화 이전 원 RMS**와 **flow_NLL(shape) 단독**의 "
             f"Spearman ρ. 작업 B 기준선 ρ≈{b_rho}. (결합 score의 ρ은 z_rms²가 RMS 단조라 재상승 → "
             "게이트 아님, 참고용만.)")
    L.append("")

    L.append("## 1) fold 유형별 집계")
    L.append("")
    L.append("| fold 유형 | n | ρ(RMS,flow) [게이트a] | 고진폭 FPR | 저진폭 FPR | AUROC(결합) | AUROC(flow) | λ* |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ftype, a in agg["by_fold_type"].items():
        L.append(f"| {ftype} | {a['n_folds']} | {fmt(a['rho_rms_flow'])} | {fmt(a['high_amp_fpr'])} | "
                 f"{fmt(a['low_amp_fpr'])} | {fmt(a['auroc_combined'])} | {fmt(a['auroc_flow_only'])} | "
                 f"{fmt(a['lambda_star'])} |")
    L.append("")
    L.append(f"> 게이트 (a) 통과 = ρ(RMS,flow)가 {b_rho}에서 유의미 하락. "
             "게이트 (b) 통과 = 고진폭 FPR 감소 + 저진폭 FPR 미증가.")
    L.append("")

    L.append("## 2) fold 상세")
    L.append("")
    L.append("| split | LONO | target-norm | ρ(RMS,flow) | ρ(RMS,결합)참고 | 고진폭FPR | 저진폭FPR | AUROC결합 | AUROC flow | λ* |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for f in folds:
        if not f:
            continue
        L.append(f"| {f['split']} | {f['lono']} | {','.join(f['target_norm_ids'])} | "
                 f"{fmt(f['gate_a']['rho_rms_flow'])} | {fmt(f['gate_a']['rho_rms_combined_ref'])} | "
                 f"{fmt(f['gate_b']['high_amp_fpr'])} | {fmt(f['gate_b']['low_amp_fpr'])} | "
                 f"{fmt(f['auroc']['combined'])} | {fmt(f['auroc']['flow_only'])} | {fmt(f['lambda_star'])} |")
    L.append("")

    L.append("## 3) 정상 bearing별 (통과 fold pooled) — 게이트 (b) 근거")
    L.append("")
    bagg = {}
    for f in folds:
        if not f:
            continue
        for bid, d in f["per_bearing_normal"].items():
            e = bagg.setdefault(bid, {"group": d["group"], "rms": [], "flow": [], "fpr_c": [], "fpr_f": []})
            e["rms"].append(d["mean_rms"]); e["flow"].append(d["mean_flow_nll"])
            e["fpr_c"].append(d["fpr_combined"]); e["fpr_f"].append(d["fpr_flow_only"])
    L.append("| bearing | 진폭군 | fold수 | mean RMS | mean flow_NLL | FPR(결합) | FPR(flow단독) |")
    L.append("|---|---|---|---|---|---|---|")
    for bid in sorted(bagg):
        e = bagg[bid]
        grp = "고진폭" if e["group"] == "high" else "저진폭"
        L.append(f"| {bid} | {grp} | {len(e['rms'])} | {fmt(np.mean(e['rms']))} | {fmt(np.mean(e['flow']))} | "
                 f"{fmt(np.mean(e['fpr_c']))} | {fmt(np.mean(e['fpr_f']))} |")
    L.append("")

    L.append("## 주의 / 한계")
    L.append("")
    L.append("- 무학습 재추론. seed 2026 단일.")
    L.append("- 게이트 (a) ρ은 원 RMS(정규화 전) vs flow_NLL 단독 — 결합 score ρ과 혼동 금지.")
    L.append("- λ\\*는 val-normal에서만 산출(test leakage 없음). test 셋 λ 스윕은 별도 민감도이며 선택자 아님.")
    L.append("- flow-branch 분리 AUROC(=AUROC flow)는 충격형 결함이 RMS 정규화로 약화되는지 점검용. "
             "B3(정규화 전) 대비 하락 시 실패모드 신호 — report_nll_decomposition.md 수치와 대조.")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--batch_dir", type=str, default="LONO_D_ampnorm_s2026",
                    help="진폭정규화 체크포인트 배치 폴더(results/Paderborn 하위).")
    ap.add_argument("--run_prefix", type=str, default="ampnorm",
                    help="run_name 접두사. run_name = <prefix>_<split>_LONO<n>_s<seed>.")
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_INFO.keys()), choices=list(SPLIT_INFO.keys()))
    ap.add_argument("--lonos", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_dir", type=str, default=os.path.join(RESULTS_ROOT, "diag_amp_correction"))
    ap.add_argument("--no_report", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    os.makedirs(args.out_dir, exist_ok=True)

    folds = []
    for split in args.splits:
        for lono in args.lonos:
            res = process_fold(args.batch_dir, args.run_prefix, split, lono, args.seed, device, args.batch_size)
            if res is None:
                continue
            folds.append(res)
            with open(os.path.join(args.out_dir, f"{split}_LONO{lono}_s{args.seed}.json"), "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            g = res
            print(f"  [{split} LONO{lono}] ρ(RMS,flow)={fmt(g['gate_a']['rho_rms_flow'])} "
                  f"고진폭FPR={fmt(g['gate_b']['high_amp_fpr'])} 저진폭FPR={fmt(g['gate_b']['low_amp_fpr'])} "
                  f"AUROC결합={fmt(g['auroc']['combined'])} λ*={fmt(g['lambda_star'])}")

    if args.no_report or not folds:
        print(f"\nfold JSON {len(folds)}개 기록. (집계/리포트 생략)")
        return

    agg = aggregate(folds)
    with open(os.path.join(args.out_dir, f"aggregate_s{args.seed}.json"), "w") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    with open(REPORT_PATH, "w") as f:
        f.write(build_report(folds, agg, args.seed))
    print(f"\nwrote {REPORT_PATH}")
    print("\n=== fold 유형별 집계 ===")
    for ftype, a in agg["by_fold_type"].items():
        print(f"  {ftype:>13} n={a['n_folds']} ρ(RMS,flow)={fmt(a['rho_rms_flow'])} "
              f"고진폭FPR={fmt(a['high_amp_fpr'])} 저진폭FPR={fmt(a['low_amp_fpr'])} "
              f"AUROC결합={fmt(a['auroc_combined'])}")


if __name__ == "__main__":
    main()
