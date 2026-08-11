# ==============================================================================
# 작업 E 1단계 Phase B — order tracking 학습 모델 재추론 평가 (raw vs raw+OT)
#
# 설계: 023→1은 source=1500rpm 단일이라 OT가 train에서 identity → raw seed2026 checkpoint
# 하나가 raw·raw+OT 양쪽 모델. 따라서 동일 checkpoint를 order_track off/on 두 loader로
# 재추론해 비교한다(OT는 순수 test-time 변환: target 900rpm window만 0.8rev로 재표현).
#
# 산출: overall AUROC(target-normal vs fault 전체) · per-fault AUROC · 정상 FPR(val 95p 고정)
#   - raw arm overall을 아카이브 B3 seed2026(LONO1 0.088 / LONO2 0.581)과 대조(재현 검증).
#   - OT nominal(primary) vs inst(sanity) 동치 확인.
#
# diagnose_F1_evaluation.py의 build_model/flow_nll/safe_auroc/per-fault·FPR 패턴 재사용.
# ==============================================================================

import contextlib
import io
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
OUT_DIR = os.path.join(RESULTS_ROOT, "E_order_tracking")

HIGH_AMP = {"K001", "K003", "K006"}
THR_PCT = 95

# fold → (checkpoint run_name(_s2026), 아카이브 B3 seed2026 raw AUROC)
SEEDS = [2024, 2025, 2026, 2027, 2028]  # 5-seed 확장

# fold → 메타. checkpoint run_dir = E_raw_023to1_{fold}_s{seed}. archive = B3 seed2026 raw AUROC(대조).
FOLDS = {
    "LONO1": {"archive_raw_auroc": 0.0876, "target": "K001", "amp": "high"},
    "LONO2": {"archive_raw_auroc": 0.5808, "target": "K002", "amp": "low"},
}


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


def build_loader(meta, batch_size, order_track, order_track_ref="nominal"):
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
            rms_eps=float(meta.get("rms_eps", 1e-8)),
            order_track=order_track,
            order_track_ref=order_track_ref,
            batch_size=batch_size,
        )


@torch.no_grad()
def flow_nll(model, windows, device, batch_size=256):
    n = len(windows)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    win = windows.shape[1]
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        xb = torch.as_tensor(windows[s:e], dtype=torch.float32, device=device)
        xb = xb.reshape(e - s, win, 1, 1).transpose(1, 2).contiguous()
        out[s:e] = (-model.test(xb, None)).cpu().numpy()
    return out


def safe_auroc(labels, scores):
    labels = np.asarray(labels, int)
    if len(labels) == 0 or labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def eval_arm(ckpt, meta, device, batch_size, order_track, order_track_ref, val_thr=None):
    tr, va, te, n_sensor = build_loader(meta, batch_size, order_track, order_track_ref)
    model = build_model(ckpt, n_sensor, device)

    def pull(ds):
        return (np.asarray(ds.windows, dtype=np.float32),
                np.asarray(ds.ids), np.asarray(ds.label, dtype=int))

    va_w, _, _ = pull(va.dataset)
    te_w, te_ids, te_lab = pull(te.dataset)
    va_f = flow_nll(model, va_w, device, batch_size)
    te_f = flow_nll(model, te_w, device, batch_size)

    tgt = te_lab == 0
    flt = te_lab == 1
    auroc = safe_auroc(te_lab, te_f)

    tgt_f = te_f[tgt]
    per_fault = {}
    for fid in sorted(set(te_ids[flt].tolist())):
        m = flt & (te_ids == fid)
        labels = np.concatenate([np.zeros(len(tgt_f), int), np.ones(int(m.sum()), int)])
        scores = np.concatenate([tgt_f, te_f[m]])
        per_fault[fid] = {"n": int(m.sum()),
                          "family": fid[:2],
                          "auroc": safe_auroc(labels, scores)}

    # threshold = val-normal 95p (raw arm에서 계산해 두 arm 공통 사용: source identity라 val 동일)
    thr = val_thr if val_thr is not None else (float(np.percentile(va_f, THR_PCT)) if len(va_f) else float("inf"))
    target_norm_fpr = float(np.mean(tgt_f >= thr)) if len(tgt_f) else float("nan")

    return {
        "order_track": order_track,
        "order_track_ref": order_track_ref if order_track else None,
        "auroc": auroc,
        "per_fault_auroc": per_fault,
        "threshold_val95": thr,
        "target_norm_fpr": target_norm_fpr,
        "n": {"val": int(len(va_w)), "target_norm": int(tgt.sum()), "fault": int(flt.sum())},
        "val_thr": thr,
    }


def eval_seed(fold, seed, device, batch_size):
    """한 (fold, seed) checkpoint → raw + ot_nominal 재추론. inst는 seed2026에서 이미 nominal과 동치 확인해 생략."""
    ckpt_path = os.path.join(RESULTS_ROOT, f"E_raw_023to1_{fold}_s{seed}", "model.pth")
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["paderborn_metadata"]
    assert not bool(meta["use_meta"]), "no-meta 전용 평가"
    raw = eval_arm(ckpt, meta, device, batch_size, order_track=False, order_track_ref="nominal")
    ot = eval_arm(ckpt, meta, device, batch_size, True, "nominal", val_thr=raw["val_thr"])
    faults = sorted(set(raw["per_fault_auroc"]) | set(ot["per_fault_auroc"]))
    per_fault = {}
    for f in faults:
        per_fault[f] = {
            "raw": raw["per_fault_auroc"].get(f, {}).get("auroc", float("nan")),
            "ot": ot["per_fault_auroc"].get(f, {}).get("auroc", float("nan")),
            "family": f[:2],
            "n": ot["per_fault_auroc"].get(f, raw["per_fault_auroc"].get(f, {})).get("n"),
        }
    return {
        "seed": seed,
        "raw_auroc": raw["auroc"], "ot_auroc": ot["auroc"],
        "delta": ot["auroc"] - raw["auroc"],
        "raw_target_norm_fpr": raw["target_norm_fpr"], "ot_target_norm_fpr": ot["target_norm_fpr"],
        "n": {"raw": raw["n"], "ot": ot["n"]},
        "per_fault": per_fault,
    }


def _ms(vals):
    a = np.asarray([v for v in vals if v == v], dtype=float)
    return (float(a.mean()), float(a.std())) if len(a) else (float("nan"), float("nan"))


def process_fold(fold, cfg, device, batch_size):
    per_seed = []
    for s in SEEDS:
        r = eval_seed(fold, s, device, batch_size)
        if r is None:
            print(f"  [skip] {fold} s{s} checkpoint 없음", file=sys.stderr)
            continue
        per_seed.append(r)
        print(f"  {fold} s{s}: raw={r['raw_auroc']:.4f} ot={r['ot_auroc']:.4f} Δ={r['delta']:+.4f}",
              file=sys.stderr)
    if not per_seed:
        return None

    raw_m, raw_s = _ms([r["raw_auroc"] for r in per_seed])
    ot_m, ot_s = _ms([r["ot_auroc"] for r in per_seed])
    d_m, d_s = _ms([r["delta"] for r in per_seed])
    rawfpr_m, _ = _ms([r["raw_target_norm_fpr"] for r in per_seed])
    otfpr_m, _ = _ms([r["ot_target_norm_fpr"] for r in per_seed])

    # per-fault: seed 평균 raw/ot/Δ
    all_faults = sorted({f for r in per_seed for f in r["per_fault"]})
    per_fault_agg = {}
    for f in all_faults:
        rr = [r["per_fault"][f]["raw"] for r in per_seed if f in r["per_fault"]]
        oo = [r["per_fault"][f]["ot"] for r in per_seed if f in r["per_fault"]]
        rm, _ = _ms(rr); om, _ = _ms(oo)
        per_fault_agg[f] = {"family": f[:2], "raw_mean": rm, "ot_mean": om, "delta_mean": om - rm,
                            "n_seeds": len(rr)}
    # family 집계
    fam_agg = {}
    for fam in sorted({v["family"] for v in per_fault_agg.values()}):
        rm, _ = _ms([v["raw_mean"] for v in per_fault_agg.values() if v["family"] == fam])
        om, _ = _ms([v["ot_mean"] for v in per_fault_agg.values() if v["family"] == fam])
        fam_agg[fam] = {"raw_mean": rm, "ot_mean": om, "delta_mean": om - rm}

    s2026 = next((r for r in per_seed if r["seed"] == 2026), None)
    return {
        "fold": fold, "target": cfg["target"], "amp_group": cfg["amp"],
        "archive_raw_auroc": cfg["archive_raw_auroc"],
        "n_seeds": len(per_seed),
        "raw_auroc_mean": raw_m, "raw_auroc_std": raw_s,
        "ot_auroc_mean": ot_m, "ot_auroc_std": ot_s,
        "delta_mean": d_m, "delta_std": d_s,
        "raw_target_norm_fpr_mean": rawfpr_m, "ot_target_norm_fpr_mean": otfpr_m,
        "raw_reproduces_archive_s2026": (abs(s2026["raw_auroc"] - cfg["archive_raw_auroc"]) < 0.02) if s2026 else None,
        "per_seed": per_seed,
        "per_fault_agg": per_fault_agg,
        "family_agg": fam_agg,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 256
    print(f"device={device}  seeds={SEEDS}", file=sys.stderr)

    results = {}
    for fold, cfg in FOLDS.items():
        print(f"\n=== {fold} ({cfg['target']}, {cfg['amp']}amp) ===", file=sys.stderr)
        r = process_fold(fold, cfg, device, batch_size)
        if r is not None:
            results[fold] = r

    out_path = os.path.join(OUT_DIR, "phaseB_raw_vs_ot_5seed.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {out_path}", file=sys.stderr)

    # ---- 콘솔 요약 ----
    print("\n" + "=" * 78)
    print(f"작업 E-1 Phase B: raw vs raw+OT (023→1, 5-seed {SEEDS})")
    print("=" * 78)
    for fold, r in results.items():
        print(f"\n[{fold}] target={r['target']}({r['amp_group']}amp)  n_seeds={r['n_seeds']}  "
              f"archive_raw(s2026)={r['archive_raw_auroc']:.3f} "
              f"재현={'OK' if r['raw_reproduces_archive_s2026'] else '확인'}")
        print(f"  overall AUROC:  raw={r['raw_auroc_mean']:.4f}±{r['raw_auroc_std']:.4f}  "
              f"ot={r['ot_auroc_mean']:.4f}±{r['ot_auroc_std']:.4f}  "
              f"Δ={r['delta_mean']:+.4f}±{r['delta_std']:.4f}")
        print(f"  target-normal FPR(val95) mean:  raw={r['raw_target_norm_fpr_mean']:.3f}  "
              f"ot={r['ot_target_norm_fpr_mean']:.3f}")
        print(f"  seed별 Δ:  " + "  ".join(f"s{p['seed']}={p['delta']:+.3f}" for p in r["per_seed"]))
        print(f"  family(seed평균 raw→ot, Δ):  " +
              "  ".join(f"{k}:{v['raw_mean']:.2f}→{v['ot_mean']:.2f}({v['delta_mean']:+.2f})"
                        for k, v in r["family_agg"].items()))
        print(f"  per-fault(seed평균 raw→ot, Δ):")
        for f in sorted(r["per_fault_agg"], key=lambda k: -abs(r['per_fault_agg'][k]['delta_mean'])):
            d = r["per_fault_agg"][f]
            print(f"    {f:6s}({d['family']}):  {d['raw_mean']:.3f} → {d['ot_mean']:.3f}  Δ={d['delta_mean']:+.3f}")


if __name__ == "__main__":
    main()
