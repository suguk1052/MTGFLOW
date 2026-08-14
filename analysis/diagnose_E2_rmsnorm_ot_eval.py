# ==============================================================================
# 작업 E 2단계 — OT × RMS-normalization 2×2 재추론 평가 (023→1 저속 LOSO)
#
# 핵심 질문: 저진폭 target(LONO2)에서 robust했던 raw+OT 개선이, 진폭 confound를 제거한
#   RMS-normalized shape 표현(작업 D amp_normalize)에서도 유지/추가 이득을 주는가.
#
# 4 arm:
#   (1) raw          = E-1 raw checkpoint, order_track OFF
#   (2) raw+OT       = 동일 raw checkpoint, order_track ON  (023→1 source=1500rpm 단일 → OT는
#                       train-identity이므로 한 checkpoint가 raw·raw+OT 양쪽 모델; OT는 test-time)
#   (3) rmsnorm      = amp_normalize checkpoint, order_track OFF
#   (4) rmsnorm+OT   = 동일 rmsnorm checkpoint, order_track ON  (동일 identity 논리 → 학습 불필요)
#
# checkpoint 경로:
#   raw:     results/Paderborn/E_raw_023to1_{fold}_s{seed}/model.pth               (E-1)
#   rmsnorm: seed 2026 → results/Paderborn/LONO_D_ampnorm_s2026/ampnorm_023to1_{fold}_s2026/model.pth (D)
#            그 외    → results/Paderborn/E2_rmsnorm_023to1_{fold}_s{seed}/model.pth (E-2 신규)
#
# 산출: 4 arm overall/per-fault AUROC · val95 정상 FPR · paired Δ 4종
#   (OT|raw, OT|rmsnorm[핵심], rmsnorm-raw, interaction) · seed별 Δ · family · per-fault.
# 재현 게이트:
#   raw off s2026     → 아카이브 B3 (LONO1 0.088 / LONO2 0.581)
#   rmsnorm off s2026 → 작업 D flow_only (LONO1 0.887 / LONO2 0.575)
#
# eval_arm/flow_nll/build_loader/build_model는 diagnose_E_order_tracking_eval.py와 동일 패턴.
# eval_arm은 checkpoint의 paderborn_metadata['amp_normalize']를 읽어 rmsnorm을 자동 재현한다.
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
D_DIR = os.path.join(RESULTS_ROOT, "LONO_D_ampnorm_s2026")

THR_PCT = 95
SEEDS = [2024, 2025, 2026, 2027, 2028]
HIGHLIGHT_FAULTS = ["KI17", "KI18", "KI21", "KA15", "KA22"]

# fold → 메타. archive_raw = B3 seed2026 raw AUROC / d_rms_flow = 작업 D seed2026 rmsnorm flow_only.
FOLDS = {
    "LONO1": {"target": "K001", "amp": "high", "archive_raw": 0.0876, "d_rms_flow": 0.8866},
    "LONO2": {"target": "K002", "amp": "low",  "archive_raw": 0.5808, "d_rms_flow": 0.5755},
}


def raw_ckpt_path(fold, seed):
    return os.path.join(RESULTS_ROOT, f"E_raw_023to1_{fold}_s{seed}", "model.pth")


def rms_ckpt_path(fold, seed):
    if seed == 2026:  # 작업 D 체크포인트 재사용
        return os.path.join(D_DIR, f"ampnorm_023to1_{fold}_s{seed}", "model.pth")
    return os.path.join(RESULTS_ROOT, f"E2_rmsnorm_023to1_{fold}_s{seed}", "model.pth")


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
    # amp_normalize는 meta에서 읽어 자동 반영(rmsnorm checkpoint면 True). order_track만 인자로 토글.
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


def eval_arm(ckpt, meta, device, batch_size, order_track, order_track_ref="nominal", val_thr=None):
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
        per_fault[fid] = {"n": int(m.sum()), "family": fid[:2], "auroc": safe_auroc(labels, scores)}

    # threshold = 해당 arm 자신의 val 정상 95p (source=1500rpm이라 OT off/on 간 val 불변 → off의 thr 공유)
    thr = val_thr if val_thr is not None else (
        float(np.percentile(va_f, THR_PCT)) if len(va_f) else float("inf"))
    target_norm_fpr = float(np.mean(tgt_f >= thr)) if len(tgt_f) else float("nan")

    return {
        "auroc": auroc,
        "per_fault_auroc": per_fault,
        "target_norm_fpr": target_norm_fpr,
        "val_thr": thr,
        "n": {"val": int(len(va_w)), "target_norm": int(tgt.sum()), "fault": int(flt.sum())},
    }


def eval_repr(ckpt_path, device, batch_size):
    """한 checkpoint(raw 또는 rmsnorm) → order_track off/on 두 arm 재추론."""
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["paderborn_metadata"]
    assert not bool(meta["use_meta"]), "no-meta 전용 평가"
    off = eval_arm(ckpt, meta, device, batch_size, order_track=False)
    on = eval_arm(ckpt, meta, device, batch_size, order_track=True, val_thr=off["val_thr"])
    return {"off": off, "on": on, "amp_normalize": bool(meta.get("amp_normalize", False))}


def eval_seed(fold, seed, device, batch_size):
    raw = eval_repr(raw_ckpt_path(fold, seed), device, batch_size)
    rms = eval_repr(rms_ckpt_path(fold, seed), device, batch_size)
    if raw is None or rms is None:
        missing = []
        if raw is None:
            missing.append(f"raw({raw_ckpt_path(fold, seed)})")
        if rms is None:
            missing.append(f"rms({rms_ckpt_path(fold, seed)})")
        return None, missing
    assert not raw["amp_normalize"], "raw checkpoint인데 amp_normalize=True"
    assert rms["amp_normalize"], "rmsnorm checkpoint인데 amp_normalize=False"

    arms = {
        "raw": raw["off"], "raw_ot": raw["on"],
        "rms": rms["off"], "rms_ot": rms["on"],
    }
    a_auroc = {k: v["auroc"] for k, v in arms.items()}
    a_fpr = {k: v["target_norm_fpr"] for k, v in arms.items()}

    # paired Δ
    deltas = {
        "ot_on_raw": a_auroc["raw_ot"] - a_auroc["raw"],
        "ot_on_rms": a_auroc["rms_ot"] - a_auroc["rms"],
        "rms_effect": a_auroc["rms"] - a_auroc["raw"],
    }
    deltas["interaction"] = deltas["ot_on_rms"] - deltas["ot_on_raw"]

    # per-fault: 4 arm AUROC + 핵심 Δ
    faults = sorted({f for k in arms for f in arms[k]["per_fault_auroc"]})
    per_fault = {}
    for f in faults:
        vals = {k: arms[k]["per_fault_auroc"].get(f, {}).get("auroc", float("nan")) for k in arms}
        per_fault[f] = {
            "family": f[:2],
            **vals,
            "ot_on_raw": vals["raw_ot"] - vals["raw"],
            "ot_on_rms": vals["rms_ot"] - vals["rms"],
            "n": arms["rms_ot"]["per_fault_auroc"].get(
                f, arms["raw"]["per_fault_auroc"].get(f, {})).get("n"),
        }

    return {
        "seed": seed,
        "auroc": a_auroc,
        "target_norm_fpr": a_fpr,
        "delta": deltas,
        "n": {k: arms[k]["n"] for k in arms},
        "per_fault": per_fault,
    }, []


def _ms(vals):
    a = np.asarray([v for v in vals if v == v], dtype=float)
    return (float(a.mean()), float(a.std())) if len(a) else (float("nan"), float("nan"))


def process_fold(fold, cfg, device, batch_size):
    per_seed = []
    for s in SEEDS:
        r, missing = eval_seed(fold, s, device, batch_size)
        if r is None:
            print(f"  [skip] {fold} s{s} checkpoint 없음: {missing}", file=sys.stderr)
            continue
        per_seed.append(r)
        d = r["delta"]
        print(f"  {fold} s{s}: raw={r['auroc']['raw']:.3f} raw+ot={r['auroc']['raw_ot']:.3f} "
              f"rms={r['auroc']['rms']:.3f} rms+ot={r['auroc']['rms_ot']:.3f}  "
              f"| OT|raw={d['ot_on_raw']:+.3f} OT|rms={d['ot_on_rms']:+.3f}", file=sys.stderr)
    if not per_seed:
        return None

    arm_keys = ["raw", "raw_ot", "rms", "rms_ot"]
    arms_agg = {}
    for k in arm_keys:
        m, sd = _ms([r["auroc"][k] for r in per_seed])
        fm, _ = _ms([r["target_norm_fpr"][k] for r in per_seed])
        arms_agg[k] = {"auroc_mean": m, "auroc_std": sd, "target_norm_fpr_mean": fm}

    delta_keys = ["ot_on_raw", "ot_on_rms", "rms_effect", "interaction"]
    deltas_agg = {}
    for k in delta_keys:
        m, sd = _ms([r["delta"][k] for r in per_seed])
        deltas_agg[k] = {"mean": m, "std": sd,
                         "per_seed": {r["seed"]: r["delta"][k] for r in per_seed}}

    # per-fault seed 평균 (4 arm + 핵심 Δ)
    all_faults = sorted({f for r in per_seed for f in r["per_fault"]})
    per_fault_agg = {}
    for f in all_faults:
        rows = [r["per_fault"][f] for r in per_seed if f in r["per_fault"]]
        entry = {"family": f[:2], "n_seeds": len(rows)}
        for k in arm_keys:
            entry[k], _ = _ms([row[k] for row in rows])
        entry["ot_on_raw"] = entry["raw_ot"] - entry["raw"]
        entry["ot_on_rms"] = entry["rms_ot"] - entry["rms"]
        per_fault_agg[f] = entry

    # family 집계 (fault 평균의 평균)
    fam_agg = {}
    for fam in sorted({v["family"] for v in per_fault_agg.values()}):
        entry = {}
        for k in arm_keys:
            entry[k], _ = _ms([v[k] for v in per_fault_agg.values() if v["family"] == fam])
        entry["ot_on_raw"] = entry["raw_ot"] - entry["raw"]
        entry["ot_on_rms"] = entry["rms_ot"] - entry["rms"]
        fam_agg[fam] = entry

    # 재현 게이트 (seed 2026)
    s2026 = next((r for r in per_seed if r["seed"] == 2026), None)
    reproduce = None
    if s2026:
        reproduce = {
            "raw_s2026": s2026["auroc"]["raw"],
            "raw_matches_archive": abs(s2026["auroc"]["raw"] - cfg["archive_raw"]) < 0.02,
            "rms_s2026": s2026["auroc"]["rms"],
            "rms_matches_D_flow": abs(s2026["auroc"]["rms"] - cfg["d_rms_flow"]) < 0.02,
        }

    return {
        "fold": fold, "target": cfg["target"], "amp_group": cfg["amp"],
        "n_seeds": len(per_seed),
        "archive_raw_auroc": cfg["archive_raw"], "d_rms_flow_auroc": cfg["d_rms_flow"],
        "arms": arms_agg,
        "deltas": deltas_agg,
        "reproduce": reproduce,
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

    out_path = os.path.join(OUT_DIR, "phaseE2_2x2_rmsnorm_ot.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {out_path}", file=sys.stderr)

    # ---- 콘솔 요약 ----
    print("\n" + "=" * 84)
    print(f"작업 E-2: OT × RMS-norm 2×2 (023→1, 5-seed {SEEDS})")
    print("=" * 84)
    for fold, r in results.items():
        rep = r["reproduce"] or {}
        print(f"\n[{fold}] target={r['target']}({r['amp_group']}amp)  n_seeds={r['n_seeds']}")
        print(f"  재현: raw_s2026={rep.get('raw_s2026', float('nan')):.3f}"
              f"(archive {r['archive_raw_auroc']:.3f}, {'OK' if rep.get('raw_matches_archive') else 'X'})  "
              f"rms_s2026={rep.get('rms_s2026', float('nan')):.3f}"
              f"(D {r['d_rms_flow_auroc']:.3f}, {'OK' if rep.get('rms_matches_D_flow') else 'X'})")
        a = r["arms"]
        print("  overall AUROC (mean±std):")
        print(f"    raw     = {a['raw']['auroc_mean']:.4f}±{a['raw']['auroc_std']:.4f}"
              f"    raw+OT  = {a['raw_ot']['auroc_mean']:.4f}±{a['raw_ot']['auroc_std']:.4f}")
        print(f"    rms     = {a['rms']['auroc_mean']:.4f}±{a['rms']['auroc_std']:.4f}"
              f"    rms+OT  = {a['rms_ot']['auroc_mean']:.4f}±{a['rms_ot']['auroc_std']:.4f}")
        d = r["deltas"]
        print("  paired Δ (mean±std):")
        print(f"    OT|raw      = {d['ot_on_raw']['mean']:+.4f}±{d['ot_on_raw']['std']:.4f}")
        print(f"    OT|rmsnorm  = {d['ot_on_rms']['mean']:+.4f}±{d['ot_on_rms']['std']:.4f}   ← 핵심")
        print(f"    rmsnorm-raw = {d['rms_effect']['mean']:+.4f}±{d['rms_effect']['std']:.4f}")
        print(f"    interaction = {d['interaction']['mean']:+.4f}±{d['interaction']['std']:.4f}")
        print("  target-normal FPR(val95) mean:  "
              f"raw={a['raw']['target_norm_fpr_mean']:.3f}  raw+OT={a['raw_ot']['target_norm_fpr_mean']:.3f}  "
              f"rms={a['rms']['target_norm_fpr_mean']:.3f}  rms+OT={a['rms_ot']['target_norm_fpr_mean']:.3f}")
        print("  seed별 Δ:")
        print("    OT|raw : " + "  ".join(f"s{s}={v:+.3f}" for s, v in d['ot_on_raw']['per_seed'].items()))
        print("    OT|rms : " + "  ".join(f"s{s}={v:+.3f}" for s, v in d['ot_on_rms']['per_seed'].items()))
        print("  family (seed평균, raw/raw+OT/rms/rms+OT | OT|raw OT|rms):")
        for fam, v in r["family_agg"].items():
            print(f"    {fam}:  {v['raw']:.2f}/{v['raw_ot']:.2f}/{v['rms']:.2f}/{v['rms_ot']:.2f}  "
                  f"| {v['ot_on_raw']:+.2f} {v['ot_on_rms']:+.2f}")
        print("  강조 per-fault (raw/raw+OT/rms/rms+OT | OT|raw OT|rms):")
        for f in HIGHLIGHT_FAULTS:
            if f in r["per_fault_agg"]:
                v = r["per_fault_agg"][f]
                print(f"    {f}({v['family']}):  {v['raw']:.2f}/{v['raw_ot']:.2f}/{v['rms']:.2f}/{v['rms_ot']:.2f}"
                      f"  | {v['ot_on_raw']:+.2f} {v['ot_on_rms']:+.2f}")


if __name__ == "__main__":
    main()
