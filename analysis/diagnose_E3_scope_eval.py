# ==============================================================================
# 작업 E-3 — order tracking 적용범위 최종 ablation 재추론 평가 (raw vs raw+OT)
#
# 대상: 4 LOSO split(123→0, 023→1, 013→2, 012→3) × LONO1~6 × seed 2024~2028 = 120 fold-seed.
#
# arm 정의 (계획 §2):
#   - raw : B3 체크포인트 `LONO_B3_5seeds/raw_vib_{split}_{lono}_s{seed}` 를 order_track=False 재추론.
#   - raw+OT :
#       * 023→1  : source가 전부 1500rpm이라 train OT가 identity → 위 raw ckpt를 그대로
#                  order_track=True(nominal) 로 재추론(순수 test-time 변환, target 900rpm만 재표현).
#       * 그 외 3 split : target=1500rpm이라 test OT는 identity, raw+OT가 raw와 다른 것은 train의
#                  900rpm source 재표현뿐 → 재추론 불가. `E3_rawOT_{split}_{lono}_s{seed}`(--order_track
#                  켜고 학습) 체크포인트를 order_track=True 로 재추론(scaler가 OT'd source에 fit).
#
# 산출: overall/per-fault AUROC · fault-family(KA/KB/KI) AUROC · val95 정상 FPR · ΔOT=ot−raw.
#   집계 축(단순 24-fold 평균 금지): split / LONO / 진폭군(고 K001·K003·K006 vs 저 K002·K004·K005) / seed.
#   재현 게이트: 023→1 raw arm s2026 overall AUROC = 아카이브 B3(LONO1 0.088 / LONO2 0.581).
#
# build_model/flow_nll/safe_auroc/eval_arm 패턴은 diagnose_E_order_tracking_eval.py 재사용.
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
B3_ROOT = os.path.join(RESULTS_ROOT, "LONO_B3_5seeds")
OUT_DIR = os.path.join(RESULTS_ROOT, "E_order_tracking")

HIGH_AMP = {"K001", "K003", "K006"}
THR_PCT = 95
SEEDS = [2024, 2025, 2026, 2027, 2028]

# split → held-out 조건 / test-time OT 실작동 여부 (계획 §사전조사)
SPLITS = {
    "123to0": {"held_out": "compositional", "test_ot": "identity"},
    "023to1": {"held_out": "speed",         "test_ot": "active"},
    "013to2": {"held_out": "torque",        "test_ot": "identity"},
    "012to3": {"held_out": "force",         "test_ot": "identity"},
}

# LONO → target bearing / 진폭군. 023→1 아카이브 재현값(B3 s2026 raw overall AUROC)만 채움.
LONOS = {
    "LONO1": {"target": "K001", "amp": "high", "archive_raw_023to1": 0.0876},
    "LONO2": {"target": "K002", "amp": "low",  "archive_raw_023to1": 0.5808},
    "LONO3": {"target": "K003", "amp": "high", "archive_raw_023to1": None},
    "LONO4": {"target": "K004", "amp": "low",  "archive_raw_023to1": None},
    "LONO5": {"target": "K005", "amp": "low",  "archive_raw_023to1": None},
    "LONO6": {"target": "K006", "amp": "high", "archive_raw_023to1": None},
}


def raw_ckpt_path(split, lono, seed):
    return os.path.join(B3_ROOT, f"raw_vib_{split}_{lono}_s{seed}", "model.pth")


def ot_ckpt_path(split, lono, seed):
    # 023→1은 train-identity → raw ckpt 재사용(test-time OT). 나머지는 --order_track 학습 전용 ckpt.
    if split == "023to1":
        return raw_ckpt_path(split, lono, seed)
    return os.path.join(RESULTS_ROOT, f"E3_rawOT_{split}_{lono}_s{seed}", "model.pth")


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


def eval_arm(ckpt, meta, device, batch_size, order_track, order_track_ref="nominal"):
    """한 arm(=한 checkpoint + order_track 설정)을 자체 val로 threshold 잡아 평가.
    non-speed split에서 raw(B3)와 OT(E3)는 서로 다른 모델·다른 val 처리라 threshold를 공유하지 않는다.
    (023→1은 source identity라 두 arm val이 동일해 결과적으로 같은 threshold가 나옴)."""
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

    thr = float(np.percentile(va_f, THR_PCT)) if len(va_f) else float("inf")
    target_norm_fpr = float(np.mean(tgt_f >= thr)) if len(tgt_f) else float("nan")

    return {
        "auroc": auroc,
        "per_fault_auroc": per_fault,
        "threshold_val95": thr,
        "target_norm_fpr": target_norm_fpr,
        "n": {"val": int(len(va_w)), "target_norm": int(tgt.sum()), "fault": int(flt.sum())},
    }


def eval_seed(split, lono, seed, device, batch_size):
    raw_p = raw_ckpt_path(split, lono, seed)
    ot_p = ot_ckpt_path(split, lono, seed)
    if not os.path.exists(raw_p):
        return {"seed": seed, "status": "missing_raw", "path": raw_p}
    if not os.path.exists(ot_p):
        return {"seed": seed, "status": "missing_ot", "path": ot_p}

    raw_ck = torch.load(raw_p, map_location=device)
    raw_meta = raw_ck["paderborn_metadata"]
    assert not bool(raw_meta["use_meta"]), "no-meta 전용 평가"
    raw = eval_arm(raw_ck, raw_meta, device, batch_size, order_track=False)

    ot_ck = torch.load(ot_p, map_location=device)
    ot_meta = ot_ck["paderborn_metadata"]
    ot = eval_arm(ot_ck, ot_meta, device, batch_size, order_track=True, order_track_ref="nominal")

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
        "seed": seed, "status": "ok",
        "raw_auroc": raw["auroc"], "ot_auroc": ot["auroc"],
        "delta": ot["auroc"] - raw["auroc"],
        "raw_target_norm_fpr": raw["target_norm_fpr"], "ot_target_norm_fpr": ot["target_norm_fpr"],
        "raw_thr": raw["threshold_val95"], "ot_thr": ot["threshold_val95"],
        "n": {"raw": raw["n"], "ot": ot["n"]},
        "per_fault": per_fault,
    }


def _ms(vals):
    a = np.asarray([v for v in vals if v == v], dtype=float)
    return (float(a.mean()), float(a.std())) if len(a) else (float("nan"), float("nan"))


def process_fold(split, lono, device, batch_size):
    scfg, lcfg = SPLITS[split], LONOS[lono]
    per_seed_ok, missing = [], []
    for s in SEEDS:
        r = eval_seed(split, lono, s, device, batch_size)
        if r.get("status") != "ok":
            missing.append(r)
            print(f"  [skip] {split} {lono} s{s}: {r.get('status')} ({r.get('path')})", file=sys.stderr)
            continue
        per_seed_ok.append(r)
        print(f"  {split} {lono} s{s}: raw={r['raw_auroc']:.4f} ot={r['ot_auroc']:.4f} Δ={r['delta']:+.4f}",
              file=sys.stderr)
    if not per_seed_ok:
        return {"split": split, "lono": lono, "held_out": scfg["held_out"], "test_ot": scfg["test_ot"],
                "target": lcfg["target"], "amp_group": lcfg["amp"], "n_seeds": 0, "missing": missing}

    raw_m, raw_s = _ms([r["raw_auroc"] for r in per_seed_ok])
    ot_m, ot_s = _ms([r["ot_auroc"] for r in per_seed_ok])
    d_m, d_s = _ms([r["delta"] for r in per_seed_ok])
    rawfpr_m, _ = _ms([r["raw_target_norm_fpr"] for r in per_seed_ok])
    otfpr_m, _ = _ms([r["ot_target_norm_fpr"] for r in per_seed_ok])
    positive_seeds = int(sum(1 for r in per_seed_ok if r["delta"] > 0))

    all_faults = sorted({f for r in per_seed_ok for f in r["per_fault"]})
    per_fault_agg = {}
    for f in all_faults:
        rr = [r["per_fault"][f]["raw"] for r in per_seed_ok if f in r["per_fault"]]
        oo = [r["per_fault"][f]["ot"] for r in per_seed_ok if f in r["per_fault"]]
        rm, _ = _ms(rr); om, _ = _ms(oo)
        per_fault_agg[f] = {"family": f[:2], "raw_mean": rm, "ot_mean": om,
                            "delta_mean": om - rm, "n_seeds": len(rr)}
    fam_agg = {}
    for fam in sorted({v["family"] for v in per_fault_agg.values()}):
        rm, _ = _ms([v["raw_mean"] for v in per_fault_agg.values() if v["family"] == fam])
        om, _ = _ms([v["ot_mean"] for v in per_fault_agg.values() if v["family"] == fam])
        fam_agg[fam] = {"raw_mean": rm, "ot_mean": om, "delta_mean": om - rm}

    # 재현 게이트 (023→1만): raw arm s2026 overall = 아카이브 B3
    repro = None
    if split == "023to1" and lcfg["archive_raw_023to1"] is not None:
        s2026 = next((r for r in per_seed_ok if r["seed"] == 2026), None)
        if s2026 is not None:
            repro = abs(s2026["raw_auroc"] - lcfg["archive_raw_023to1"]) < 0.02

    return {
        "split": split, "lono": lono, "held_out": scfg["held_out"], "test_ot": scfg["test_ot"],
        "target": lcfg["target"], "amp_group": lcfg["amp"], "n_seeds": len(per_seed_ok),
        "raw_auroc_mean": raw_m, "raw_auroc_std": raw_s,
        "ot_auroc_mean": ot_m, "ot_auroc_std": ot_s,
        "delta_mean": d_m, "delta_std": d_s, "positive_seeds": positive_seeds,
        "raw_target_norm_fpr_mean": rawfpr_m, "ot_target_norm_fpr_mean": otfpr_m,
        "archive_raw_023to1": lcfg["archive_raw_023to1"],
        "raw_reproduces_archive_s2026": repro,
        "per_seed": per_seed_ok, "missing": missing,
        "per_fault_agg": per_fault_agg, "family_agg": fam_agg,
    }


def _agg_group(folds):
    """fold 리스트(각 5-seed 집계) → 그룹 요약. delta는 fold별 delta_mean을 평균."""
    ok = [f for f in folds if f.get("n_seeds", 0) > 0]
    if not ok:
        return {"n_folds": 0}
    rm, _ = _ms([f["raw_auroc_mean"] for f in ok])
    om, _ = _ms([f["ot_auroc_mean"] for f in ok])
    dm, ds = _ms([f["delta_mean"] for f in ok])
    return {"n_folds": len(ok), "raw_auroc_mean": rm, "ot_auroc_mean": om,
            "delta_mean": dm, "delta_std_across_folds": ds,
            "folds_positive": int(sum(1 for f in ok if f["delta_mean"] > 0))}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 256
    print(f"device={device}  seeds={SEEDS}", file=sys.stderr)

    folds = {}
    for split in SPLITS:
        for lono in LONOS:
            print(f"\n=== {split} {lono} ({LONOS[lono]['target']}, {LONOS[lono]['amp']}amp,"
                  f" test_ot={SPLITS[split]['test_ot']}) ===", file=sys.stderr)
            folds[f"{split}/{lono}"] = process_fold(split, lono, device, batch_size)

    ok_folds = [f for f in folds.values() if f.get("n_seeds", 0) > 0]
    summary = {
        "overall_24fold": _agg_group(ok_folds),
        "by_split": {sp: _agg_group([f for f in ok_folds if f["split"] == sp]) for sp in SPLITS},
        "by_lono": {lo: _agg_group([f for f in ok_folds if f["lono"] == lo]) for lo in LONOS},
        "by_amp": {amp: _agg_group([f for f in ok_folds if f["amp_group"] == amp]) for amp in ("high", "low")},
    }

    out = {"seeds": SEEDS, "folds": folds, "summary": summary}
    out_path = os.path.join(OUT_DIR, "phaseE3_scope_4split.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {out_path}", file=sys.stderr)

    # ---- 콘솔 요약 ----
    print("\n" + "=" * 92)
    print(f"작업 E-3: raw vs raw+OT 적용범위 (4 split × LONO1~6 × 5-seed {SEEDS})")
    print("=" * 92)
    print(f"{'split':8s} {'held_out':13s} {'testOT':9s} {'LONO':6s} {'amp':4s} "
          f"{'raw':>15s} {'raw+OT':>15s} {'ΔOT':>16s} {'+s/5':>5s} {'rawFPR':>7s} {'otFPR':>7s}")
    for key, r in folds.items():
        if r.get("n_seeds", 0) == 0:
            print(f"{r['split']:8s} {r['held_out']:13s} {r['test_ot']:9s} {r['lono']:6s} "
                  f"{r['amp_group']:4s}  (no seeds — missing ckpt)")
            continue
        print(f"{r['split']:8s} {r['held_out']:13s} {r['test_ot']:9s} {r['lono']:6s} {r['amp_group']:4s} "
              f"{r['raw_auroc_mean']:7.4f}±{r['raw_auroc_std']:.3f} "
              f"{r['ot_auroc_mean']:7.4f}±{r['ot_auroc_std']:.3f} "
              f"{r['delta_mean']:+7.4f}±{r['delta_std']:.3f} {r['positive_seeds']:5d} "
              f"{r['raw_target_norm_fpr_mean']:7.3f} {r['ot_target_norm_fpr_mean']:7.3f}")

    print("\n--- 집계 축 ---")
    for name, grp in (("by_split", summary["by_split"]), ("by_lono", summary["by_lono"]),
                      ("by_amp", summary["by_amp"])):
        print(f"[{name}]")
        for k, v in grp.items():
            if v.get("n_folds", 0) == 0:
                print(f"  {k}: (없음)"); continue
            print(f"  {k:14s} n={v['n_folds']:2d}  raw={v['raw_auroc_mean']:.4f} ot={v['ot_auroc_mean']:.4f} "
                  f"ΔOT={v['delta_mean']:+.4f}±{v['delta_std_across_folds']:.3f} "
                  f"fold+={v['folds_positive']}/{v['n_folds']}")
    ov = summary["overall_24fold"]
    if ov.get("n_folds"):
        print(f"[overall_24fold] n={ov['n_folds']} raw={ov['raw_auroc_mean']:.4f} "
              f"ot={ov['ot_auroc_mean']:.4f} ΔOT={ov['delta_mean']:+.4f}±{ov['delta_std_across_folds']:.3f}")

    # 재현 게이트
    print("\n--- 재현 게이트 (023→1 raw arm s2026 = 아카이브 B3) ---")
    for lo in ("LONO1", "LONO2"):
        r = folds.get(f"023to1/{lo}")
        if r and r.get("raw_reproduces_archive_s2026") is not None:
            s2026 = next((p for p in r["per_seed"] if p["seed"] == 2026), None)
            got = s2026["raw_auroc"] if s2026 else float("nan")
            print(f"  023to1 {lo}: raw s2026={got:.4f} vs archive={r['archive_raw_023to1']:.4f} "
                  f"→ {'OK' if r['raw_reproduces_archive_s2026'] else '불일치'}")


if __name__ == "__main__":
    main()
