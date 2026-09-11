"""B-1 고전 baseline — UODS per-window score dump (IF·OC-SVM × raw·band, CPU, 학습 없음).

UODS는 eval 100 split manifest(results/UODS/splits/uods_split_eval_run_{5..104}.json)를 직접 읽어
로더를 amp_normalize=False, amp_n_bands=6, amp_band_scheme='log' 로 호출한다(PU와 동일 규약).
  (a) raw 2048 = loader.dataset.windows,  (b) 6-log-band z-vec = loader.dataset.rms_z.

산출: results/UODS/b1_{if,ocsvm}_{raw,band}_window_scores/eval_run<idx>_s<seed>.npz
  (uods_window_scores raw 스키마 동형) te_raw, va_raw, tr_raw, te_lab, te_ids, te_rms,
  te_fam, te_state, te_isball, va_ids, va_rms, tr_ids, meta_json.
  UODS train-fit이 작아(≈수천 window) OC-SVM raw도 전량 fit(min(N,n)=n). IF=5 seed, OC-SVM=1(s2024).
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))

from Dataset.uods import loader_UODS_OCC  # noqa: E402
from baselines.if_ocsvm import (  # noqa: E402
    fit_score_if, fit_score_ocsvm, val_threshold_and_fpr, OCSVM_SUBSAMPLE_N,
)

RESULTS_UODS = os.path.join(PROJECT_ROOT, "results", "UODS")
SPLITS_DIR = os.path.join(RESULTS_UODS, "splits")
UODS_DATA_ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "Data", "UODS-VAFDC"))

B1_N_BANDS = 6
B1_BAND_SCHEME = "log"
B1_SEEDS = [2024, 2025, 2026, 2027, 2028]
EVAL_RUNS = list(range(5, 105))  # eval 100 split


def build_b1_loader(manifest, batch_size=256):
    if not os.path.isabs(manifest):
        manifest = os.path.join(PROJECT_ROOT, manifest)
    with contextlib.redirect_stdout(io.StringIO()):
        return loader_UODS_OCC(
            root=UODS_DATA_ROOT, manifest=manifest, batch_size=batch_size,
            window_size=2048, stride_size=1024,
            amp_normalize=False, amp_n_bands=B1_N_BANDS, amp_band_scheme=B1_BAND_SCHEME,
            sampling_rate=42000)


def _pull(ds):
    return (np.asarray(ds.windows, dtype=np.float32),
            np.asarray(ds.rms_raw, dtype=np.float64),
            np.asarray(ds.rms_z, dtype=np.float32),
            np.asarray(ds.ids), np.asarray(ds.label, dtype=int),
            np.asarray(ds.families), np.asarray(ds.states, dtype=int),
            np.asarray(ds.is_ball, dtype=bool))


def _meta_json(run_idx, seed, model, inp, band_edges, extra):
    d = {"run_idx": int(run_idx), "seed": int(seed), "b1_model": model, "b1_input": inp,
         "run_name": f"b1_{model}_{inp}_eval_run{run_idx}_s{seed}",
         "source": f"uods_split_eval_run_{run_idx}.json",
         "amp_branch": False, "kind": "raw",
         "amp_n_bands": B1_N_BANDS, "amp_band_scheme": B1_BAND_SCHEME,
         "amp_band_edges": (list(band_edges) if band_edges is not None else None),
         "window_size": 2048, "stride_size": 1024, "sampling_rate": 42000}
    d.update(extra)
    return d


def _save(out_root, model, inp, run_idx, seed, scores, common, meta_json):
    out_dir = os.path.join(out_root, f"b1_{model}_{inp}_window_scores")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eval_run{run_idx}_s{seed}.npz")
    np.savez_compressed(
        out_path,
        te_raw=scores["te"], va_raw=scores["va"], tr_raw=scores["tr"],
        meta_json=json.dumps(meta_json, ensure_ascii=False), **common)
    return out_path


def dump_run(run_idx, out_root, batch_size, overwrite, seeds):
    t0 = time.time()
    def _exists(model, inp, seed):
        return os.path.exists(os.path.join(
            out_root, f"b1_{model}_{inp}_window_scores", f"eval_run{run_idx}_s{seed}.npz"))
    all_done = (all(_exists("if", inp, s) for inp in ("raw", "band") for s in seeds)
                and all(_exists("ocsvm", inp, 2024) for inp in ("raw", "band")))
    if all_done and not overwrite:
        print(f"[skip] eval_run{run_idx} 전 combo 존재")
        return

    manifest = os.path.join(SPLITS_DIR, f"uods_split_eval_run_{run_idx}.json")
    tr, va, te, _ = build_b1_loader(manifest, batch_size)
    t_load = time.time() - t0
    tr_w, tr_rms, tr_z, tr_ids, _, _, _, _ = _pull(tr.dataset)
    va_w, va_rms, va_z, va_ids, _, _, _, _ = _pull(va.dataset)
    te_w, te_rms, te_z, te_ids, te_lab, te_fam, te_st, te_ball = _pull(te.dataset)
    band_edges = getattr(te.dataset, "amp_band_edges", None)
    assert band_edges == [1, 3, 10, 32, 102, 323, 1025], f"band edges 예상과 다름: {band_edges}"

    common = dict(te_lab=te_lab, te_ids=te_ids, te_rms=te_rms,
                  te_fam=te_fam, te_state=te_st, te_isball=te_ball,
                  va_ids=va_ids, va_rms=va_rms, tr_ids=tr_ids)
    inputs = {"raw": (tr_w, va_w, te_w), "band": (tr_z, va_z, te_z)}
    log = {"run_idx": run_idx, "t_load": round(t_load, 1),
           "n_tr": int(len(tr_w)), "n_va": int(len(va_w)), "n_te": int(len(te_w)), "combos": {}}

    for inp, (Xtr, Xva, Xte) in inputs.items():
        for seed in seeds:
            sc = fit_score_if(Xtr, Xva, Xte, seed=seed)
            thr, vfpr = val_threshold_and_fpr(sc["va"])
            mj = _meta_json(run_idx, seed, "if", inp, band_edges,
                            {"n_estimators": 100, "n_fit": sc["n_fit"], "val_thr95": thr, "val_fpr": vfpr})
            _save(out_root, "if", inp, run_idx, seed, sc, common, mj)
        cap = OCSVM_SUBSAMPLE_N if inp == "raw" else None
        sc = fit_score_ocsvm(Xtr, Xva, Xte, subsample_cap=cap)
        thr, vfpr = val_threshold_and_fpr(sc["va"])
        mj = _meta_json(run_idx, 2024, "ocsvm", inp, band_edges,
                        {"nu": 0.05, "gamma": "scale", "n_fit": sc["n_fit"],
                         "n_support": sc["n_support"], "subsample_cap": cap,
                         "val_thr95": thr, "val_fpr": vfpr})
        _save(out_root, "ocsvm", inp, run_idx, 2024, sc, common, mj)
        log["combos"][f"ocsvm_{inp}"] = {"n_fit": sc["n_fit"], "n_sv": sc["n_support"], "val_fpr": round(vfpr, 4)}

    log["t_total"] = round(time.time() - t0, 1)
    print(json.dumps(log, ensure_ascii=False))
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, help="한 split 처리(잡 단위). 미지정시 --run_idxs.")
    ap.add_argument("--run_idxs", type=int, nargs="+", default=EVAL_RUNS)
    ap.add_argument("--seeds", type=int, nargs="+", default=B1_SEEDS)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_root", type=str, default=RESULTS_UODS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    runs = [args.run_idx] if args.run_idx else args.run_idxs
    print(f"B-1 UODS dump: {len(runs)} split, seeds={args.seeds}, out_root={args.out_root}")
    for r in runs:
        dump_run(r, args.out_root, args.batch_size, args.overwrite, args.seeds)


if __name__ == "__main__":
    main()
