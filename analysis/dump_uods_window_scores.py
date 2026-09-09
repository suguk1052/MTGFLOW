"""UODS 외부 검증 — per-window score dump (forward-only, 학습 없음). GPU 권장(CPU도 가능).

동결된 build_model/branch_scores/flow_nll(diagnose_G3a_amp_bands)을 재사용해 UODS checkpoint를
재추론한다. checkpoint metadata의 amp_branch로 proposed/raw를 자동 구분:
  - proposed(amp_branch=True) : tr/va/te per-window (S_shape, S_amp) 저장.
  - raw MTGFlow(amp_branch=False): tr/va/te per-window flow_NLL 저장.
UODS 전용 subgroup metadata(family/state/is_ball)를 함께 저장해 developing/faulty·family·ball 분해 지원.

산출: results/UODS/uods_window_scores/<run_name>.npz
사용: conda run -n mtgflow python analysis/dump_uods_window_scores.py --run_name <run> [--out_dir ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))

from diagnose_G3a_amp_bands import build_model, branch_scores, flow_nll  # noqa: E402
from Dataset.uods import loader_UODS_OCC  # noqa: E402

RESULTS_UODS = os.path.join(PROJECT_ROOT, "results", "UODS")
OUT_DIR = os.path.join(RESULTS_UODS, "uods_window_scores")
UODS_DATA_ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "Data", "UODS-VAFDC"))


def build_uods_loader(meta, batch_size):
    import contextlib, io
    manifest = meta["uods_split_manifest"]
    if not os.path.isabs(manifest):
        manifest = os.path.join(PROJECT_ROOT, manifest)
    with contextlib.redirect_stdout(io.StringIO()):
        return loader_UODS_OCC(
            root=UODS_DATA_ROOT,
            manifest=manifest,
            batch_size=batch_size,
            window_size=int(meta["window_size"]),
            stride_size=int(meta["stride_size"]),
            amp_normalize=bool(meta.get("amp_normalize", False)),
            amp_normalize_channels=meta.get("amp_normalize_channels", "all"),
            amp_n_bands=int(meta.get("amp_n_bands", 1)),
            amp_band_scheme=meta.get("amp_band_scheme", "linear"),
            amp_band_min_width=int(meta.get("amp_band_min_width", 4)),
            rms_eps=float(meta.get("rms_eps", 1e-8)),
            sampling_rate=float(meta.get("sampling_rate", 42000)),
        )


def _pull(ds):
    return (np.asarray(ds.windows, dtype=np.float32),
            np.asarray(ds.rms_raw, dtype=np.float64),
            np.asarray(ds.rms_z, dtype=np.float32),
            np.asarray(ds.ids),
            np.asarray(ds.label, dtype=int),
            np.asarray(ds.families),
            np.asarray(ds.states, dtype=int),
            np.asarray(ds.is_ball, dtype=bool))


def dump_run(run_name, batch_size, out_dir, overwrite, device):
    ckpt_path = os.path.join(RESULTS_UODS, run_name, "model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    out_path = os.path.join(out_dir, f"{run_name}.npz")
    if os.path.exists(out_path) and not overwrite:
        print(f"[skip] 이미 존재: {out_path}")
        return out_path

    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["paderborn_metadata"]
    assert meta.get("dataset") == "uods", f"UODS checkpoint 아님: {run_name}"
    assert not bool(meta["use_meta"]), f"use_meta=True: {run_name}"
    amp_branch = bool(meta.get("amp_branch", False))
    n_bands = int(meta.get("amp_n_bands", 1))

    tr, va, te, n_sensor = build_uods_loader(meta, batch_size)
    tr_w, tr_rms, tr_z, tr_ids, tr_lab, tr_fam, tr_st, tr_ball = _pull(tr.dataset)
    va_w, va_rms, va_z, va_ids, va_lab, va_fam, va_st, va_ball = _pull(va.dataset)
    te_w, te_rms, te_z, te_ids, te_lab, te_fam, te_st, te_ball = _pull(te.dataset)

    common = dict(
        te_lab=te_lab, te_ids=te_ids, te_rms=te_rms,
        te_fam=te_fam, te_state=te_st, te_isball=te_ball,
        va_ids=va_ids, va_rms=va_rms, tr_ids=tr_ids,
    )
    meta_json = dict(run_name=run_name, source=os.path.basename(meta.get("uods_split_manifest", "")),
                     amp_branch=amp_branch, amp_n_bands=n_bands,
                     amp_band_scheme=meta.get("amp_band_scheme"), amp_band_edges=meta.get("amp_band_edges"),
                     window_size=int(meta["window_size"]), stride_size=int(meta["stride_size"]),
                     sampling_rate=float(meta.get("sampling_rate", 42000)))

    os.makedirs(out_dir, exist_ok=True)
    if amp_branch:
        model = build_model(ckpt, n_sensor, device, amp_branch=True, amp_n_bands=n_bands)
        tr_sh, tr_am = branch_scores(model, tr_w, tr_z, device, n_bands, batch_size)
        va_sh, va_am = branch_scores(model, va_w, va_z, device, n_bands, batch_size)
        te_sh, te_am = branch_scores(model, te_w, te_z, device, n_bands, batch_size)
        meta_json["kind"] = "proposed"
        np.savez_compressed(out_path,
                            te_sh=te_sh, te_am=te_am, va_sh=va_sh, va_am=va_am, tr_sh=tr_sh, tr_am=tr_am,
                            meta_json=json.dumps(meta_json, ensure_ascii=False), **common)
        print(f"[proposed] wrote {out_path}  (te {len(te_sh)} / va {len(va_sh)} / tr {len(tr_sh)}, {n_bands}밴드)")
    else:
        model = build_model(ckpt, n_sensor, device, amp_branch=False, amp_n_bands=1)
        tr_raw = flow_nll(model, tr_w, device, batch_size)
        va_raw = flow_nll(model, va_w, device, batch_size)
        te_raw = flow_nll(model, te_w, device, batch_size)
        meta_json["kind"] = "raw"
        np.savez_compressed(out_path,
                            te_raw=te_raw, va_raw=va_raw, tr_raw=tr_raw,
                            meta_json=json.dumps(meta_json, ensure_ascii=False), **common)
        print(f"[raw] wrote {out_path}  (te {len(te_raw)} / va {len(va_raw)} / tr {len(tr_raw)})")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True, help="results/UODS/<run_name>/model.pth")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_dir", type=str, default=OUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} run={args.run_name}")
    dump_run(args.run_name, args.batch_size, args.out_dir, args.overwrite, device)


if __name__ == "__main__":
    main()
