"""UODS loader CPU unit/sanity check (Step 4). GPU 불필요.

동결 파이프라인 재사용의 누수·무결성 가드를 코드로 검증한다:
  ① 데이터 audit(60/20·state{0,1,2}×1)  ② train/val/test bearing disjoint
  ③ train·val 라벨 0·healthy(state0)만  ④ test 라벨 state 매핑(0→0, 1/2→1)
  ⑤ 밴드 경계 = [1,3,10,32,102,323,1025](window 2048, N=6, log)
  ⑥ window shape/finite·metadata(family/state/is_ball) 보존  ⑦ ball/non-ball 분리 가능
  ⑧ train-fit-only 통계 불변: test 파일을 바꿔도 scaler(=train z 통계)·band 경계 불변(누수 없음)

사용: conda run -n mtgflow python analysis/test_uods_loader.py
"""
import os
import sys
import json
import copy
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from Dataset.uods import loader_UODS_OCC, FAMILY_OF, parse_uods_name  # noqa: E402

MANIFEST = os.path.join(PROJECT_ROOT, "results", "UODS", "splits", "uods_split_tuning_run_0.json")
INVENTORY = os.path.join(PROJECT_ROOT, "results", "UODS", "uods_inventory.json")
WIN, STRIDE = 2048, 1024
EXPECT_EDGES = [1, 3, 10, 32, 102, 323, 1025]

_fails = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        _fails.append(name)


def bearing_ids(ds):
    return set(int(s[1:]) for s in np.unique(ds.ids))  # "b03" -> 3


def build(manifest, **kw):
    tr, va, te, _n = loader_UODS_OCC(
        root=os.path.join(PROJECT_ROOT, "..", "Data", "UODS-VAFDC"),
        manifest=manifest, batch_size=256, window_size=WIN, stride_size=STRIDE,
        amp_normalize=True, amp_n_bands=6, amp_band_scheme="log",
        rms_eps=1e-8, sampling_rate=42000, **kw)
    return tr, va, te


def main():
    print("=== UODS loader unit/sanity ===")
    # ① audit 결과 재확인
    with open(INVENTORY) as f:
        inv = json.load(f)
    cl = inv["checklist"]
    check("① audit 60/20·state{0,1,2}×1",
          cl["total_files_ok"] and cl["total_bearings_ok"] and cl["each_bearing_states_0_1_2_once"],
          f"files={cl['total_files']} bearings={cl['total_bearings']}")

    with open(MANIFEST) as f:
        man = json.load(f)
    tr, va, te = build(man)
    tr_ds, va_ds, te_ds = tr.dataset, va.dataset, te.dataset

    # ② disjoint
    bt, bv, be = bearing_ids(tr_ds), bearing_ids(va_ds), bearing_ids(te_ds)
    check("② train/val/test bearing disjoint",
          not (bt & bv) and not (bt & be) and not (bv & be),
          f"train={sorted(bt)} val={sorted(bv)} test={sorted(be)}")

    # ③ train·val 라벨 0 + healthy(state0)만
    check("③ train 라벨 전부 0", int(np.asarray(tr_ds.label).sum()) == 0)
    check("③ val 라벨 전부 0", int(np.asarray(va_ds.label).sum()) == 0)
    check("③ train state 전부 0(healthy만)", set(np.unique(tr_ds.states).tolist()) == {0})
    check("③ val state 전부 0(healthy만)", set(np.unique(va_ds.states).tolist()) == {0})

    # ④ test 라벨 state 매핑
    lab = np.asarray(te_ds.label, dtype=int)
    st = np.asarray(te_ds.states, dtype=int)
    map_ok = np.all(lab[st == 0] == 0) and np.all(lab[st >= 1] == 1)
    check("④ test 라벨 매핑(state0→0, state1/2→1)", bool(map_ok),
          f"states={sorted(set(st.tolist()))} n_norm={int((lab==0).sum())} n_fault={int((lab==1).sum())}")

    # ⑤ 밴드 경계
    check("⑤ log 밴드 경계 = [1,3,10,32,102,323,1025]",
          list(tr_ds.amp_band_edges) == EXPECT_EDGES, f"{tr_ds.amp_band_edges}")

    # ⑥ window shape/finite + metadata 보존
    x0, l0, i0, m0, z0 = te_ds[0]
    shape_ok = tuple(x0.shape) == (1, WIN, 1)
    finite_ok = bool(np.isfinite(np.asarray(tr_ds.windows)).all() and
                     np.isfinite(np.asarray(te_ds.windows)).all())
    meta_ok = (len(te_ds.families) == len(te_ds.windows) == len(te_ds.states) == len(te_ds.is_ball))
    zshape_ok = tuple(np.asarray(te_ds.rms_z).shape[1:]) == (6,)
    check("⑥ window shape [1,L,1]", shape_ok, f"{tuple(x0.shape)}")
    check("⑥ 모든 window finite", finite_ok)
    check("⑥ per-window metadata(family/state/is_ball) 정렬", meta_ok)
    check("⑥ band z target (N,6)", zshape_ok)

    # ⑦ ball/non-ball 분리
    is_ball = np.asarray(te_ds.is_ball, dtype=bool)
    fams = np.asarray(te_ds.families)
    ball_from_fam = (fams == "ball")
    check("⑦ ball/non-ball 분리 가능(is_ball==family)",
          bool(np.array_equal(is_ball, ball_from_fam)) and is_ball.any() and (~is_ball).any(),
          f"ball_win={int(is_ball.sum())} nonball_win={int((~is_ball).sum())}")

    # ⑧ train-fit-only 통계 불변: test 파일을 절반으로 줄여도 train z 통계·band 경계 불변
    man2 = copy.deepcopy(man)
    man2["test_files"] = man["test_files"][: len(man["test_files"]) // 2]
    tr2, _, te2 = build(man2)
    bm1 = np.asarray(tr.dataset.train_logrms_mean, dtype=float)
    bm2 = np.asarray(tr2.dataset.train_logrms_mean, dtype=float)
    bs1 = np.asarray(tr.dataset.train_logrms_std, dtype=float)
    bs2 = np.asarray(tr2.dataset.train_logrms_std, dtype=float)
    inv_ok = (np.allclose(bm1, bm2) and np.allclose(bs1, bs2)
              and list(tr.dataset.amp_band_edges) == list(tr2.dataset.amp_band_edges)
              and len(te2.dataset.windows) < len(te.dataset.windows))
    check("⑧ train-fit-only 통계 불변(test 바꿔도 band mean/std·edges 동일)", bool(inv_ok),
          f"|Δmean|={np.abs(bm1-bm2).max():.2e} |Δstd|={np.abs(bs1-bs2).max():.2e}")

    print(f"\n결과: {'ALL PASS' if not _fails else 'FAIL -> ' + ', '.join(_fails)}")
    print(f"  train {len(tr_ds.windows)} / val {len(va_ds.windows)} / test {len(te_ds.windows)} windows")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
