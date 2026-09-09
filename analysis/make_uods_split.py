# ==============================================================================
# UODS 누수-free bearing-wise split manifest 생성 (Step 3). GPU 불필요.
#
# 입력: Vieira(gama-ufsc/bearing-data-leakage) uored-vafcls split을 catch env에서
#       parquet→JSON 변환한 results/UODS/splits/vieira_splits_decoded.json.
#       (각 split = train_ids 12(3/family) + held_ids 8(2/family), 전 105 split 누수0 검증됨.)
#
# 재구성: 동결 파이프라인이 val-normal을 요구하므로 저자 train-side 3/family 중
#   1개를 val로 deterministic carve(결과·신호 미확인) → 패밀리당 2 train-fit / 1 val / 2 test.
#   - train-fit·val = 해당 bearing의 healthy(H_n_0)만.
#   - test = 저자 held-out 8 bearing의 state 0/1/2 전부.
#   - val carve = sha256(f"{source}|{family}") % 3 (data-blind, 재현 가능).
#
# 사용:
#   conda run -n mtgflow python analysis/make_uods_split.py --source tuning:run_0   # pilot
#   conda run -n mtgflow python analysis/make_uods_split.py --source eval:run_5     # 전량용(개별)
# ==============================================================================
import os
import re
import json
import hashlib
import argparse

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_ROOT = os.path.normpath(os.path.join(_REPO, '..', 'Data', 'UODS-VAFDC'))
_DECODED = os.path.join(_REPO, 'results', 'UODS', 'splits', 'vieira_splits_decoded.json')
_OUT_DIR = os.path.join(_REPO, 'results', 'UODS', 'splits')

FAMILY_OF = {}
for b in range(1, 6):   FAMILY_OF[b] = "inner"
for b in range(6, 11):  FAMILY_OF[b] = "outer"
for b in range(11, 16): FAMILY_OF[b] = "ball"
for b in range(16, 21): FAMILY_OF[b] = "cage"

# family -> (폴더, fault prefix). healthy는 항상 1_Healthy/H_n_0.mat.
FAMILY_DIR = {"inner": ("2_Inner_Race_Faults", "I"),
              "outer": ("3_Outer_Race_Faults", "O"),
              "ball":  ("4_Ball_Faults", "B"),
              "cage":  ("5_Cage_Faults", "C")}


def healthy_file(bid):
    return os.path.join("1_Healthy", f"H_{bid}_0.mat")


def fault_files(bid):
    fam = FAMILY_OF[bid]
    folder, prefix = FAMILY_DIR[fam]
    return [os.path.join(folder, f"{prefix}_{bid}_{s}.mat") for s in (1, 2)]


def carve_val(train_ids, source):
    """저자 train 3/family 중 1개를 deterministic carve → (train_fit, val)."""
    by_fam = {}
    for b in train_ids:
        by_fam.setdefault(FAMILY_OF[b], []).append(b)
    train_fit, val = [], []
    for fam, ids in by_fam.items():
        ids = sorted(ids)
        h = int(hashlib.sha256(f"{source}|{fam}".encode()).hexdigest(), 16)
        pick = ids[h % len(ids)]
        val.append(pick)
        train_fit.extend([x for x in ids if x != pick])
    return sorted(train_fit), sorted(val)


def build_manifest(kind, run, decoded):
    key = run
    entry = decoded[kind][key]
    source = f"vieira_{kind}_{run}"
    train_ids = [int(x) for x in entry["train_ids"]]     # 12, 3/family
    test_ids = [int(x) for x in entry["held_ids"]]       # 8, 2/family (우리 test)
    train_fit, val = carve_val(train_ids, source)        # 8 / 4

    # 파일 목록 (존재 검증)
    def check_exists(rel):
        p = os.path.join(_DATA_ROOT, rel)
        if not os.path.exists(p):
            raise FileNotFoundError(rel)
        return rel
    train_files = [check_exists(healthy_file(b)) for b in train_fit]
    val_files = [check_exists(healthy_file(b)) for b in val]
    test_files = []
    for b in test_ids:
        test_files.append(check_exists(healthy_file(b)))          # state 0
        test_files.extend(check_exists(f) for f in fault_files(b))  # state 1,2

    # 검증
    s_tr, s_va, s_te = set(train_fit), set(val), set(test_ids)
    checks = {
        "train_fit_n": len(train_fit), "val_n": len(val), "test_n": len(test_ids),
        "total_bearings": len(s_tr | s_va | s_te),
        "all_20_covered": (s_tr | s_va | s_te) == set(range(1, 21)),
        "disjoint_train_val": len(s_tr & s_va) == 0,
        "disjoint_train_test": len(s_tr & s_te) == 0,
        "disjoint_val_test": len(s_va & s_te) == 0,
        "train_fit_2_per_family": all(sum(FAMILY_OF[b] == f for b in train_fit) == 2 for f in FAMILY_DIR),
        "val_1_per_family": all(sum(FAMILY_OF[b] == f for b in val) == 1 for f in FAMILY_DIR),
        "test_2_per_family": all(sum(FAMILY_OF[b] == f for b in test_ids) == 2 for f in FAMILY_DIR),
        "train_files_all_healthy": all(os.path.basename(f).startswith("H_") for f in train_files),
        "val_files_all_healthy": all(os.path.basename(f).startswith("H_") for f in val_files),
        "test_files_n": len(test_files),  # 8 bearing × 3 state = 24
    }
    manifest = dict(
        source=source, kind=kind, run=run,
        carve_method="sha256(source|family)%3 (data-blind)",
        train_fit_ids=train_fit, val_ids=val, test_ids=sorted(test_ids),
        family_map={str(b): FAMILY_OF[b] for b in range(1, 21)},
        train_fit_files=train_files, val_files=val_files, test_files=test_files,
        vieira_train_ids=sorted(train_ids), vieira_held_ids=sorted(test_ids),
        checks=checks,
    )
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tuning:run_0", help="kind:run  예) tuning:run_0, eval:run_5")
    ap.add_argument("--decoded", default=_DECODED)
    args = ap.parse_args()

    with open(args.decoded) as f:
        decoded = json.load(f)
    kind, run = args.source.split(":")
    if kind not in decoded or run not in decoded[kind]:
        raise SystemExit(f"source {args.source} 없음. tuning: {list(decoded['tuning'])}, eval 예: run_5..run_104")

    man = build_manifest(kind, run, decoded)
    os.makedirs(_OUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUT_DIR, f"uods_split_{kind}_{run}.json")
    with open(out_path, "w") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)

    print(f"=== UODS split manifest: {man['source']} ===")
    print(f"  train_fit({len(man['train_fit_ids'])}): {man['train_fit_ids']}")
    print(f"  val      ({len(man['val_ids'])}): {man['val_ids']}")
    print(f"  test     ({len(man['test_ids'])}): {man['test_ids']}")
    print("  checks:")
    for k, v in man["checks"].items():
        flag = "" if (not isinstance(v, bool) or v) else "  <-- FAIL"
        print(f"    {k}: {v}{flag}")
    print(f"-> {out_path}")
