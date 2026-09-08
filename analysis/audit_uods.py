# ==============================================================================
# UODS-VAFDC 데이터 audit (Step 1). GPU 불필요.
# 목적: 외부 검증 전 데이터 무결성·구조 확인 + inventory JSON 생성.
#   - 60 파일 / 20 물리 bearing
#   - 각 bearing에 state {0,1,2} 정확히 1개씩
#   - 각 .mat: 변수명==파일명 stem, shape 420000x4, 모두 finite
#   - col0=vibration(기본 입력). col3(load) 통계로 ball no-load(≈0) 기록(파일명 family가 판정 근거).
# 사용: conda run -n mtgflow python analysis/audit_uods.py
# ==============================================================================
import os
import re
import json
import numpy as np
import scipy.io

# 데이터 루트: 이 파일(MTGFLOW/analysis/)의 두 단계 상위(MTGFLOW) 옆 ../Data/UODS-VAFDC
_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '..', 'Data', 'UODS-VAFDC'))

# 최종 fault family: 파일명 prefix. 파일명 = <family>_<bearing_id>_<state>.mat
FAMILY_NAME = {"H": "healthy", "I": "inner", "O": "outer", "B": "ball", "C": "cage"}
# bearing_id -> 최종 fault family(문서 §3 표). healthy(H)는 물리 bearing의 정상 기록.
BEARING_FAMILY = {}
for b in range(1, 6):   BEARING_FAMILY[b] = "inner"
for b in range(6, 11):  BEARING_FAMILY[b] = "outer"
for b in range(11, 16): BEARING_FAMILY[b] = "ball"
for b in range(16, 21): BEARING_FAMILY[b] = "cage"

FNAME_RE = re.compile(r'^([HIOBC])_(\d+)_(\d+)$')


def parse_name(stem):
    m = FNAME_RE.match(stem)
    if not m:
        return None
    fam, bid, state = m.group(1), int(m.group(2)), int(m.group(3))
    return fam, bid, state


def audit():
    records = []
    problems = []
    for sub in sorted(os.listdir(_ROOT)):
        subpath = os.path.join(_ROOT, sub)
        if not os.path.isdir(subpath):
            continue
        for f in sorted(os.listdir(subpath)):
            if not f.endswith('.mat'):
                continue
            stem = f[:-4]
            parsed = parse_name(stem)
            if parsed is None:
                problems.append(f"파일명 파싱 실패: {f}")
                continue
            fam, bid, state = parsed
            path = os.path.join(subpath, f)
            mat = scipy.io.loadmat(path)
            keys = [k for k in mat if not k.startswith('__')]
            var_ok = (keys == [stem])
            arr = np.asarray(mat[stem]) if stem in mat else None
            shape_ok = arr is not None and arr.shape == (420000, 4)
            finite_ok = arr is not None and bool(np.isfinite(arr).all())
            load_std = float(arr[:, 3].std()) if arr is not None else None
            vib_std = float(arr[:, 0].std()) if arr is not None else None
            # 최종 fault family 일관성: healthy는 물리 bearing의 fault family로 기록,
            # fault 파일의 prefix와 bearing_id→family 표가 일치해야 함.
            expect_fam = BEARING_FAMILY.get(bid)
            fam_consistent = (fam == "H") or (FAMILY_NAME[fam] == expect_fam)
            if not var_ok:
                problems.append(f"{f}: 변수명 불일치 {keys}")
            if not shape_ok:
                problems.append(f"{f}: shape {None if arr is None else arr.shape} != (420000,4)")
            if not finite_ok:
                problems.append(f"{f}: non-finite 값 존재")
            if not fam_consistent:
                problems.append(f"{f}: family 불일치 prefix={fam} bearing_family={expect_fam}")
            records.append(dict(
                file=f, folder=sub, stem=stem, prefix=fam,
                fault_family=expect_fam, bearing_id=bid, state=state,
                is_ball=(expect_fam == "ball"),
                var_ok=var_ok, shape_ok=shape_ok, finite_ok=finite_ok,
                load_col_std=load_std, vib_col_std=vib_std,
            ))
    return records, problems


def summarize(records, problems):
    bearings = {}
    for r in records:
        bearings.setdefault(r['bearing_id'], {})[r['state']] = r['file']
    # 각 bearing에 state 0/1/2 정확히 1개
    state_ok = all(sorted(v.keys()) == [0, 1, 2] for v in bearings.values())
    checklist = {
        "total_files": len(records),
        "total_files_ok": len(records) == 60,
        "total_bearings": len(bearings),
        "total_bearings_ok": len(bearings) == 20,
        "each_bearing_states_0_1_2_once": state_ok,
        "all_var_name_ok": all(r['var_ok'] for r in records),
        "all_shape_420000x4_ok": all(r['shape_ok'] for r in records),
        "all_finite_ok": all(r['finite_ok'] for r in records),
        "family_consistency_ok": len([p for p in problems if "family 불일치" in p]) == 0,
        "n_problems": len(problems),
    }
    # ball no-load 확인: ball state 1/2 load-col std ≈ 0 vs healthy load-col std > 0
    ball_dev_fault = [r for r in records if r['is_ball'] and r['state'] in (1, 2)]
    ball_healthy = [r for r in records if r['is_ball'] and r['state'] == 0]
    nonball_healthy = [r for r in records if (not r['is_ball']) and r['state'] == 0]
    confound = {
        "ball_fault_load_std_max": max((r['load_col_std'] for r in ball_dev_fault), default=None),
        "ball_healthy_load_std_min": min((r['load_col_std'] for r in ball_healthy), default=None),
        "nonball_healthy_load_std_min": min((r['load_col_std'] for r in nonball_healthy), default=None),
    }
    return checklist, confound, bearings


if __name__ == "__main__":
    records, problems = audit()
    checklist, confound, bearings = summarize(records, problems)

    out_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'UODS'))
    os.makedirs(out_dir, exist_ok=True)
    inv = dict(
        root=_ROOT,
        n_files=len(records), n_bearings=len(bearings),
        checklist=checklist, ball_load_confound=confound,
        bearings={str(k): v for k, v in sorted(bearings.items())},
        records=records, problems=problems,
    )
    out_path = os.path.join(out_dir, 'uods_inventory.json')
    with open(out_path, 'w') as fp:
        json.dump(inv, fp, indent=2, ensure_ascii=False)

    print("=== UODS audit ===")
    print(f"root: {_ROOT}")
    for k, v in checklist.items():
        print(f"  {k}: {v}")
    print("--- ball no-load confound (load-col std) ---")
    for k, v in confound.items():
        print(f"  {k}: {v}")
    if problems:
        print("--- PROBLEMS ---")
        for p in problems:
            print("  !", p)
    else:
        print("문제 없음 (all checks passed)" if all(
            v for kk, v in checklist.items() if kk.endswith('_ok')) else "일부 체크 실패 — 위 checklist 확인")
    print(f"\ninventory -> {out_path}")
