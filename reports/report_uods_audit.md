# UODS-VAFDC 데이터 audit (외부 검증 Step 1)

> 브랜치 `claude/external-uods`. 스크립트 `analysis/audit_uods.py` (CPU). inventory: `results/UODS/uods_inventory.json`.
> 데이터 루트 `/home/dyhwang/DCP/Data/UODS-VAFDC` (원본 미수정).

## 결과 요약 — 전 항목 통과

| 항목 | 값 | 판정 |
|---|---|---|
| 총 파일 | 60 | ✅ |
| 총 물리 bearing | 20 | ✅ |
| 각 bearing state {0,1,2} 정확히 1개 | True | ✅ |
| 변수명 = 파일명 stem | 60/60 | ✅ |
| shape 420000×4 double | 60/60 | ✅ |
| 전 값 finite | 60/60 | ✅ |
| family 일관성(prefix ↔ bearing_id→family) | True | ✅ |
| 문제 수 | 0 | ✅ |

문서 §10 sanity checklist 중 데이터 무결성 항목(파일/bearing 수, state 구성, 변수명·shape·finite)을 모두 충족.
누수·split 관련 항목(train/test 교집합, train-normal-only 통계 등)은 Step 3(manifest)·Step 4(loader unit test)에서 검증.

## 구조 (확인됨)

- 파일명 `<family>_<bearing_id>_<state>.mat`, MATLAB 변수명 = 파일명 stem, 배열 `420000×4 double`.
- **col0 = vibration(기본 입력)**, col1=acoustic, col2=speed(Hall), col3=load-cell.
- bearing_id(가운데 숫자)가 물리 bearing. 최종 fault family: 1–5 inner / 6–10 outer / 11–15 ball / 16–20 cage.
  각 bearing = healthy(H_n_0) + developing(state1) + faulty(state2) 3파일.

## Ball no-load confound (핵심)

load-cell(col3) std로 확인 (col 값은 물리 N 단위 raw 아님 — std로 로드/무로드만 구분):

| 그룹 | load-col std |
|---|---|
| ball fault(state 1/2) 최대 | **0.0** (전부 무로드) |
| ball healthy(state 0) 최소 | 0.617 (로드) |
| non-ball healthy(state 0) 최소 | 0.617 (로드) |

→ **ball 결함 기록은 health state와 load가 동시에 바뀐다**(faulty=무로드). healthy는 ball 포함 전부 로드 상태.
따라서 ball AUROC는 순수 결함 signature가 아닐 수 있으므로 **ball / non-ball 분리 집계 필수**(문서 §5). loader는
window metadata에 `family`·`is_ball`·`state`를 보존해 developing-only / faulty-only / family별 분해가 가능하도록 한다.

## col2/col3 주의

speed(col2) 평균 ≈ 0(AC/pulse Hall 신호), load(col3)도 raw 물리량 아님 → **열 평균을 speed/load로 쓰지 않는다.**
운행조건 판정은 파일명 metadata(family·state)로만. 기본 입력은 col0 vibration 단일 채널.
