# ==============================================================================
# UODS-VAFDC (University of Ottawa, UORED-VAFCLS) 외부 검증용 OCC 로더.
#
# PU 최종 파이프라인(loader_Paderborn_OCC)의 동결 규칙을 그대로 재사용한다:
#   - scaler(StandardScaler): train-fit healthy만으로 fit → val/test에 transform만.
#   - band log-RMS z-score 통계(band_mean/std): train-fit healthy만으로 계산 → val/test 동일 적용.
#   - 밴드 경계 산출(compute_band_boundaries)·band RMS·z-score 수식: paderborn.py 헬퍼를 import(수식 동일).
# UODS 전용 교체분만 새로 구현: 파일 탐색/파일명 파싱/.mat 읽기(단일 col0)/bearing-wise split/metadata.
#
# 누수 방지: split은 manifest(bearing-wise)로 고정. train/val/test bearing 교집합=0 assert.
#   train·val = 해당 bearing의 healthy(state 0)만. test = test bearing의 state 0/1/2 전부.
#   test bearing은 어떤 fit/정규화/보정에도 사용하지 않는다(scaler·band 통계는 train-fit만).
#
# Fisher-tail 보정·threshold(val-normal 사용)는 PU와 동일하게 추론단(analysis)에서 수행.
# ==============================================================================
import os
import re
import json
import numpy as np
import scipy.io
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# 동결된 수치 헬퍼·데이터셋 클래스를 그대로 재사용(복사 금지, 수식 동일성 보장).
from Dataset.paderborn import (
    compute_band_boundaries,
    compute_band_rms,
    compute_band_ms_per_bin,
    band_rms_from_ms,
    Paderborn_dataset,
    _SUPPORTED_BAND_SCHEMES,
    _ADAPTIVE_BAND_SCHEMES,
)

_DATA_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '..', 'Data', 'UODS-VAFDC'))

FAMILY_OF = {}
for _b in range(1, 6):   FAMILY_OF[_b] = "inner"
for _b in range(6, 11):  FAMILY_OF[_b] = "outer"
for _b in range(11, 16): FAMILY_OF[_b] = "ball"
for _b in range(16, 21): FAMILY_OF[_b] = "cage"

_FNAME_RE = re.compile(r'^([HIOBC])_(\d+)_(\d+)$')


def parse_uods_name(stem):
    """파일명 stem <family>_<bearing_id>_<state> → (prefix, bearing_id:int, state:int)."""
    m = _FNAME_RE.match(stem)
    if not m:
        raise ValueError(f"UODS 파일명 파싱 실패: {stem}")
    return m.group(1), int(m.group(2)), int(m.group(3))


def _load_vibration(path):
    """UODS .mat: 변수명=파일명 stem, 배열 (420000,4). col0=vibration 단일 채널 (T,1) 반환."""
    stem = os.path.basename(path)[:-4]
    mat = scipy.io.loadmat(path)
    if stem not in mat:
        raise ValueError(f"{path}: 변수 {stem} 없음 (keys={[k for k in mat if not k.startswith('__')]})")
    arr = np.asarray(mat[stem], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"{path}: shape {arr.shape} 예상과 다름")
    return arr[:, 0:1]  # (T,1)


def _resolve_manifest(manifest):
    if isinstance(manifest, dict):
        return manifest
    with open(manifest) as f:
        return json.load(f)


def loader_UODS_OCC(root=_DATA_ROOT,
                    manifest=None,
                    batch_size=256,
                    window_size=2048,
                    stride_size=1024,
                    label=False,
                    amp_normalize=False,
                    amp_normalize_channels='all',
                    amp_n_bands=1,
                    amp_band_scheme='linear',
                    amp_band_min_width=4,
                    rms_eps=1e-8,
                    sampling_rate=42000):
    """UODS bearing-wise OCC 로더. 반환 (train_loader, val_loader, test_loader, n_sensor).

    manifest = dict 또는 JSON 경로. 키: train_fit_files·val_files·test_files(root 상대),
    train_fit_ids·val_ids·test_ids(bearing int). band/scaler 통계는 train-fit healthy만으로 fit.
    """
    if manifest is None:
        raise ValueError("loader_UODS_OCC requires a split manifest (dict or path).")
    man = _resolve_manifest(manifest)

    # --- split 무결성 assert (누수 방지 가드; PU가 생략한 명시 검증) ---
    tr_ids = set(int(x) for x in man['train_fit_ids'])
    va_ids = set(int(x) for x in man['val_ids'])
    te_ids = set(int(x) for x in man['test_ids'])
    assert not (tr_ids & va_ids), f"train∩val bearing != 0: {sorted(tr_ids & va_ids)}"
    assert not (tr_ids & te_ids), f"train∩test bearing != 0: {sorted(tr_ids & te_ids)}"
    assert not (va_ids & te_ids), f"val∩test bearing != 0: {sorted(va_ids & te_ids)}"

    amp_n_bands = int(amp_n_bands)
    if amp_n_bands < 1:
        raise ValueError("amp_n_bands must be >=1")
    amp_band_scheme = str(amp_band_scheme)
    if amp_band_scheme not in _SUPPORTED_BAND_SCHEMES:
        raise ValueError(f"amp_band_scheme must be one of {_SUPPORTED_BAND_SCHEMES}, got {amp_band_scheme!r}")
    amp_band_min_width = int(amp_band_min_width)
    _band_F = int(window_size) // 2 + 1
    _band_adaptive = amp_band_scheme in _ADAPTIVE_BAND_SCHEMES
    # 비-adaptive(linear/log)는 데이터 무의존이라 미리 확정. window 2048 → F=1025 → PU와 동일 bin 경계.
    _band_boundaries = (None if (amp_n_bands <= 1 or _band_adaptive)
                        else compute_band_boundaries(_band_F, amp_n_bands, amp_band_scheme,
                                                     min_width=amp_band_min_width))

    def _abs(rel):
        return rel if os.path.isabs(rel) else os.path.join(root, rel)

    train_files = [_abs(f) for f in man['train_fit_files']]
    val_files = [_abs(f) for f in man['val_files']]
    # test 파일은 state로 normal(0) / fault(1,2) 분리(PU와 동일 순서: norm 먼저).
    test_norm_files, test_fault_files = [], []
    for rel in man['test_files']:
        _, _, state = parse_uods_name(os.path.basename(rel)[:-4])
        (test_norm_files if state == 0 else test_fault_files).append(_abs(rel))

    if not train_files:
        raise ValueError("UODS: train-fit 파일이 비었습니다.")

    # --- StandardScaler: train-fit healthy만으로 fit (누수 방지) ---
    raw_train = [_load_vibration(p) for p in train_files]
    scaler = StandardScaler()
    scaler.fit(np.concatenate(raw_train, axis=0))

    def extract_scaled_windows(file_list):
        wins, ids, fams, states, rms_list = [], [], [], [], []
        band_rms_list, band_ms_list = [], []
        for path in file_list:
            stem = os.path.basename(path)[:-4]
            prefix, bid, state = parse_uods_name(stem)
            fam = FAMILY_OF[bid]
            sig = _load_vibration(path)          # (T,1)
            scaled = scaler.transform(sig)       # (T,1)
            start = 0
            while start + window_size <= len(scaled):
                w = scaled[start:start + window_size]     # (win,1)
                rms_vec = np.sqrt(np.mean(w ** 2, axis=0))  # (1,)
                if amp_n_bands > 1:
                    if _band_adaptive:
                        band_ms_list.append(compute_band_ms_per_bin(w[:, 0]))  # (F,)
                    else:
                        band_rms_list.append(compute_band_rms(w[:, 0], _band_boundaries))  # (K,)
                if amp_normalize:
                    # 단일 채널: 'all'/'vib_only' 동치. per-window RMS로 나눠 shape-only.
                    w = w / (rms_vec + rms_eps)
                wins.append(w[:, 0])                       # (win,)
                rms_list.append(float(rms_vec[0]))
                ids.append(f"b{bid:02d}")
                fams.append(fam)
                states.append(int(state))
                start += stride_size

        _feat_width = _band_F if (amp_n_bands > 1 and _band_adaptive) else amp_n_bands
        if not wins:
            return (np.empty((0, window_size)), np.array([], dtype=str),
                    np.array([], dtype=str), np.array([], dtype=int),
                    np.empty((0,), dtype=np.float32), np.empty((0, _feat_width), dtype=np.float64))
        if amp_n_bands > 1 and _band_adaptive:
            band_feat = np.array(band_ms_list, dtype=np.float64)     # (N,F)
        elif amp_n_bands > 1:
            band_feat = np.array(band_rms_list, dtype=np.float32)    # (N,K)
        else:
            band_feat = np.empty((len(wins), amp_n_bands), dtype=np.float32)
        return (np.array(wins), np.array(ids), np.array(fams), np.array(states, dtype=int),
                np.array(rms_list, dtype=np.float32), band_feat)

    train_x, train_ids, train_fam, train_state, train_rms, train_band_rms = extract_scaled_windows(train_files)
    val_x, val_ids, val_fam, val_state, val_rms, val_band_rms = extract_scaled_windows(val_files)
    tn_x, tn_ids, tn_fam, tn_state, tn_rms, tn_band_rms = extract_scaled_windows(test_norm_files)
    tf_x, tf_ids, tf_fam, tf_state, tf_rms, tf_band_rms = extract_scaled_windows(test_fault_files)

    # --- 밴드 경계 확정(비-adaptive는 이미 확정). adaptive(energy)만 train-normal PSD로 산출. ---
    _band_edges_serial = None
    _band_boundaries_used = _band_boundaries
    if amp_n_bands > 1 and _band_adaptive:
        if not len(train_band_rms):
            raise ValueError("energy scheme: train-fit 윈도우가 비어 PSD 경계를 산출할 수 없음.")
        train_psd_mean = np.asarray(train_band_rms, dtype=np.float64).mean(axis=0)
        _band_boundaries_used, _info = compute_band_boundaries(
            _band_F, amp_n_bands, amp_band_scheme, psd=train_psd_mean,
            min_width=amp_band_min_width, return_info=True)
        _band_edges_serial = _info.get('edges')

        def _reduce_ms(ms_arr):
            if not len(ms_arr):
                return np.empty((0, amp_n_bands), dtype=np.float32)
            return np.stack([band_rms_from_ms(row, _band_boundaries_used) for row in ms_arr]).astype(np.float32)
        train_band_rms = _reduce_ms(train_band_rms)
        val_band_rms = _reduce_ms(val_band_rms)
        tn_band_rms = _reduce_ms(tn_band_rms)
        tf_band_rms = _reduce_ms(tf_band_rms)
    elif amp_n_bands > 1 and _band_boundaries_used is not None:
        _g, _ns_info = compute_band_boundaries(
            _band_F, amp_n_bands, amp_band_scheme, min_width=amp_band_min_width, return_info=True)
        _band_edges_serial = (_ns_info["edges"] if _ns_info["edges"] is not None
                              else [int(g[0]) for g in _band_boundaries_used] + [int(_band_F)])

    # test = norm 먼저, fault 다음 (PU와 동일).
    test_x = np.concatenate([tn_x, tf_x], axis=0)
    test_ids = np.concatenate([tn_ids, tf_ids], axis=0)
    test_fam = np.concatenate([tn_fam, tf_fam], axis=0)
    test_state = np.concatenate([tn_state, tf_state], axis=0)
    test_rms = np.concatenate([tn_rms, tf_rms], axis=0)
    test_band_rms = np.concatenate([tn_band_rms, tf_band_rms], axis=0)

    # --- 스칼라 log-RMS z-score (train-fit만으로 fit; 하위호환/게이트용) ---
    def _logsafe(a):
        return np.log(a + rms_eps) if len(a) else a
    train_logrms = _logsafe(train_rms)
    if len(train_logrms):
        train_logrms_mean = float(train_logrms.mean())
        train_logrms_std = float(train_logrms.std())
    else:
        train_logrms_mean, train_logrms_std = 0.0, 1.0

    def _z_scalar(a):
        if not len(a):
            return np.empty((0,), dtype=np.float32)
        return ((_logsafe(a) - train_logrms_mean) / (train_logrms_std + 1e-8)).astype(np.float32)
    train_rms_z = _z_scalar(train_rms)
    val_rms_z = _z_scalar(val_rms)
    test_rms_z = _z_scalar(test_rms)

    # --- band 모드: band별 log-RMS z-score(band_mean/std = train-fit만). PU와 동일. ---
    if amp_n_bands > 1:
        train_logband = _logsafe(train_band_rms)
        val_logband = _logsafe(val_band_rms)
        test_logband = _logsafe(test_band_rms)
        if len(train_logband):
            band_mean = train_logband.mean(axis=0)   # (K,)  train-fit only
            band_std = train_logband.std(axis=0)
        else:
            band_mean = np.zeros(amp_n_bands, dtype=np.float64)
            band_std = np.ones(amp_n_bands, dtype=np.float64)

        def _z_band(a):
            if not len(a):
                return np.empty((0, amp_n_bands), dtype=np.float32)
            return ((a - band_mean) / (band_std + 1e-8)).astype(np.float32)
        train_rms_z = _z_band(train_logband)
        val_rms_z = _z_band(val_logband)
        test_rms_z = _z_band(test_logband)
        train_logrms_mean = band_mean.astype(float).tolist()
        train_logrms_std = band_std.astype(float).tolist()

    # 이진 라벨 (정상 0, 이상 1). train/val = 전부 healthy(0). test = norm(0)+fault(1).
    train_y = np.zeros(len(train_x))
    val_y = np.zeros(len(val_x))
    test_y = np.array([0] * len(tn_x) + [1] * len(tf_x))

    n_sensor = 1
    # meta는 사용 안 함(use_meta off). batch tuple shape 유지를 위해 zeros(N,3).
    train_meta = np.zeros((len(train_x), 3), dtype=np.float32)
    val_meta = np.zeros((len(val_x), 3), dtype=np.float32)
    test_meta = np.zeros((len(test_x), 3), dtype=np.float32)

    print(f'UODS OCC | source={man.get("source")} | train {len(train_x)} / val {len(val_x)} / test {len(test_x)} windows '
          f'(norm {len(tn_x)} + fault {len(tf_x)}), bands={amp_n_bands}/{amp_band_scheme}, edges={_band_edges_serial}')

    train_ds = Paderborn_dataset(train_x, train_y, window_size, train_ids, train_meta, train_rms_z)
    val_ds = Paderborn_dataset(val_x, val_y, window_size, val_ids, val_meta, val_rms_z)
    test_ds = Paderborn_dataset(test_x, test_y, window_size, test_ids, test_meta, test_rms_z)

    for ds, rms_arr, fam_arr, st_arr in (
            (train_ds, train_rms, train_fam, train_state),
            (val_ds, val_rms, val_fam, val_state),
            (test_ds, test_rms, test_fam, test_state)):
        ds.rms_raw = np.asarray(rms_arr, dtype=np.float32)
        ds.train_logrms_mean = train_logrms_mean
        ds.train_logrms_std = train_logrms_std
        ds.amp_normalize = bool(amp_normalize)
        ds.amp_n_bands = int(amp_n_bands)
        ds.amp_band_scheme = str(amp_band_scheme)
        ds.amp_band_edges = (list(_band_edges_serial) if _band_edges_serial is not None else None)
        # UODS 전용 metadata(ball/non-ball·developing/faulty·family별 분해용)
        ds.families = np.asarray(fam_arr)
        ds.states = np.asarray(st_arr, dtype=int)
        ds.is_ball = np.asarray([f == "ball" for f in fam_arr], dtype=bool)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=not label)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_sensor
