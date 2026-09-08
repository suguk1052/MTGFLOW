#!/usr/bin/env python3
"""P-2(Band partition scheme sweep) runner 생성기.

작업 G-3a의 5-seed fold별 train/test .sh를 소스로 삼아(split·bearing id·amp 인자 그대로 재사용,
전사 오류 방지) **밴드 수는 N\\*=6으로 고정**하고 `--amp_band_scheme <scheme>`만 추가해
P-2 24-fold 스크립트를 생성한다. G-3a(5seeds) 대비 변경점만:
  - RUN_NAME: g3a_<split>_LONO<n>  →  p2_<scheme>_<split>_LONO<n>
  - 끝에 `--amp_band_scheme <scheme>` 추가(--amp_n_bands 6 유지)
  - --seeds 2024 2025 2027 2028  →  --seeds 2024 2025 2026 2027 2028  (P-G1: 처음부터 5-seed 전량)
  - 헤더 주석 교체

linear(현행)은 **P-1 N=6 캐시(g3a_window_scores) 재사용 → 재학습 불필요**. 이 생성기는 신규 학습
대상 scheme(기본 log·energy)에만 쓴다. (cond·physical은 이번 실행 제외.)

사용:
  python gen_p2_runners.py energy                # energy 24 fold → P2_band_partition/energy/
  python gen_p2_runners.py log --splits 023to1 --lonos 1   # sanity: 1 fold만
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))          # runners/Paderborn/P2_band_partition
SRC_DIR = os.path.join(os.path.dirname(HERE), "LONO_G3a_5seeds")  # runners/Paderborn/LONO_G3a_5seeds
ALL_SPLITS = ["123to0", "023to1", "013to2", "012to3"]
ALL_LONOS = [1, 2, 3, 4, 5, 6]
NEW_SCHEMES = ["log", "energy"]  # linear는 재사용, cond/physical 제외


def transform(text, scheme, split, lono):
    src_run = f"g3a_{split}_LONO{lono}"
    dst_run = f"p2_{scheme}_{split}_LONO{lono}"
    assert f'RUN_NAME="{src_run}"' in text, f"소스에 RUN_NAME={src_run} 없음"
    text = text.replace(f'RUN_NAME="{src_run}"', f'RUN_NAME="{dst_run}"')
    # 헤더 주석(작업 G-3a … 2줄) 교체
    text = text.replace(
        "# 작업 G-3a: K=6 band log-RMS amplitude(dual-branch disentangle 확장). 파일명 _5seeds는 slurm_run.sh\n"
        "# 짝 매칭(_5seeds->_test_5seeds) 규약을 위한 토큰. 2026 재사용, 나머지 4 seed(2024/2025/2027/2028) 추가 실행 → 최종 5-seed.",
        f"# 작업 P-2(Band partition scheme): --amp_n_bands 6 고정 + --amp_band_scheme {scheme}, 나머지 G-3a 동일.\n"
        "# 파일명 _5seeds는 slurm_run.sh 짝 매칭(_5seeds->_test_5seeds) 규약 토큰. P-G1: 처음부터 5-seed 전량(2024~2028) 한 잡.")
    # seeds: G-3a는 2026 별도였으나 P-2는 5-seed 전량 → 2026 포함.
    assert "--seeds 2024 2025 2027 2028" in text, "예상한 seeds 라인이 없음"
    text = text.replace("--seeds 2024 2025 2027 2028", "--seeds 2024 2025 2026 2027 2028")
    # 마지막 줄 `--amp_n_bands 6` 뒤에 `--amp_band_scheme <scheme>` 추가(6은 유지).
    assert text.rstrip().endswith("--amp_n_bands 6"), "예상한 마지막 --amp_n_bands 6 라인이 없음"
    text = text.rstrip() + f" \\\n    --amp_band_scheme {scheme}\n"
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scheme", choices=NEW_SCHEMES, help="band 분할 방식(신규 학습 대상)")
    ap.add_argument("--splits", nargs="+", default=ALL_SPLITS, choices=ALL_SPLITS)
    ap.add_argument("--lonos", type=int, nargs="+", default=ALL_LONOS)
    args = ap.parse_args()

    out_dir = os.path.join(HERE, args.scheme)
    os.makedirs(out_dir, exist_ok=True)

    n = 0
    for split in args.splits:
        for lono in args.lonos:
            for kind in ("", "_test"):
                src = os.path.join(SRC_DIR, f"run_Paderborn_g3a_{split}_LONO{lono}{kind}_5seeds.sh")
                if not os.path.exists(src):
                    raise FileNotFoundError(src)
                with open(src) as f:
                    text = f.read()
                out_text = transform(text, args.scheme, split, lono)
                dst = os.path.join(out_dir, f"run_Paderborn_p2_{args.scheme}_{split}_LONO{lono}{kind}_5seeds.sh")
                with open(dst, "w") as f:
                    f.write(out_text)
                n += 1
    print(f"생성 완료: {n}개 .sh (scheme={args.scheme}, splits={args.splits}, lonos={args.lonos}) → {out_dir}")


if __name__ == "__main__":
    main()
