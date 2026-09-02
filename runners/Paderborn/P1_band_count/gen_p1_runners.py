#!/usr/bin/env python3
"""P-1(Band count N sweep) runner 생성기.

작업 G-3a의 5-seed fold별 train/test .sh를 소스로 삼아(split·bearing id·amp 인자 그대로
재사용, 전사 오류 방지) `--amp_n_bands N`만 바꿔 P-1 24-fold 스크립트를 생성한다.
G-3a(5seeds) 대비 변경점만:
  - RUN_NAME: g3a_<split>_LONO<n>  →  p1_n{N}_<split>_LONO<n>
  - --amp_n_bands 6  →  --amp_n_bands {N}
  - --seeds 2024 2025 2027 2028  →  --seeds 2024 2025 2026 2027 2028
    (P-G1: 스크리닝 없이 처음부터 5-seed 전량을 한 잡에서 학습)
  - 헤더 주석 교체

N=1(=G-1)·N=6(=G-3a)은 재학습 불필요(기존 체크포인트 재사용) → 이 생성기는 신규 재학습
대상 N ∈ {2,3,4,8,12,24}에만 쓴다.

사용:
  python gen_p1_runners.py 2                 # N=2, 24 fold → P1_band_count/n2/
  python gen_p1_runners.py 2 --splits 023to1 --lonos 1   # sanity: 1 fold만
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))          # runners/Paderborn/P1_band_count
RUNNERS = os.path.dirname(os.path.dirname(HERE))           # runners
SRC_DIR = os.path.join(os.path.dirname(HERE), "LONO_G3a_5seeds")  # runners/Paderborn/LONO_G3a_5seeds
ALL_SPLITS = ["123to0", "023to1", "013to2", "012to3"]
ALL_LONOS = [1, 2, 3, 4, 5, 6]


def transform(text, n_bands, split, lono):
    src_run = f"g3a_{split}_LONO{lono}"
    dst_run = f"p1_n{n_bands}_{split}_LONO{lono}"
    assert f'RUN_NAME="{src_run}"' in text, f"소스에 RUN_NAME={src_run} 없음"
    text = text.replace(f'RUN_NAME="{src_run}"', f'RUN_NAME="{dst_run}"')
    # 헤더 주석(작업 G-3a … 줄) 교체. 두 줄 주석 블록을 한 줄로 치환.
    text = text.replace(
        "# 작업 G-3a: K=6 band log-RMS amplitude(dual-branch disentangle 확장). 파일명 _5seeds는 slurm_run.sh\n"
        "# 짝 매칭(_5seeds->_test_5seeds) 규약을 위한 토큰. 2026 재사용, 나머지 4 seed(2024/2025/2027/2028) 추가 실행 → 최종 5-seed.",
        f"# 작업 P-1(Band count N sweep): --amp_n_bands {n_bands}, 나머지 G-3a 동일. 파일명 _5seeds는\n"
        "# slurm_run.sh 짝 매칭(_5seeds->_test_5seeds) 규약 토큰. P-G1: 처음부터 5-seed 전량(2024~2028) 한 잡.")
    # seeds: G-3a는 2026 별도 게이트였으나 P-1은 5-seed 전량 → 2026 포함.
    assert "--seeds 2024 2025 2027 2028" in text, "예상한 seeds 라인이 없음"
    text = text.replace("--seeds 2024 2025 2027 2028", "--seeds 2024 2025 2026 2027 2028")
    # amp_n_bands: 마지막 줄(백슬래시 없음) 6 → N.
    assert text.rstrip().endswith("--amp_n_bands 6"), "예상한 마지막 --amp_n_bands 6 라인이 없음"
    text = text.rstrip()[: -len("6")] + f"{n_bands}\n"
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_bands", type=int, help="amp band 수 N (신규 재학습 대상: 2,3,4,8,12,24)")
    ap.add_argument("--splits", nargs="+", default=ALL_SPLITS, choices=ALL_SPLITS)
    ap.add_argument("--lonos", type=int, nargs="+", default=ALL_LONOS)
    args = ap.parse_args()

    out_dir = os.path.join(HERE, f"n{args.n_bands}")
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
                out_text = transform(text, args.n_bands, split, lono)
                dst = os.path.join(out_dir, f"run_Paderborn_p1_n{args.n_bands}_{split}_LONO{lono}{kind}_5seeds.sh")
                with open(dst, "w") as f:
                    f.write(out_text)
                n += 1
    print(f"생성 완료: {n}개 .sh (N={args.n_bands}, splits={args.splits}, lonos={args.lonos}) → {out_dir}")


if __name__ == "__main__":
    main()
