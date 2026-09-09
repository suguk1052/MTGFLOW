"""UODS 전량(eval) 러너 생성기. GPU 불필요.

split run_5..run_104 × {proposed, raw}에 대해 train 러너 + 짝 dump("test") 러너를 생성한다.
run_name·경로는 split·model·seed로 유일: ckpt results/UODS/uods_eval_run<i>_<model>_s<seed>/,
dump uods_window_scores/uods_eval_run<i>_<model>_s<seed>.npz, log <jobname>_<jobid>.out.
동결 하이퍼 고정: window2048/stride1024/fs42000/n_blocks2/batch256, seeds 2024~2028.
  proposed = --amp_branch --amp_n_bands 6 --amp_normalize --amp_band_scheme log
  raw      = amp 인자 없음(flow_NLL)

사용: conda run -n mtgflow python runners/UODS/eval/gen_eval_runners.py
"""
import os
import stat

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EVAL_DIR = os.path.join(PROJECT_ROOT, "runners", "UODS", "eval")
SPLITS_DIR_REL = "results/UODS/splits"   # MTGFLOW 기준 상대(러너는 cd MTGFLOW 후 실행)
SEEDS = "2024 2025 2026 2027 2028"
SEED_LIST = [2024, 2025, 2026, 2027, 2028]
RUNS = list(range(5, 105))   # eval run_5 .. run_104 (100 split)

TRAIN_COMMON = (
    "    --n_blocks=2 \\\n"
    "    --batch_size=256 \\\n"
    "    --window_size=2048 \\\n"
    "    --stride_size=1024 \\\n"
    "    --sampling_rate=42000 \\\n"
    f"    --seeds {SEEDS}"
)


def train_script(i, model):
    run_name = f"uods_eval_run{i}_{model}"
    manifest = f"{SPLITS_DIR_REL}/uods_split_eval_run_{i}.json"
    amp = (" \\\n    --amp_normalize \\\n    --amp_branch \\\n    --amp_n_bands 6 \\\n    --amp_band_scheme log"
           if model == "proposed" else "")
    return (
        "#!/bin/bash\n"
        f"# UODS 전량 eval run_{i} {model} — train(5 seed). 동결 하이퍼(재선택 없음).\n"
        f'RUN_NAME="{run_name}"\n\n'
        "CUDA_VISIBLE_DEVICES=0 python3 main.py \\\n"
        "    --name=uods \\\n"
        '    --run_name="$RUN_NAME" \\\n'
        f"    --uods_split_manifest {manifest} \\\n"
        f"{TRAIN_COMMON}{amp}\n"
    )


def dump_script(i, model):
    run_name = f"uods_eval_run{i}_{model}"
    lines = [
        "#!/bin/bash",
        f"# UODS 전량 eval run_{i} {model} — 5 seed dump(GPU forward-only). slurm_run.sh test 슬롯.",
        "for SEED in " + " ".join(str(s) for s in SEED_LIST) + "; do",
        "    CUDA_VISIBLE_DEVICES=0 python3 analysis/dump_uods_window_scores.py \\",
        f"        --run_name {run_name}_s${{SEED}} \\",
        "        --overwrite",
        "done",
        "",
    ]
    return "\n".join(lines)


def write(path, content):
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)


def main():
    n = 0
    train_rel = {"proposed": [], "raw": []}
    for i in RUNS:
        d = os.path.join(EVAL_DIR, f"run{i}")
        os.makedirs(d, exist_ok=True)
        for model in ("proposed", "raw"):
            tr = os.path.join(d, f"run_UODS_eval_run{i}_{model}_5seeds.sh")
            te = os.path.join(d, f"run_UODS_eval_run{i}_{model}_test_5seeds.sh")
            write(tr, train_script(i, model))
            write(te, dump_script(i, model))
            n += 2
            train_rel[model].append(os.path.relpath(tr, PROJECT_ROOT))
    # 노드 분배 리스트(짝수 split→n17, 홀수 split→n16). 각 모델 섞어 100잡/노드.
    n17, n16 = [], []
    for model in ("proposed", "raw"):
        for i, rel in zip(RUNS, train_rel[model]):
            (n17 if i % 2 == 0 else n16).append(rel)
    with open(os.path.join(EVAL_DIR, "n17_train_scripts.txt"), "w") as f:
        f.write("\n".join(n17) + "\n")
    with open(os.path.join(EVAL_DIR, "n16_train_scripts.txt"), "w") as f:
        f.write("\n".join(n16) + "\n")
    print(f"generated {n} runner scripts under {EVAL_DIR}")
    print(f"train scripts: proposed {len(train_rel['proposed'])} + raw {len(train_rel['raw'])} = {len(train_rel['proposed'])+len(train_rel['raw'])}")
    print(f"node split: n17 {len(n17)} jobs, n16 {len(n16)} jobs")


if __name__ == "__main__":
    main()
