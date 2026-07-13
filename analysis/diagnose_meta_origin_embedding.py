"""
C2(static meta) checkpoint의 meta_encoder가 unseen 조건(특히 held-out setting)에서
시드에 따라 어떻게 다르게 반응하는지 비교하는 진단 스크립트.

배경: LOSO "123to0" split(기준조건 N15_M07_F10 held-out)에서만 static meta(C2)의 시드 간
AUROC가 0.0~1.0으로 요동친다. 원인 가설: static meta 정규화가 기준조건을 원점 [0,0,0]으로
두는데, 이 split의 학습 데이터(setting 1,2,3)에는 세 축이 동시에 0인 지점이 등장하지 않아
meta_encoder가 unseen 원점에서 시드마다 임의의 출력을 낼 수 있다.

GPU 불필요 - 저장된 model.pth를 CPU로 로드해 meta_encoder만 forward한다.

사용 예:
    conda run -n mtgflow python analysis/diagnose_meta_origin_embedding.py \\
        --checkpoints \\
        good_s2026=results/Paderborn/LONO_C2_5seeds/CA_raw_vib_123to0_LONO4_s2026/model.pth \\
        bad_s2027=results/Paderborn/LONO_C2_5seeds/CA_raw_vib_123to0_LONO4_s2027/model.pth
"""
import argparse
import os
import sys

import torch

# analysis/ 스크립트에서 MTGFLOW 루트의 models/Dataset 패키지를 import하기 위한 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.MTGFLOW import MTGFLOW
from Dataset.paderborn import PADERBORN_SETTING_META, get_paderborn_setting_meta


def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    cfg = checkpoint['args']
    metadata = checkpoint.get('paderborn_metadata', {})
    meta_input_dim = metadata.get('meta_input_dim', 3)
    model = MTGFLOW(
        cfg['n_blocks'], cfg['input_size'], cfg['hidden_size'], cfg['n_hidden'],
        cfg['window_size'], 1,
        model=cfg['model'], batch_norm=cfg['batch_norm'],
        use_meta=cfg['use_meta'], meta_input_dim=meta_input_dim, meta_emb_dim=cfg['meta_emb_dim'],
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoints', nargs='+', required=True,
                         help='"label=경로" 형식으로 여러 checkpoint 지정')
    args = parser.parse_args()

    setting_names = list(PADERBORN_SETTING_META.keys())

    for entry in args.checkpoints:
        label, path = entry.split('=', 1)
        model, metadata = load_model(path)
        train_settings = set(metadata.get('train_load_setting') or metadata.get('load_setting') or [])
        print(f"\n=== {label} ({path}) ===")
        print(f"train settings: {sorted(train_settings)}")
        with torch.no_grad():
            for setting_name in setting_names:
                meta = torch.tensor(get_paderborn_setting_meta(setting_name), dtype=torch.float32).unsqueeze(0)
                emb = model.meta_encoder(meta).squeeze(0)
                seen = 'seen' if setting_name in train_settings else 'HELD-OUT'
                print(f"  {setting_name:14s} [{seen:9s}] emb_norm={emb.norm().item():.4f}  emb={emb.numpy().round(3)}")


if __name__ == '__main__':
    main()
