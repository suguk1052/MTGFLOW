"""재현성 검증용: 두 run(같은 시드로 재학습되었어야 하는)의 test set raw score 배열과
AUROC을 직접 비교한다. TODO.md §0-(1) 검증 스크립트. main.py/test.py는 수정하지 않고
동일한 로더/모델 구성 로직만 재사용한다.

사용법:
  conda run -n mtgflow python analysis/check_repro_scores.py \
      --run_a <results/Paderborn 기준 run_name A> \
      --run_b <run_name B>
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# analysis/ 아래에 있으므로 MTGFLOW 루트(analysis/의 부모)를 sys.path에 넣어야 Dataset/models를 찾는다.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from Dataset.paderborn import loader_Paderborn_OCC
from models.MTGFLOW import MTGFLOW

parser = argparse.ArgumentParser()
parser.add_argument('--run_a', type=str, required=True)
parser.add_argument('--run_b', type=str, required=True)
parser.add_argument('--results_root', type=str, default='results/Paderborn')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_checkpoint(run_name):
    path = os.path.join(args.results_root, run_name, 'model.pth')
    ckpt = torch.load(path, map_location=device)
    return path, ckpt


def build_model_and_loaders(metadata):
    train_loader, val_loader, test_loader, n_sensor = loader_Paderborn_OCC(
        root=os.path.join(PROJECT_ROOT, '..', 'Data', 'Paderborn'),
        loads=metadata['load_setting'],
        train_loads=metadata.get('train_load_setting'),
        test_loads=metadata.get('test_load_setting'),
        sensor_mode=metadata['sensor_mode'],
        batch_size=256,
        window_size=metadata['window_size'],
        stride_size=metadata['stride_size'],
        train_ids=metadata['train_ids'],
        val_ids=metadata['val_ids'],
        test_norm_ids=metadata['test_norm_ids'],
        exclude_ids=metadata['exclude_ids'],
        meta_source=metadata['meta_source'],
        measured_meta_stats=metadata['measured_meta_stats'],
    )
    model = MTGFLOW(
        n_blocks=2, input_size=1, hidden_size=32, n_hidden=1,
        window_size=metadata['window_size'], n_sensor=n_sensor, dropout=0.0,
        model='MAF', batch_norm=False, use_meta=metadata['use_meta'],
        meta_input_dim=metadata['meta_input_dim'], meta_emb_dim=metadata['meta_emb_dim'],
    ).to(device)
    return model, test_loader


def compute_scores(model, test_loader):
    scores = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            meta = batch[3].to(device) if metadata_a['use_meta'] and len(batch) > 3 else None
            loss = -model.test(x, meta)
            scores.append(loss.cpu().numpy())
    return np.concatenate(scores)


path_a, ckpt_a = load_checkpoint(args.run_a)
path_b, ckpt_b = load_checkpoint(args.run_b)
metadata_a = ckpt_a['paderborn_metadata']
metadata_b = ckpt_b['paderborn_metadata']

model_a, test_loader = build_model_and_loaders(metadata_a)
model_a.load_state_dict(ckpt_a['model'])
model_a.eval()

model_b, _ = build_model_and_loaders(metadata_b)
model_b.load_state_dict(ckpt_b['model'])
model_b.eval()

labels = np.asarray(test_loader.dataset.label, dtype=int)

scores_a = compute_scores(model_a, test_loader)
scores_b = compute_scores(model_b, test_loader)

auroc_a = roc_auc_score(labels, scores_a)
auroc_b = roc_auc_score(labels, scores_b)

max_abs_diff = float(np.max(np.abs(scores_a - scores_b)))
bit_identical = bool(np.array_equal(scores_a, scores_b))

print(f"run_a: {path_a}")
print(f"run_b: {path_b}")
print(f"scores bit-identical: {bit_identical}")
print(f"scores max abs diff:  {max_abs_diff!r}")
print(f"auroc_a: {auroc_a!r}")
print(f"auroc_b: {auroc_b!r}")
print(f"auroc abs diff: {abs(auroc_a - auroc_b)!r}")
