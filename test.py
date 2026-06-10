#%%
import os
import argparse
import json
import time
import torch
from models.MTGFLOW import MTGFLOW
import numpy as np
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, 
                    default='../u_s/input/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
parser.add_argument('--output_dir', type=str, 
                    default='./checkpoint/')
parser.add_argument('--run_name', type=str, default=None,
                    help='Paderborn run name to evaluate from results/Paderborn/{run_name}/model.pth.')
parser.add_argument('--name',default='SWaT', help='the name of dataset')

parser.add_argument('--model', type=str, default='MAF')


parser.add_argument('--n_blocks', type=int, default=1, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--input_size', type=int, default=1)
parser.add_argument('--batch_norm', type=bool, default=False)
parser.add_argument('--train_split', type=float, default=0.6)
parser.add_argument('--stride_size', type=int, default=10)
parser.add_argument('--sampling_rate', type=float, default=1.0,
                    help='Sampling rate in samples/sec. Used to estimate realtime inference requirements.')

# 🚀 [수정 포인트 1] 파더보른 하중 조건 폴더 지정을 위한 인자 추가
parser.add_argument('--load_setting', nargs='+', default=['N15_M07_F10'], help='Paderborn operational setting directories')
parser.add_argument('--train_ids', nargs='+', default=['K001', 'K002', 'K003'], help='Paderborn normal bearing IDs for training.')
parser.add_argument('--val_ids', nargs='+', default=['K004'], help='Paderborn normal bearing IDs for validation.')
parser.add_argument('--test_norm_ids', nargs='+', default=['K005', 'K006'], help='Paderborn normal bearing IDs for testing.')
parser.add_argument('--exclude_ids', nargs='*', default=[], help='Paderborn bearing IDs to exclude from train, validation, and test splits.')
parser.add_argument('--feature_type', choices=['raw', 'stft'], default='raw', help='Paderborn feature type used at train time.')
parser.add_argument('--stft_n_fft', type=int, default=256, help='Paderborn STFT FFT size used at train time.')
parser.add_argument('--stft_hop_length', type=int, default=64, help='Paderborn STFT hop length used at train time.')
parser.add_argument('--stft_n_bands', type=int, default=16, help='Paderborn STFT pooled band count used at train time.')
parser.add_argument('--stft_logmag', action='store_true', default=True, help='Use log1p magnitude for Paderborn STFT features (enabled by default).')
parser.add_argument('--threshold_percentile', type=float, default=95, help='Percentile of validation normal scores for label-free thresholding.')

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--window_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')



args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


def resolve_checkpoint_path(args):
    if args.name.lower() == 'paderborn':
        if not args.run_name:
            raise ValueError('Paderborn evaluation requires --run_name.')
        return os.path.join('results', 'Paderborn', args.run_name, 'model.pth')
    return os.path.join(args.output_dir, args.name, 'model.pth')


def resolve_result_path(args):
    if args.name.lower() == 'paderborn':
        return os.path.join('results', 'Paderborn', args.run_name, 'paderborn_per_bearing_metrics.json')
    return os.path.join(args.output_dir, args.name, 'test_metrics.json')


def synchronize_if_cuda():
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.synchronize(device)


def compute_realtime_stats(num_windows, model_only_sec, end_to_end_sec, args):
    model_only_wps = float(num_windows / model_only_sec) if model_only_sec > 0 else None
    end_to_end_wps = float(num_windows / end_to_end_sec) if end_to_end_sec > 0 else None
    required_wps_for_realtime = float(args.sampling_rate / args.stride_size) if args.stride_size > 0 else None
    realtime_factor = (
        float(model_only_wps / required_wps_for_realtime)
        if model_only_wps is not None and required_wps_for_realtime is not None and required_wps_for_realtime > 0
        else None
    )
    end_to_end_realtime_factor = (
        float(end_to_end_wps / required_wps_for_realtime)
        if end_to_end_wps is not None and required_wps_for_realtime is not None and required_wps_for_realtime > 0
        else None
    )
    return {
        'sampling_rate': float(args.sampling_rate),
        'stride_size': int(args.stride_size),
        'required_wps_for_realtime': required_wps_for_realtime,
        'total_windows': int(num_windows),
        'model_only_inference_time_sec': float(model_only_sec),
        'model_only_wps': model_only_wps,
        'realtime_factor': realtime_factor,
        'end_to_end_inference_time_sec': float(end_to_end_sec),
        'end_to_end_wps': end_to_end_wps,
        'end_to_end_realtime_factor': end_to_end_realtime_factor,
        'device': str(device),
    }

checkpoint_path = resolve_checkpoint_path(args)
print(f'Loading checkpoint metadata from {checkpoint_path}')
checkpoint = torch.load(checkpoint_path, map_location=device)


def validate_paderborn_feature_config(args, checkpoint):
    feature_config = checkpoint.get('feature_config')
    if args.name.lower() != 'paderborn' or feature_config is None:
        return
    cli_config = {
        'feature_type': args.feature_type,
        'stft_n_fft': int(args.stft_n_fft),
        'stft_hop_length': int(args.stft_hop_length),
        'stft_n_bands': int(args.stft_n_bands),
        'stft_logmag': bool(args.stft_logmag),
        'raw_window_size': int(args.window_size),
    }
    mismatches = []
    for key, cli_value in cli_config.items():
        saved_value = feature_config.get(key)
        if saved_value != cli_value:
            mismatches.append(f'{key}: checkpoint={saved_value}, cli={cli_value}')
    if mismatches:
        raise ValueError('Paderborn feature configuration mismatch. Train and test must use identical feature settings: ' + '; '.join(mismatches))


validate_paderborn_feature_config(args, checkpoint)

from Dataset import load_smd_smap_msl, loader_SWat, loader_WADI, loader_PSM, loader_WADI_OCC

# 🚀 [수정 포인트 2] 우리가 작성한 파더보른 OCC 로더 함수 임포트
from Dataset.paderborn import loader_Paderborn_OCC

if args.name == 'SWaT':
    train_loader, val_loader, test_loader, n_sensor = loader_SWat(args.data_dir, \
                                                                    args.batch_size, args.window_size, args.stride_size, args.train_split)

elif args.name == 'Wadi':
    train_loader, val_loader, test_loader, n_sensor = loader_WADI(args.data_dir, \
                                                                args.batch_size, args.window_size, args.stride_size, args.train_split)

elif args.name == 'SMAP' or args.name == 'MSL' or args.name.startswith('machine'):
    train_loader, val_loader, test_loader, n_sensor = load_smd_smap_msl(args.name, \
                                                                args.batch_size, args.window_size, args.stride_size, args.train_split)

elif args.name == 'PSM':
    train_loader, val_loader, test_loader, n_sensor = loader_PSM(args.name, \
                                                                args.batch_size, args.window_size, args.stride_size, args.train_split)

# 🚀 [수정 포인트 3] 쉘 스크립트에서 --name=paderborn 을 줬을 때 작동할 분기 연결
elif args.name.lower() == 'paderborn':
    train_loader, val_loader, test_loader, n_sensor = loader_Paderborn_OCC(
        root="/home/dayoon/DCP/Data/Paderborn", 
        loads=args.load_setting,
        batch_size=args.batch_size,
        window_size=args.window_size,
        stride_size=args.stride_size,
        train_ids=args.train_ids,
        val_ids=args.val_ids,
        test_norm_ids=args.test_norm_ids,
        exclude_ids=args.exclude_ids,
        feature_type=args.feature_type,
        stft_n_fft=args.stft_n_fft,
        stft_hop_length=args.stft_hop_length,
        stft_n_bands=args.stft_n_bands,
        stft_logmag=args.stft_logmag
    )
else:
    raise ValueError(f'Unsupported dataset name: {args.name}')

if args.name.lower() == 'paderborn':
    model_window_size = train_loader.dataset.window_size
else:
    model_window_size = args.window_size

#%%
model = MTGFLOW(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, model_window_size, n_sensor, dropout=0.0, model = args.model, batch_norm=args.batch_norm)
model = model.to(device)

print(f'Loading checkpoint weights from {checkpoint_path}')
model.load_state_dict(checkpoint['model'])


model.eval()

def compute_scores(loader, measure_speed=False):
    scores = []
    num_windows = 0
    model_only_sec = 0.0

    synchronize_if_cuda()
    end_to_end_start = time.perf_counter()
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            batch_size = x.shape[0]

            synchronize_if_cuda()
            model_start = time.perf_counter()
            loss_tensor = -model.test(x,)
            synchronize_if_cuda()
            model_only_sec += time.perf_counter() - model_start

            loss = loss_tensor.cpu().numpy()
            scores.append(loss)
            num_windows += batch_size
    synchronize_if_cuda()
    end_to_end_sec = time.perf_counter() - end_to_end_start

    scores = np.concatenate(scores)
    if measure_speed:
        return scores, compute_realtime_stats(num_windows, model_only_sec, end_to_end_sec, args)
    return scores

loss_test, inference_speed = compute_scores(test_loader, measure_speed=True)
test_labels = np.asarray(test_loader.dataset.label,dtype=int)
roc_test = roc_auc_score(test_labels,loss_test)
print("The ROC score on {} dataset is {}".format(args.name, roc_test))
print(
    "Test inference speed on {} dataset: "
    "model-only={:.2f} windows/sec ({:.6f}s), "
    "end-to-end={:.2f} windows/sec ({:.6f}s), "
    "required_realtime={:.2f} windows/sec, "
    "realtime_factor={:.2f}x".format(
        args.name,
        inference_speed['model_only_wps'] or 0.0,
        inference_speed['model_only_inference_time_sec'],
        inference_speed['end_to_end_wps'] or 0.0,
        inference_speed['end_to_end_inference_time_sec'],
        inference_speed['required_wps_for_realtime'] or 0.0,
        inference_speed['realtime_factor'] or 0.0,
    )
)

metrics = {
    'checkpoint_path': checkpoint_path,
    'dataset': args.name,
    'overall_auroc': float(roc_test),
    'total_windows': int(len(loss_test)),
    'inference_speed': inference_speed,
}

if args.name.lower() == 'paderborn':
    val_scores = compute_scores(val_loader)
    threshold = float(np.percentile(val_scores, args.threshold_percentile))
    predictions = (loss_test >= threshold).astype(int)
    overall_accuracy = float(np.mean(predictions == test_labels))

    per_bearing_metrics = []
    test_ids = np.asarray(test_loader.dataset.ids)
    for bearing_id in sorted(np.unique(test_ids)):
        mask = test_ids == bearing_id
        labels = test_labels[mask]
        preds = predictions[mask]
        unique_labels = np.unique(labels)
        per_bearing_metrics.append({
            'id': str(bearing_id),
            'label': int(unique_labels[0]) if len(unique_labels) == 1 else 'mixed',
            'num_windows': int(mask.sum()),
            'accuracy': float(np.mean(preds == labels)),
        })

    metrics.update({
        'run_name': args.run_name,
        'paderborn_config': {
            'root': '/home/dayoon/DCP/Data/Paderborn',
            'loads': list(args.load_setting),
            'batch_size': int(args.batch_size),
            'window_size': int(args.window_size),
            'stride_size': int(args.stride_size),
            'sampling_rate': float(args.sampling_rate),
            'train_ids': list(args.train_ids),
            'val_ids': list(args.val_ids),
            'test_norm_ids': list(args.test_norm_ids),
            'exclude_ids': list(args.exclude_ids),
            'feature_type': args.feature_type,
            'stft_n_fft': int(args.stft_n_fft),
            'stft_hop_length': int(args.stft_hop_length),
            'stft_n_bands': int(args.stft_n_bands),
            'stft_logmag': bool(args.stft_logmag),
            'model_window_size': int(model_window_size),
        },
        'model_config': {
            'model': args.model,
            'n_blocks': int(args.n_blocks),
            'hidden_size': int(args.hidden_size),
            'n_hidden': int(args.n_hidden),
            'input_size': int(args.input_size),
            'batch_norm': bool(args.batch_norm),
        },
        'threshold': threshold,
        'threshold_percentile': float(args.threshold_percentile),
        'overall_accuracy': overall_accuracy,
        'per_bearing': per_bearing_metrics,
    })
else:
    metrics.update({
        'model_config': {
            'model': args.model,
            'n_blocks': int(args.n_blocks),
            'hidden_size': int(args.hidden_size),
            'n_hidden': int(args.n_hidden),
            'input_size': int(args.input_size),
            'batch_norm': bool(args.batch_norm),
        },
        'data_config': {
            'data_dir': args.data_dir,
            'batch_size': int(args.batch_size),
            'window_size': int(args.window_size),
            'stride_size': int(args.stride_size),
            'sampling_rate': float(args.sampling_rate),
            'train_split': float(args.train_split),
        },
    })

json_path = resolve_result_path(args)
os.makedirs(os.path.dirname(json_path), exist_ok=True)
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"Saved test metrics to {json_path}")
