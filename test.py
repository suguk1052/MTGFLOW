#%%
import os
import argparse
import json
import time
import sys
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
parser.add_argument('--use_meta', action='store_true', help='Use normalized Paderborn operational metadata as MTGFlow context.')
parser.add_argument('--meta_emb_dim', type=int, default=8, help='Metadata embedding dimension for context-aware MTGFlow.')
parser.add_argument('--meta_source', type=str, default='static', choices=['static', 'measured'],
                    help='Metadata source for Paderborn: static setting metadata or measured operational signals.')
parser.add_argument('--measured_meta_stats', type=str, default='meanstd', choices=['mean', 'meanstd'],
                    help='Window-level statistics for measured operational metadata.')
parser.add_argument('--train_split', type=float, default=0.6)
parser.add_argument('--stride_size', type=int, default=10)
parser.add_argument('--sampling_rate', type=float, default=1.0,
                    help='Sampling rate in samples/sec. Used to estimate realtime inference requirements.')

# 🚀 [수정 포인트 1] 파더보른 하중 조건 폴더 지정을 위한 인자 추가
parser.add_argument('--load_setting', nargs='+', default=['N15_M07_F10'], help='Paderborn operational setting directories (backward-compatible pooled default).')
parser.add_argument('--train_load_setting', nargs='+', default=None,
                    help='Paderborn operational settings used for train/val splits.')
parser.add_argument('--test_load_setting', nargs='+', default=None,
                    help='Paderborn operational settings used for test splits.')
parser.add_argument('--sensor_mode', type=str, default='vibration_1', help='Paderborn sensor channel name to load from .mat files.')
parser.add_argument('--train_ids', nargs='+', default=['K001', 'K002', 'K003'], help='Paderborn normal bearing IDs for training.')
parser.add_argument('--val_ids', nargs='+', default=['K004'], help='Paderborn normal bearing IDs for validation.')
parser.add_argument('--test_norm_ids', nargs='+', default=['K005', 'K006'], help='Paderborn normal bearing IDs for testing.')
parser.add_argument('--exclude_ids', nargs='*', default=[], help='Paderborn bearing IDs to exclude from train, validation, and test splits.')
parser.add_argument('--threshold_percentile', type=float, default=95, help='Percentile of validation normal scores for label-free thresholding.')

parser.add_argument('--seeds', type=int, nargs='+', default=None,
                    help='연속시드 평가. run_name에 _s{seed} 자동 부착. 미지정 시 run_name 그대로 단일 평가.')

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--window_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')



args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


def normalize_cli_list(values):
    normalized = []
    for item in values or []:
        normalized.extend(part.strip() for part in str(item).split(',') if part.strip())
    return normalized


def configure_paderborn_args(args):
    args.load_setting = normalize_cli_list(args.load_setting)
    if (args.train_load_setting is None) != (args.test_load_setting is None):
        raise ValueError('Paderborn cross-domain mode requires both --train_load_setting and --test_load_setting.')

    if args.train_load_setting is None and args.test_load_setting is None:
        mode = 'pooled'
        train_settings = list(args.load_setting)
        test_settings = list(args.load_setting)
    else:
        mode = 'cross-domain'
        args.train_load_setting = normalize_cli_list(args.train_load_setting)
        args.test_load_setting = normalize_cli_list(args.test_load_setting)
        train_settings = list(args.train_load_setting)
        test_settings = list(args.test_load_setting)

    return mode, train_settings, test_settings


def resolve_meta_input_dim(args):
    if args.meta_source == 'static':
        return 3
    if args.measured_meta_stats == 'mean':
        return 3
    return 6


def build_paderborn_metadata(args):
    return {
        'run_name': args.run_name,
        'load_setting': list(args.load_setting),
        'train_load_setting': None if args.train_load_setting is None else list(args.train_load_setting),
        'test_load_setting': None if args.test_load_setting is None else list(args.test_load_setting),
        'sensor_mode': args.sensor_mode,
        'train_ids': list(args.train_ids),
        'val_ids': list(args.val_ids),
        'test_norm_ids': list(args.test_norm_ids),
        'exclude_ids': list(args.exclude_ids),
        'window_size': int(args.window_size),
        'stride_size': int(args.stride_size),
        'sampling_rate': float(args.sampling_rate),
        'use_meta': bool(args.use_meta),
        'meta_source': args.meta_source,
        'measured_meta_stats': args.measured_meta_stats,
        'meta_input_dim': int(resolve_meta_input_dim(args)),
        'meta_emb_dim': int(args.meta_emb_dim),
    }


def option_was_provided(option_name):
    return any(argv == option_name or argv.startswith(option_name + '=') for argv in sys.argv[1:])


def reconcile_paderborn_args_with_checkpoint(args, checkpoint):
    metadata = checkpoint.get('paderborn_metadata')
    if not metadata:
        return configure_paderborn_args(args)

    configure_paderborn_args(args)
    current_metadata = build_paderborn_metadata(args)
    for key in ('load_setting', 'train_load_setting', 'test_load_setting', 'sensor_mode'):
        if option_was_provided('--' + key) and current_metadata.get(key) != metadata.get(key):
            print(f"⚠️ Warning: CLI {key}={current_metadata.get(key)} differs from checkpoint {key}={metadata.get(key)}. Using checkpoint metadata for reproducible evaluation.")

    args.load_setting = list(metadata.get('load_setting', args.load_setting))
    args.train_load_setting = metadata.get('train_load_setting')
    args.test_load_setting = metadata.get('test_load_setting')
    args.sensor_mode = metadata.get('sensor_mode', args.sensor_mode)
    args.train_ids = list(metadata.get('train_ids', args.train_ids))
    args.val_ids = list(metadata.get('val_ids', args.val_ids))
    args.test_norm_ids = list(metadata.get('test_norm_ids', args.test_norm_ids))
    args.exclude_ids = list(metadata.get('exclude_ids', args.exclude_ids))
    if not option_was_provided('--use_meta'):
        args.use_meta = bool(metadata.get('use_meta', args.use_meta))
    if not option_was_provided('--meta_source'):
        args.meta_source = metadata.get('meta_source', args.meta_source)
    if not option_was_provided('--measured_meta_stats'):
        args.measured_meta_stats = metadata.get('measured_meta_stats', args.measured_meta_stats)
    if not option_was_provided('--meta_emb_dim'):
        args.meta_emb_dim = int(metadata.get('meta_emb_dim', args.meta_emb_dim))
    return configure_paderborn_args(args)

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
        'use_meta': bool(args.use_meta),
        'meta_source': args.meta_source,
        'measured_meta_stats': args.measured_meta_stats,
        'meta_input_dim': int(resolve_meta_input_dim(args)),
        'meta_emb_dim': int(args.meta_emb_dim),
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

from Dataset import load_smd_smap_msl, loader_SWat, loader_WADI, loader_PSM, loader_WADI_OCC

# 🚀 [수정 포인트 2] 우리가 작성한 파더보른 OCC 로더 함수 임포트
from Dataset.paderborn import loader_Paderborn_OCC


def build_loaders(args):
    """데이터셋 로더를 1회 생성. 시드가 달라도 데이터 구성(metadata)은 동일하므로 재사용한다."""
    if args.name == 'SWaT':
        return loader_SWat(args.data_dir, \
                           args.batch_size, args.window_size, args.stride_size, args.train_split)

    elif args.name == 'Wadi':
        return loader_WADI(args.data_dir, \
                          args.batch_size, args.window_size, args.stride_size, args.train_split)

    elif args.name == 'SMAP' or args.name == 'MSL' or args.name.startswith('machine'):
        return load_smd_smap_msl(args.name, \
                                args.batch_size, args.window_size, args.stride_size, args.train_split)

    elif args.name == 'PSM':
        return loader_PSM(args.name, \
                         args.batch_size, args.window_size, args.stride_size, args.train_split)

    # 🚀 [수정 포인트 3] 쉘 스크립트에서 --name=paderborn 을 줬을 때 작동할 분기 연결
    elif args.name.lower() == 'paderborn':
        return loader_Paderborn_OCC(
            root="/home/dayoon/DCP/Data/Paderborn",
            loads=args.load_setting,
            train_loads=args.train_load_setting,
            test_loads=args.test_load_setting,
            sensor_mode=args.sensor_mode,
            batch_size=args.batch_size,
            window_size=args.window_size,
            stride_size=args.stride_size,
            train_ids=args.train_ids,
            val_ids=args.val_ids,
            test_norm_ids=args.test_norm_ids,
            exclude_ids=args.exclude_ids,
            meta_source=args.meta_source,
            measured_meta_stats=args.measured_meta_stats
        )
    else:
        raise ValueError(f'Unsupported dataset name: {args.name}')


def compute_scores(loader, model, measure_speed=False):
    scores = []
    num_windows = 0
    model_only_sec = 0.0

    synchronize_if_cuda()
    end_to_end_start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            meta = batch[3].to(device) if args.use_meta and len(batch) > 3 else None
            batch_size = x.shape[0]

            synchronize_if_cuda()
            model_start = time.perf_counter()
            loss_tensor = -model.test(x, meta)
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

def warn_if_metadata_mismatch(checkpoint, reference_metadata):
    """시드별 checkpoint의 핵심 metadata가 기준(첫 시드)과 다르면 경고. 데이터/모델 구성 불일치 방지."""
    if not reference_metadata:
        return
    metadata = checkpoint.get('paderborn_metadata')
    if not metadata:
        return
    for key in ('load_setting', 'train_load_setting', 'test_load_setting', 'sensor_mode',
                'train_ids', 'val_ids', 'test_norm_ids', 'window_size', 'stride_size',
                'use_meta', 'meta_source', 'measured_meta_stats', 'meta_input_dim', 'meta_emb_dim'):
        if metadata.get(key) != reference_metadata.get(key):
            print(f"⚠️ Warning: seed checkpoint {key}={metadata.get(key)} differs from "
                  f"reference {key}={reference_metadata.get(key)}. 결과가 시드 간 비교 불가능할 수 있음.")


def evaluate_run(run_name, model, test_loader, val_loader, paderborn_mode, reference_metadata):
    """단일 run_name(=시드)에 대해 checkpoint를 로드하고 평가 후 metrics를 저장/반환한다."""
    args.run_name = run_name
    checkpoint_path = resolve_checkpoint_path(args)
    print(f'\n=== Evaluating run: {run_name} ===')
    print(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    warn_if_metadata_mismatch(checkpoint, reference_metadata)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    loss_test, inference_speed = compute_scores(test_loader, model, measure_speed=True)
    test_labels = np.asarray(test_loader.dataset.label, dtype=int)
    roc_test = roc_auc_score(test_labels, loss_test)
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
        val_scores = compute_scores(val_loader, model)
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
                'train_load_setting': None if args.train_load_setting is None else list(args.train_load_setting),
                'test_load_setting': None if args.test_load_setting is None else list(args.test_load_setting),
                'sensor_mode': args.sensor_mode,
                'mode': paderborn_mode,
                'batch_size': int(args.batch_size),
                'window_size': int(args.window_size),
                'stride_size': int(args.stride_size),
                'sampling_rate': float(args.sampling_rate),
                'use_meta': bool(args.use_meta),
                'meta_source': args.meta_source,
                'measured_meta_stats': args.measured_meta_stats,
                'meta_input_dim': int(resolve_meta_input_dim(args)),
                'meta_emb_dim': int(args.meta_emb_dim),
                'train_ids': list(args.train_ids),
                'val_ids': list(args.val_ids),
                'test_norm_ids': list(args.test_norm_ids),
                'exclude_ids': list(args.exclude_ids),
            },
            'model_config': {
                'model': args.model,
                'n_blocks': int(args.n_blocks),
                'hidden_size': int(args.hidden_size),
                'n_hidden': int(args.n_hidden),
                'input_size': int(args.input_size),
                'batch_norm': bool(args.batch_norm),
                'use_meta': bool(args.use_meta),
                'meta_source': args.meta_source,
                'measured_meta_stats': args.measured_meta_stats,
                'meta_input_dim': int(resolve_meta_input_dim(args)),
                'meta_emb_dim': int(args.meta_emb_dim),
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
                'use_meta': bool(args.use_meta),
                'meta_source': args.meta_source,
                'measured_meta_stats': args.measured_meta_stats,
                'meta_input_dim': int(resolve_meta_input_dim(args)),
                'meta_emb_dim': int(args.meta_emb_dim),
                'train_split': float(args.train_split),
            },
        })

    json_path = resolve_result_path(args)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved test metrics to {json_path}")
    return metrics


def resolve_summary_path(base_run_name):
    """집계 요약 저장 경로. paderborn은 base run_name 폴더 아래에 둔다(없으면 생성)."""
    if args.name.lower() == 'paderborn':
        return os.path.join('results', 'Paderborn', base_run_name, 'summary_seeds.json')
    return os.path.join(args.output_dir, args.name, 'summary_seeds.json')


# ---- 드라이버: 시드 루프 ----
# 시드 미지정(None) → run_name 그대로 단일 평가(하위호환). 지정 시 _s{seed} 접미사로 루프.
base_run_name = args.run_name
seeds = args.seeds if args.seeds else [None]
run_names = [base_run_name if s is None else f"{base_run_name}_s{s}" for s in seeds]

# 첫 run의 checkpoint로 args(metadata)를 확정하고 로더/모델을 1회만 생성한다.
args.run_name = run_names[0]
first_ckpt_path = resolve_checkpoint_path(args)
print(f'Loading checkpoint metadata from {first_ckpt_path}')
first_checkpoint = torch.load(first_ckpt_path, map_location=device)
paderborn_mode = None
reference_metadata = first_checkpoint.get('paderborn_metadata')
if args.name.lower() == 'paderborn':
    paderborn_mode, train_settings, test_settings = reconcile_paderborn_args_with_checkpoint(args, first_checkpoint)
    print(f"Mode: {paderborn_mode}")
    print(f"Train Settings: {train_settings}")
    print(f"Test Settings: {test_settings}")

train_loader, val_loader, test_loader, n_sensor = build_loaders(args)

model = MTGFLOW(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.window_size, n_sensor, dropout=0.0, model=args.model, batch_norm=args.batch_norm, use_meta=args.use_meta, meta_input_dim=resolve_meta_input_dim(args), meta_emb_dim=args.meta_emb_dim)
model = model.to(device)

per_seed = []
for seed, run_name in zip(seeds, run_names):
    metrics = evaluate_run(run_name, model, test_loader, val_loader, paderborn_mode, reference_metadata)
    per_seed.append({
        'seed': seed,
        'run_name': run_name,
        'auroc': float(metrics['overall_auroc']),
        'accuracy': metrics.get('overall_accuracy'),
    })

# ---- 집계 요약 (--seeds가 주어진 경우) ----
if args.seeds:
    aurocs = np.asarray([r['auroc'] for r in per_seed], dtype=float)
    accs = [r['accuracy'] for r in per_seed if r['accuracy'] is not None]
    summary = {
        'base_run_name': base_run_name,
        'dataset': args.name,
        'seeds': [r['seed'] for r in per_seed],
        'per_seed': per_seed,
        'auroc_mean': float(np.mean(aurocs)),
        'auroc_std': float(np.std(aurocs)),
        'accuracy_mean': float(np.mean(accs)) if accs else None,
        'accuracy_std': float(np.std(accs)) if accs else None,
    }
    summary_path = resolve_summary_path(base_run_name)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== Seed aggregate summary =====")
    for r in per_seed:
        acc_str = f", accuracy={r['accuracy']:.4f}" if r['accuracy'] is not None else ""
        print(f"  seed {r['seed']}: AUROC={r['auroc']:.4f}{acc_str}")
    print(f"AUROC: {summary['auroc_mean']:.4f} ± {summary['auroc_std']:.4f}")
    if summary['accuracy_mean'] is not None:
        print(f"Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"Saved seed summary to {summary_path}")
