# %%
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import time
import torch
from models.MTGFLOW import MTGFLOW
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Paderborn 데이터 기본 위치: 이 파일(MTGFLOW/)의 상위(../)에 있는 Data/Paderborn.
# 실행 위치(cwd)와 무관하게 항상 올바른 경로를 가리킨다.
_DATA_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'Data', 'Paderborn'))

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, 
                    default='Data/input/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
parser.add_argument('--output_dir', type=str, 
                    default='./checkpoint/')
parser.add_argument('--run_name', type=str, default=None,
                    help='Paderborn run name. Required for Paderborn runs.')
parser.add_argument('--name', default='SWaT', help='the name of dataset')

parser.add_argument('--graph', type=str, default='None')
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
parser.add_argument('--meta_inject', type=str, default='concat', choices=['concat', 'film'],
                    help="Meta 주입 방식: 'concat'(기존, condition C에 C_op를 이어붙임) 또는 'film'(C_op로 C를 곱·덧셈 변조; 항등 초기화).")
# 작업 D(진폭 confound 교정 전처리)
parser.add_argument('--amp_normalize', action='store_true',
                    help='작업 D: window별 RMS로 진동 window를 정규화(shape-only)하고 떼어낸 log-RMS를 별도 feature로 보존.')
parser.add_argument('--amp_normalize_channels', type=str, default='all', choices=['all', 'vib_only'],
                    help="작업 F-2 다채널: amp_normalize 시 정규화 대상 채널. 'all'=전 채널 각자 RMS(shape-only), "
                         "'vib_only'=채널0(진동)만 정규화하고 전류는 raw 진폭 보존. 단일채널이면 두 값이 동치.")
parser.add_argument('--rms_lambda', type=float, default=0.0,
                    help='작업 D: 진폭 페널티 가중치. anomaly score = flow_NLL(shape) + rms_lambda*0.5*penalty(z_rms). 학습엔 무영향(eval-only). 0이면 순수 shape 스코어.')
parser.add_argument('--rms_penalty', type=str, default='one-sided', choices=['one-sided', 'two-sided'],
                    help="작업 D: z_rms 페널티 형태. 'one-sided'=max(0,z_rms)²(고진폭만), 'two-sided'=z_rms².")
parser.add_argument('--rms_eps', type=float, default=1e-8,
                    help='작업 D: per-window RMS 정규화/로그의 0-분산 방어 eps.')
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
parser.add_argument('--seeds', type=int, nargs='+', default=[2026],
                    help='여러 시드 연달아 학습. run_name에 _s{seed} 자동 부착.')

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--window_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')
parser.add_argument('--epochs', type=int, default=40, help='학습 epoch 수(기본 40). 스모크 등 짧은 실행에 사용.')
parser.add_argument('--log_test_auroc', action='store_true',
                    help='(관찰용, 결과에 영향 없음) Paderborn 학습 중 매 epoch test AUROC를 계산해 '
                         'train_log.jsonl에 기록. checkpoint 선택 기준(val loss)에는 사용하지 않음.')



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
        'meta_inject': args.meta_inject,
        # 작업 D(진폭 confound 교정) / 작업 F-2(다채널 정규화 채널 선택)
        'amp_normalize': bool(args.amp_normalize),
        'amp_normalize_channels': str(args.amp_normalize_channels),
        'rms_lambda': float(args.rms_lambda),
        'rms_penalty': args.rms_penalty,
        'rms_eps': float(args.rms_eps),
        'rms_feature': 'log_rms',
        'train_logrms_mean': float(getattr(args, 'train_logrms_mean', 0.0)),
        'train_logrms_std': float(getattr(args, 'train_logrms_std', 1.0)),
    }

def resolve_save_path(args):
    if args.name.lower() == 'paderborn':
        if not args.run_name:
            raise ValueError('Paderborn training requires --run_name.')
        return os.path.join('results', 'Paderborn', args.run_name)
    return os.path.join(args.output_dir, args.name)


base_run_name = args.run_name
for seed in args.seeds:
    args.seed = seed
    # 시드별로 run_name에 _s{seed}를 붙여 결과 폴더/checkpoint/metadata를 분기(덮어쓰기 방지).
    # 매 반복마다 base에서 새로 붙여 다음 시드로 넘어갈 때 오염(_s7_s42)되지 않게 함.
    if base_run_name is not None:
        args.run_name = f"{base_run_name}_s{seed}"
    print(args)
    if args.name.lower() == 'paderborn':
        if not args.run_name:
            raise ValueError('Paderborn training requires --run_name.')
        paderborn_mode, train_settings, test_settings = configure_paderborn_args(args)
        print(f"Paderborn run_name: {args.run_name}")
        print(f"Mode: {paderborn_mode}")
        print(f"Train Settings: {train_settings}")
        print(f"Test Settings: {test_settings}")
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # %%
    print("Loading dataset")
    print(args.name)
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
            root=_DATA_ROOT,
            loads=args.load_setting,               # 스크립트에서 넘겨받은 하중 조건 세팅 주입
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
            measured_meta_stats=args.measured_meta_stats,
            amp_normalize=args.amp_normalize,
            amp_normalize_channels=args.amp_normalize_channels,
            rms_eps=args.rms_eps
        )
        # 작업 D: 진폭 정규화 시 train-normal log-RMS 통계를 checkpoint metadata에 앵커로 저장
        # (test 재구성이 동일 통계를 쓰는지 검증용). 로더가 dataset 속성으로 노출.
        args.train_logrms_mean = float(getattr(train_loader.dataset, 'train_logrms_mean', 0.0))
        args.train_logrms_std = float(getattr(train_loader.dataset, 'train_logrms_std', 1.0))

    # %%
    model = MTGFLOW(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.window_size, n_sensor, dropout=0.0, model=args.model, batch_norm=args.batch_norm, use_meta=args.use_meta, meta_input_dim=resolve_meta_input_dim(args), meta_emb_dim=args.meta_emb_dim, meta_inject=args.meta_inject)
    model = model.to(device)

    # %%
    from torch.nn.utils import clip_grad_value_
    import seaborn as sns
    import matplotlib.pyplot as plt
    save_path = resolve_save_path(args)
    os.makedirs(save_path, exist_ok=True)
    if args.name.lower() == 'paderborn':
        print(f"Saving Paderborn model to {os.path.join(save_path, 'model.pth')}")


    loss_best = np.inf
    roc_max = 0

    lr = args.lr
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'weight_decay': args.weight_decay},
        ], lr=lr, weight_decay=0.0)

    train_start_time = time.perf_counter()

    # 관찰용 학습 로그(Paderborn 전용): 매 epoch train/val loss(+옵션 test AUROC)를
    # 구조화된 jsonl로 저장. checkpoint 선택/학습 결과에는 영향 없음(순수 기록용).
    train_log_path = os.path.join(save_path, 'train_log.jsonl') if args.name.lower() == 'paderborn' else None
    if train_log_path and os.path.exists(train_log_path):
        os.remove(train_log_path)

    for epoch in range(args.epochs):
        epoch_start_time = time.perf_counter()
        loss_train = []

        model.train()
        for batch in train_loader:
            x = batch[0].to(device)
            meta = batch[3].to(device) if args.use_meta and len(batch) > 3 else None

            optimizer.zero_grad()
            loss = -model(x, meta)

            total_loss = loss

            total_loss.backward()
            clip_grad_value_(model.parameters(), 1)
            optimizer.step()
            loss_train.append(loss.item())



        if args.name.lower() == 'paderborn':
            loss_val = []
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    meta = batch[3].to(device) if args.use_meta and len(batch) > 3 else None
                    loss = -model.test(x, meta).cpu().numpy()
                    loss_val.append(loss)
            loss_val = np.concatenate(loss_val)
            mean_val_loss = np.mean(loss_val)

            # 관찰용 test AUROC(옵션): val split은 정상 데이터만 있어 AUROC 계산이 불가능하므로
            # 대신 test set으로 매 epoch AUROC를 계산해 기록한다. checkpoint 선택(위 loss_best
            # 기준)에는 전혀 관여하지 않는 순수 모니터링 값이다.
            test_auroc_log = None
            if args.log_test_auroc:
                loss_test_log = []
                with torch.no_grad():
                    for batch in test_loader:
                        x = batch[0].to(device)
                        meta = batch[3].to(device) if args.use_meta and len(batch) > 3 else None
                        loss = -model.test(x, meta).cpu().numpy()
                        loss_test_log.append(loss)
                loss_test_log = np.concatenate(loss_test_log)
                test_labels_log = np.asarray(test_loader.dataset.label, dtype=int)
                test_auroc_log = float(roc_auc_score(test_labels_log, loss_test_log))

            if loss_best > mean_val_loss:
                loss_best = mean_val_loss
                torch.save({
                    'model': model.state_dict(),
                    'run_name': args.run_name,
                    'args': vars(args),
                    'paderborn_metadata': build_paderborn_metadata(args),
                }, os.path.join(save_path, 'model.pth'))

            if train_log_path:
                with open(train_log_path, 'a', encoding='utf-8') as logf:
                    logf.write(json.dumps({
                        'epoch': epoch,
                        'seed': seed,
                        'train_loss_mean': float(np.mean(loss_train)),
                        'val_loss_mean': float(mean_val_loss),
                        'val_loss_best': float(loss_best),
                        'test_auroc_log_only': test_auroc_log,
                    }) + '\n')

            epoch_wall_clock_sec = time.perf_counter() - epoch_start_time
            auroc_log_str = f" | Test AUROC(log-only): {test_auroc_log:.4f}" if test_auroc_log is not None else ""
            log_string = f"[Seed {seed}] Epoch {epoch:02d}/{args.epochs} -> Mean Train Loss: {np.mean(loss_train):.4f} | Val Loss: {mean_val_loss:.4f} | Best Val Loss: {loss_best:.4f}{auroc_log_str} | Epoch Wall-Clock: {epoch_wall_clock_sec:.2f}s"
            print(log_string)
        else:
            loss_test = []
            with torch.no_grad():
                for batch in test_loader:

                    x = batch[0].to(device)
                    meta = batch[3].to(device) if args.use_meta and len(batch) > 3 else None
                    loss = -model.test(x, meta).cpu().numpy()
                    loss_test.append(loss)
            loss_test = np.concatenate(loss_test)

    

            roc_test = roc_auc_score(np.asarray(test_loader.dataset.label, dtype=int), loss_test)

    
            if roc_max < roc_test:
                roc_max = roc_test
                torch.save({
                    'model': model.state_dict(),
                }, os.path.join(save_path, 'model.pth'))

            roc_max = max(roc_test, roc_max)

            # 터미널 출력 포맷 수정
            epoch_wall_clock_sec = time.perf_counter() - epoch_start_time
            log_string = f"[Seed {seed}] Epoch {epoch:02d}/{args.epochs} -> Mean Train Loss: {np.mean(loss_train):.4f} | Test AUROC: {roc_test:.4f} | Best AUROC: {roc_max:.4f} | Epoch Wall-Clock: {epoch_wall_clock_sec:.2f}s"
            print(log_string)

    train_wall_clock_sec = time.perf_counter() - train_start_time
    print(f"[Seed {seed}] Train wall-clock time: {train_wall_clock_sec:.2f}s ({train_wall_clock_sec / 60:.2f} min)")
