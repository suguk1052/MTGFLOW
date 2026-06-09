# %%
import os
import argparse
import re
from datetime import datetime
import torch
from models.MTGFLOW import MTGFLOW
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve 

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, 
                    default='Data/input/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
parser.add_argument('--output_dir', type=str, 
                    default='./checkpoint/')
parser.add_argument('--checkpoint_root', type=str, default='./checkpoints',
                    help='Root directory for non-overwriting Paderborn checkpoints.')
parser.add_argument('--run_name', type=str, default=None,
                    help='Paderborn run name. If omitted, generated from timestamp and key settings.')
parser.add_argument('--name', default='SWaT', help='the name of dataset')

parser.add_argument('--graph', type=str, default='None')
parser.add_argument('--model', type=str, default='MAF')


parser.add_argument('--n_blocks', type=int, default=1, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--input_size', type=int, default=1)
parser.add_argument('--batch_norm', type=bool, default=False)
parser.add_argument('--train_split', type=float, default=0.6)
parser.add_argument('--stride_size', type=int, default=10)

# 🚀 [수정 포인트 1] 파더보른 하중 조건 폴더 지정을 위한 인자 추가
parser.add_argument('--load_setting', nargs='+', default=['N15_M07_F10'], help='Paderborn operational setting directories')
parser.add_argument('--train_ids', nargs='+', default=['K001', 'K002', 'K003'], help='Paderborn normal bearing IDs for training.')
parser.add_argument('--val_ids', nargs='+', default=['K004'], help='Paderborn normal bearing IDs for validation.')
parser.add_argument('--test_norm_ids', nargs='+', default=['K005', 'K006'], help='Paderborn normal bearing IDs for testing.')

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--window_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')



args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


def _slugify_run_part(value):
    value = str(value)
    value = re.sub(r'[^A-Za-z0-9_.-]+', '-', value)
    return value.strip('-') or 'none'


def build_paderborn_run_name(args):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    loads = '+'.join(args.load_setting)
    parts = [
        timestamp,
        f"loads-{loads}",
        f"model-{args.model}",
        f"blocks-{args.n_blocks}",
        f"hidden-{args.hidden_size}",
        f"win-{args.window_size}",
        f"stride-{args.stride_size}",
        f"bs-{args.batch_size}",
        f"lr-{args.lr}",
    ]
    return '_'.join(_slugify_run_part(part) for part in parts)


def resolve_save_path(args):
    if args.name.lower() == 'paderborn':
        if not args.run_name:
            args.run_name = build_paderborn_run_name(args)
        return os.path.join(args.checkpoint_root, 'Paderborn', args.run_name)
    return os.path.join(args.output_dir, args.name)


for seed in [2026]:
    args.seed = seed
    print(args)
    if args.name.lower() == 'paderborn':
        print(f"Paderborn run_name: {args.run_name or '<auto>'}")
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
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
            root="/home/dayoon/DCP/Data/Paderborn",
            loads=args.load_setting,               # 스크립트에서 넘겨받은 하중 조건 세팅 주입
            batch_size=args.batch_size,
            window_size=args.window_size,
            stride_size=args.stride_size,
            train_ids=args.train_ids,
            val_ids=args.val_ids,
            test_norm_ids=args.test_norm_ids
        )

    # %%
    model = MTGFLOW(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.window_size, n_sensor, dropout=0.0, model=args.model, batch_norm=args.batch_norm)
    model = model.to(device)

    # %%
    from torch.nn.utils import clip_grad_value_
    import seaborn as sns
    import matplotlib.pyplot as plt
    save_path = resolve_save_path(args)
    os.makedirs(save_path, exist_ok=True)
    if args.name.lower() == 'paderborn':
        print(f"Saving Paderborn checkpoint to {os.path.join(save_path, 'model.pth')}")


    loss_best = np.inf
    roc_max = 0
  
    lr = args.lr 
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'weight_decay': args.weight_decay},
        ], lr=lr, weight_decay=0.0)

    for epoch in range(40):
        loss_train = []

        model.train()
        for x, _, idx in train_loader:
            x = x.to(device)

            optimizer.zero_grad()
            loss = -model(x,)

            total_loss = loss

            total_loss.backward()
            clip_grad_value_(model.parameters(), 1)
            optimizer.step()
            loss_train.append(loss.item())



        if args.name.lower() == 'paderborn':
            loss_val = []
            model.eval()
            with torch.no_grad():
                for x, _, idx in val_loader:
                    x = x.to(device)
                    loss = -model.test(x, ).cpu().numpy()
                    loss_val.append(loss)
            loss_val = np.concatenate(loss_val)
            mean_val_loss = np.mean(loss_val)

            if loss_best > mean_val_loss:
                loss_best = mean_val_loss
                torch.save({
                    'model': model.state_dict(),
                    'run_name': args.run_name,
                    'args': vars(args),
                }, os.path.join(save_path, 'model.pth'))

            log_string = f"[Seed {seed}] Epoch {epoch:02d}/40 -> Mean Train Loss: {np.mean(loss_train):.4f} | Val Loss: {mean_val_loss:.4f} | Best Val Loss: {loss_best:.4f}"
            print(log_string)
        else:
            loss_test = []
            with torch.no_grad():
                for x, _, idx in test_loader:

                    x = x.to(device)
                    loss = -model.test(x, ).cpu().numpy()
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
            log_string = f"[Seed {seed}] Epoch {epoch:02d}/40 -> Mean Train Loss: {np.mean(loss_train):.4f} | Test AUROC: {roc_test:.4f} | Best AUROC: {roc_max:.4f}"
            print(log_string)