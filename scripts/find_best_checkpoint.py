"""Evaluate every PLANET.iter-* checkpoint on a dataset and report metrics.

Methodologically correct usage:
    1. Run this on the VALID set to pick the best checkpoint.
    2. Then run evaluate.py once on the TEST set with that checkpoint.

Do NOT pick a checkpoint based on test-set metrics — that is data leakage.

Usage:
    uv run scripts/find_best_checkpoint.py \\
        --checkpoints_dir checkpoints_2020/ \\
        --data_dir PLANET_dataset/PDBbind2020-PLANET \\
        --index PLANET_dataset/index/PDBbind_PLANET_ValidSet.2020 \\
        --metric pearson \\
        [--out_csv results/checkpoint_sweep.csv]

Metrics: mae, rmse, pearson, spearman, ci
"""
import argparse, csv, glob, os, re, sys
import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader
from rdkit import RDLogger

from planet.model import PLANET
from planet.data import ProLigDataset


def concordance_index(pred, actual):
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    pairs = 0
    concordant = 0.0
    for i in range(len(actual)):
        for j in range(i + 1, len(actual)):
            if actual[i] == actual[j]:
                continue
            pairs += 1
            dp = pred[i] - pred[j]
            da = actual[i] - actual[j]
            if dp * da > 0:
                concordant += 1.0
            elif dp == 0:
                concordant += 0.5
    return concordant / pairs if pairs > 0 else 0.0


def evaluate_checkpoint(model, dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                        drop_last=False, collate_fn=lambda x: x[0])
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for (res_batch, mol_batch, targets) in loader:
            try:
                _, _, pred_aff = model(res_batch, mol_batch)
                _, _, pK, _, _ = targets
                preds.append(np.array(pred_aff.squeeze().detach().cpu()))
                trues.append(np.array(pK.squeeze().detach().cpu()))
            except Exception as e:
                print(f'  [skip batch] {e}', file=sys.stderr)
                continue
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean(np.square(preds - trues))))
    pearson, _ = stats.pearsonr(preds, trues)
    spearman, _ = stats.spearmanr(preds, trues)
    ci = concordance_index(preds, trues)
    return {'mae': mae, 'rmse': rmse, 'pearson': float(pearson),
            'spearman': float(spearman), 'ci': ci, 'n': int(len(preds))}


def parse_index(path):
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            ids.add(line.split()[0].lower())
    return ids


def iter_step(path):
    m = re.search(r'iter-(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    RDLogger.DisableLog('rdApp.*')
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--checkpoints_dir', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--index', required=True, help='PDBbind index file (valid set recommended)')
    p.add_argument('--metric', default='pearson',
                   choices=['mae', 'rmse', 'pearson', 'spearman', 'ci'])
    p.add_argument('--out_csv', default=None)
    p.add_argument('--pattern', default='PLANET.iter-*')
    p.add_argument('--feature_dims', type=int, default=300)
    p.add_argument('--nheads', type=int, default=8)
    p.add_argument('--key_dims', type=int, default=300)
    p.add_argument('--value_dims', type=int, default=300)
    p.add_argument('--pro_update_inters', type=int, default=3)
    p.add_argument('--lig_update_iters', type=int, default=10)
    p.add_argument('--pro_lig_update_iters', type=int, default=1)
    args = p.parse_args()

    ckpts = sorted(glob.glob(os.path.join(args.checkpoints_dir, args.pattern)),
                   key=iter_step)
    if not ckpts:
        print(f'No checkpoints matching {args.pattern} in {args.checkpoints_dir}',
              file=sys.stderr)
        return 1
    print(f'Found {len(ckpts)} checkpoints')

    pdb_ids = parse_index(args.index)
    print(f'Index: {len(pdb_ids)} entries')
    dataset = ProLigDataset(args.data_dir, pdb_ids=pdb_ids,
                            batch_size=16, shuffle=False, decoy_flag=False)
    print(f'Dataset batches: {len(dataset)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PLANET(args.feature_dims, args.nheads, args.key_dims, args.value_dims,
                   args.pro_update_inters, args.lig_update_iters,
                   args.pro_lig_update_iters, device).to(device)

    rows = []
    for ckpt in ckpts:
        step = iter_step(ckpt)
        print(f'\n[iter-{step}] {ckpt}')
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m = evaluate_checkpoint(model, dataset)
        m['step'] = step
        m['checkpoint'] = os.path.basename(ckpt)
        rows.append(m)
        print(f"  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
              f"R={m['pearson']:.3f}  ρ={m['spearman']:.3f}  CI={m['ci']:.3f}")

    higher_better = args.metric in ('pearson', 'spearman', 'ci')
    rows.sort(key=lambda r: r[args.metric], reverse=higher_better)

    print('\n' + '=' * 70)
    print(f'Ranked by {args.metric} ({"higher" if higher_better else "lower"} is better)')
    print('=' * 70)
    print(f'{"step":>8} {"MAE":>7} {"RMSE":>7} {"R":>7} {"ρ":>7} {"CI":>7}')
    print('-' * 70)
    for r in rows:
        print(f"{r['step']:>8} {r['mae']:>7.3f} {r['rmse']:>7.3f} "
              f"{r['pearson']:>7.3f} {r['spearman']:>7.3f} {r['ci']:>7.3f}")

    best = rows[0]
    print(f"\nBest checkpoint by {args.metric}: {best['checkpoint']} (step {best['step']})")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['step', 'checkpoint', 'n',
                                              'mae', 'rmse', 'pearson', 'spearman', 'ci'])
            w.writeheader()
            for r in sorted(rows, key=lambda x: x['step']):
                w.writerow(r)
        print(f'Saved sweep to {args.out_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
