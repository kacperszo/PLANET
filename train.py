"""
PLANET training script.

Basic usage:
    uv run train.py -d /data/pdbbind -c /data/casf2016 -s checkpoints/

Resume from checkpoint:
    uv run train.py -d /data/pdbbind -c /data/casf2016 -s checkpoints/ \\
        --checkpoint checkpoints/PLANET.iter-50000 --initial_step 50000
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from planet.data import ProLigDataset
from planet.model import PLANET
import argparse, os, sys
import numpy as np
from rdkit import RDLogger


if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')
    parser = argparse.ArgumentParser(
        description='Train PLANET on PDBbind',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── data ──────────────────────────────────────────────────────────────────
    data = parser.add_argument_group('data')
    data.add_argument('-d', '--data_dir', required=True,
                      help='PDBbind directory — expects <pdb>/<pdb>_pocket.h5 per entry')
    data.add_argument('-c', '--casf_dir', required=True,
                      help='CASF test directory (held out as test set, or test index dir)')
    data.add_argument('--train_index', default=None,
                      help='PDBbind index file for training set (explicit split mode)')
    data.add_argument('--valid_index', default=None,
                      help='PDBbind index file for validation set (explicit split mode)')
    data.add_argument('--test_index', default=None,
                      help='PDBbind index file for test set (explicit split mode)')
    data.add_argument('--valid_frac', type=float, default=0.1,
                      help='Fraction of PDBbind entries used for validation (random split mode)')

    # ── model ─────────────────────────────────────────────────────────────────
    model_args = parser.add_argument_group('model architecture')
    model_args.add_argument('--feature_dims', type=int, default=300,
                            help='Hidden feature dimensionality')
    model_args.add_argument('-n', '--nheads', type=int, default=8,
                            help='Number of attention heads')
    model_args.add_argument('--key_dims', type=int, default=300,
                            help='Attention key dimensionality')
    model_args.add_argument('-va', '--value_dims', type=int, default=300,
                            help='Attention value dimensionality')
    model_args.add_argument('-pu', '--pro_update_inters', type=int, default=3,
                            help='Protein EGNN update iterations')
    model_args.add_argument('-lu', '--lig_update_iters', type=int, default=10,
                            help='Ligand GAT update iterations')
    model_args.add_argument('-pl', '--pro_lig_update_iters', type=int, default=1,
                            help='Protein-ligand cross-attention iterations')

    # ── training ──────────────────────────────────────────────────────────────
    train_args = parser.add_argument_group('training')
    train_args.add_argument('--epoch', type=int, default=250,
                            help='Number of training epochs')
    train_args.add_argument('--batch_size', type=int, default=16,
                            help='Complexes per batch')
    train_args.add_argument('--lr', type=float, default=1e-4,
                            help='Initial learning rate')
    train_args.add_argument('--clip_norm', type=float, default=200.0,
                            help='Gradient clipping norm')
    train_args.add_argument('--anneal_iter', type=int, default=20000,
                            help='LR decay interval in steps (applied after step 60k)')

    # ── checkpointing & logging ───────────────────────────────────────────────
    ckpt = parser.add_argument_group('checkpointing and logging')
    ckpt.add_argument('-s', '--save_dir', required=True,
                      help='Directory to save checkpoints')
    ckpt.add_argument('--checkpoint', default=None,
                      help='Path to checkpoint file to resume from')
    ckpt.add_argument('--initial_step', type=int, default=0,
                      help='Global step to start from (set when resuming)')
    ckpt.add_argument('--save_iter', type=int, default=5000,
                      help='Save checkpoint and run eval every N steps')
    ckpt.add_argument('--print_iter', type=int, default=200,
                      help='Print training metrics every N steps')

    args = parser.parse_args()
    print(args)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── split mode ────────────────────────────────────────────────────────────
    def parse_index_ids(path):
        """Return set of PDB codes from a PDBbind index file."""
        ids = set()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                ids.add(line.split()[0].lower())
        return ids

    given = sum(x is not None for x in [args.train_index, args.valid_index, args.test_index])
    if 0 < given < 3:
        parser.error("Provide all three index flags (--train_index, --valid_index, --test_index) or none.")
    explicit_split = given == 3
    if explicit_split:
        train_ids = parse_index_ids(args.train_index)
        valid_ids = parse_index_ids(args.valid_index)
        test_ids  = parse_index_ids(args.test_index)
        print(f"Explicit split: {len(train_ids)} train / {len(valid_ids)} valid / {len(test_ids)} test")
    else:
        casf_ids = {d.lower() for d in os.listdir(args.casf_dir)}
        print(f"Random split: valid_frac={args.valid_frac}, excluding {len(casf_ids)} CASF entries")

    # ── datasets ──────────────────────────────────────────────────────────────
    if explicit_split:
        valid_dataset = ProLigDataset(
            args.data_dir, pdb_ids=valid_ids,
            batch_size=args.batch_size, shuffle=False, decoy_flag=False)
        test_dataset = ProLigDataset(
            args.data_dir, pdb_ids=test_ids,
            batch_size=args.batch_size, shuffle=False, decoy_flag=False)
    else:
        valid_dataset = ProLigDataset(
            args.data_dir, split='valid',
            exclude_ids=casf_ids, valid_frac=args.valid_frac,
            batch_size=args.batch_size, shuffle=False, decoy_flag=False)
        test_dataset = ProLigDataset(
            args.casf_dir, split='all',
            batch_size=args.batch_size, shuffle=False, decoy_flag=False)

    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                              num_workers=4, drop_last=False, collate_fn=lambda x: x[0])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=4, drop_last=False, collate_fn=lambda x: x[0])

    # ── model ─────────────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PLANET(
        args.feature_dims, args.nheads, args.key_dims, args.value_dims,
        args.pro_update_inters, args.lig_update_iters, args.pro_lig_update_iters,
        device).to(device)

    if args.checkpoint:
        print(f"Resuming from checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    else:
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    print("Model #Params: {:d}K".format(sum(x.nelement() for x in model.parameters()) // 1000))
    print(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.8)
    total_step = args.initial_step
    meters = np.zeros(6)
    beta = 0.0 if total_step <= 500 else 1.0

    # ── training loop ─────────────────────────────────────────────────────────
    model.train()
    for epoch in range(1, 1 + args.epoch):
        if explicit_split:
            train_dataset = ProLigDataset(
                args.data_dir, pdb_ids=train_ids,
                batch_size=args.batch_size, shuffle=True, decoy_flag=True)
        else:
            train_dataset = ProLigDataset(
                args.data_dir, split='train',
                exclude_ids=casf_ids, valid_frac=args.valid_frac,
                batch_size=args.batch_size, shuffle=True, decoy_flag=True)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                                  num_workers=4, drop_last=False, collate_fn=lambda x: x[0])

        for (res_feature_batch, mol_feature_batch, targets) in train_loader:
            optimizer.zero_grad()
            predictions = model(res_feature_batch, mol_feature_batch)
            lig_loss, prolig_loss, aff_loss = model.compute_loss(
                predictions, targets, res_feature_batch, mol_feature_batch)
            (lig_loss + prolig_loss + beta * aff_loss).backward()
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), args.clip_norm)
            optimizer.step()

            lig_acc, prolig_acc, aff_mae = model.compute_metrics(predictions, targets)
            meters += np.array([lig_loss.item(), lig_acc.item(), prolig_loss.item(),
                                 prolig_acc.item(), aff_loss.item(), aff_mae.item()])
            total_step += 1
            beta = 0.0 if total_step <= 500 else 1.0

            if total_step % args.print_iter == 0:
                meters /= args.print_iter
                print("[{}]\tLig_L:{:.3f}\tLig_ACC:{:.3f}\tProLig_L:{:.3f}"
                      "\tProLig_ACC:{:.3f}\tAffinity_L:{:.3f}\tMAE:{:.3f}".format(
                          total_step, *meters))
                sys.stdout.flush()
                meters[:] = 0.

            if total_step > 60000 and total_step % args.anneal_iter == 0:
                scheduler.step()
                print("[learning rate]: {:.6f}".format(scheduler.get_last_lr()[0]))

            if total_step > 0 and total_step % args.save_iter == 0:
                ckpt_path = os.path.join(args.save_dir, f"PLANET.iter-{total_step}")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

                model.eval()
                with torch.no_grad():
                    for tag, loader in [('Valid', valid_loader), ('Test', test_loader)]:
                        eval_meters = np.zeros(6)
                        count = 0
                        for (res_batch, mol_batch, targets) in loader:
                            try:
                                preds = model(res_batch, mol_batch)
                                ll, pll, al = model.compute_loss(preds, targets, res_batch, mol_batch)
                                la, pla, mae = model.compute_metrics(preds, targets)
                                eval_meters += np.array([ll.item(), la.item(), pll.item(),
                                                         pla.item(), al.item(), mae.item()])
                                count += 1
                            except Exception as e:
                                print(f"[{tag}] skipping batch: {e}")
                                continue
                        if count > 0:
                            eval_meters /= count
                            print("[{}_{}]\tLig_L:{:.3f}\tLig_ACC:{:.3f}\tProLig_L:{:.3f}"
                                  "\tProLig_ACC:{:.3f}\tAffinity_L:{:.3f}\tMAE:{:.3f}".format(
                                      tag, total_step, *eval_meters))
                model.train()
