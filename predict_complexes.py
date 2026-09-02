"""Score complexes with PLANET, without needing ground-truth labels.

`evaluate.py` cannot do this: it computes metrics against true pK values and writes them
alongside the predictions, so it only works on sets whose affinities are already known.
Here the labels come out of the pipeline entirely — `preprocess.py` writes pK=0 for any
complex missing from its index, and nothing downstream reads it.

Expects a directory already preprocessed into per-complex `<id>_pocket.h5` files.

usage:
    python predict_complexes.py --complexes DIR --model checkpoints_2020/PLANET.iter-145000 \
        --out /outputs/predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from planet.data import ProLigDataset
from planet.model import PLANET


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict binding affinity for arbitrary complexes")
    parser.add_argument("--complexes", required=True, help="directory of preprocessed complexes")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    # architecture — must match the checkpoint, so the defaults mirror evaluate.py
    parser.add_argument("--feature_dims", type=int, default=300)
    parser.add_argument("-n", "--nheads", type=int, default=8)
    parser.add_argument("--key_dims", type=int, default=300)
    parser.add_argument("-va", "--value_dims", type=int, default=300)
    parser.add_argument("-pu", "--pro_update_inters", type=int, default=3)
    parser.add_argument("-lu", "--lig_update_iters", type=int, default=10)
    parser.add_argument("-pl", "--pro_lig_update_iters", type=int, default=1)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = PLANET(args.feature_dims, args.nheads, args.key_dims, args.value_dims,
                   args.pro_update_inters, args.lig_update_iters,
                   args.pro_lig_update_iters, device).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    dataset = ProLigDataset(args.complexes, pdb_ids=None, split="all",
                            batch_size=args.batch_size, shuffle=False, decoy_flag=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                        drop_last=False, collate_fn=lambda x: x[0])

    # ProLigDataset groups complexes into batches itself; batch i of the loader corresponds
    # to dataset.batches[i], which is how a prediction gets back to its complex id
    batch_paths = dataset.batches

    ids: list[str] = []
    preds: list[float] = []
    failed = 0

    with torch.no_grad():
        for batch_idx, (res_batch, mol_batch, _targets) in enumerate(
                tqdm(loader, desc="Predicting")):
            try:
                _, _, affinity = model(res_batch, mol_batch)
            except Exception as e:
                failed += len(batch_paths[batch_idx])
                print(f"batch {batch_idx} failed: {e}")
                continue
            values = affinity.squeeze().detach().cpu().reshape(-1).tolist()
            paths = batch_paths[batch_idx]
            if len(values) != len(paths):
                print(f"batch {batch_idx}: {len(values)} predictions for {len(paths)} complexes, skipping")
                failed += len(paths)
                continue
            for path, value in zip(paths, values):
                ids.append(os.path.basename(os.path.dirname(path)))
                preds.append(float(value))

    if failed:
        print(f"{failed} complexes produced no prediction")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["complex_id", "y_pred"])
        writer.writerows(zip(ids, preds))
    print(f"\n{len(ids)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
