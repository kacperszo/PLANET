"""Train PLANET to the harness training contract.

Beside the authors' `train.py`, not instead of it. Theirs trains on a directory, splits by
PDBbind index files or a random fraction, counts in optimiser steps and writes checkpoints named
`PLANET.iter-N`. That reproduces their run. The transfer programme needs the split to be an
argument, needs to start from somebody else's encoder, and needs a per-epoch curve that can be
put beside another model's — so this file exists and the model is imported unchanged.

Outputs follow `harness/training.py` — `model.pt`, `history.csv`, `summary.json`.

## Three things reproduced deliberately

**The authors' initialisation.** `train.py` does not use torch's defaults: every 1-dimensional
parameter is set to zero and everything else gets `xavier_uniform_`. That runs here too, and it
runs *before* any encoder is transferred, so a fine-tune starts from their initialisation in the
heads rather than from torch's.

**The affinity warm-up.** Their loss is `lig_loss + prolig_loss + beta * aff_loss` with
`beta = 0` for the first 500 optimiser steps. The affinity term is switched on only once the two
interaction predictions have something to say. Counted in steps, not epochs, exactly as theirs.

What is optimised is that weighted sum. What `history.csv` records as `train_loss` is the
*unweighted* one, so that it measures the same quantity as `val_loss` throughout — otherwise the
two columns would disagree about whether affinity counts for exactly the first 500 steps. The
three components are written out beside them, so the warm-up is still visible.

**Decoy ligands — requested as theirs, and inert on this data.** `ProLigDataset(decoy_flag=True)`
is passed exactly as the authors pass it, and the dataset is rebuilt every epoch as theirs is.
But the mechanism needs a `<id>_decoy.sdf` beside each complex, and **the PDBbind held here has
none**. With `decoys_count == 0`, `chem.random_ligand_decoy` forces `complex_label = 1` for every
pocket, so no negative is ever drawn. The authors trained on a pre-cleaned PDBbind carrying decoy
non-binders; that objective is not reproduced here, and the trainer says so at start-up and in
`summary.json` rather than letting the flag suggest otherwise. Found on 2026-09-24 — this
paragraph previously claimed the decoys were being swapped in.

## What the split file means here

Every other model in the roster reads `y_true` out of the split. PLANET does not: its pK comes
from the HDF5 attribute that `preprocess.py` wrote from the PDBbind index, and the dataset is
addressed by PDB code. So the split's `complex_id` column selects the complexes and its `y_true`
column is **not** consulted. That is worth stating rather than hiding — if the two ever disagree,
this trainer follows the preprocessing, and `summary.json` records the split file it was given.

usage:
    python train_contract.py --data-dir /cache/planet_h5 --train-labels /splits/train.csv \
        --val-labels /splits/val.csv --out /outputs --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from planet.data import ProLigDataset
from planet.model import PLANET
from planet.transfer import set_training_mode, transfer_encoder

#: The authors' values, from the argparse defaults in train.py.
AUTHOR_EPOCHS = 250
AUTHOR_BATCH_SIZE = 16
AUTHOR_LR = 1e-4
AUTHOR_CLIP_NORM = 200.0
AUTHOR_FEATURE_DIMS = 300
AUTHOR_NHEADS = 8
AUTHOR_KEY_DIMS = 300
AUTHOR_VALUE_DIMS = 300
AUTHOR_PRO_UPDATE_ITERS = 3
AUTHOR_LIG_UPDATE_ITERS = 10
AUTHOR_PRO_LIG_UPDATE_ITERS = 1
#: beta stays 0 until this many optimiser steps have passed — train.py:167
AFFINITY_WARMUP_STEPS = 500


def pearson(a, b) -> float:
    """Pearson R in numpy; nan where it is undefined rather than a warning and a number."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def count_decoys(data_dir: str, ids: set[str]) -> int:
    """How many of these complexes' HDF5 files carry at least one decoy ligand."""
    import h5py
    with_decoys = 0
    for pdb in ids:
        path = os.path.join(data_dir, pdb, f"{pdb}_pocket.h5")
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            if int(f.attrs.get("decoys_count", 0)) > 0:
                with_decoys += 1
    return with_decoys


def read_ids(path: str) -> set[str]:
    """PDB codes from a harness split file (complex_id,y_true). See the note in the docstring
    about why `y_true` is read and discarded."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "complex_id" not in rows[0]:
        raise SystemExit(f"{path}: split file needs a complex_id column")
    return {r["complex_id"].lower() for r in rows}


def authors_init(model: nn.Module) -> None:
    """train.py:152-156, verbatim in effect."""
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_uniform_(param)


def loader_for(dataset) -> DataLoader:
    """The authors' loader shape: the dataset already batches, so this hands over one
    pre-batched record at a time (train.py:180)."""
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                      drop_last=False, collate_fn=lambda x: x[0])


def run_eval(model, loader, device) -> tuple[float, float]:
    """Total loss and affinity Pearson R over a loader, model left as it was found."""
    was_training = model.training
    model.eval()
    losses, preds, trues = [], [], []
    with torch.no_grad():
        for res_batch, mol_batch, targets in loader:
            try:
                predictions = model(res_batch, mol_batch)
                lig_loss, prolig_loss, aff_loss = model.compute_loss(
                    predictions, targets, res_batch, mol_batch)
            except Exception as e:
                print(f"  skipping a validation batch: {type(e).__name__}: {e}")
                continue
            losses.append(float(lig_loss + prolig_loss + aff_loss))
            # Only complexes that actually carry an affinity contribute to R; pK_flags is the
            # authors' own mask for the ones that do.
            pKs, flags = targets[2], targets[3]
            affinities = predictions[2].detach().cpu().numpy()
            pKs, flags = pKs.cpu().numpy(), flags.cpu().numpy()
            # With decoy_flag=False the authors set every pK_flag to 1, including complexes
            # whose pK is 0 because the index had no entry for them. Those are not
            # measurements, so they are kept out of R here.
            keep = (flags > 0) & (pKs != 0)
            preds.extend(affinities[keep].tolist())
            trues.extend(pKs[keep].tolist())
    model.train(was_training)
    return (sum(losses) / max(len(losses), 1)), pearson(trues, preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PLANET to the harness contract")
    parser.add_argument("--data-dir", required=True,
                        help="directory of <pdb>/<pdb>_pocket.h5 from preprocess.py")
    parser.add_argument("--train-labels", required=True, help="csv with complex_id,y_true")
    parser.add_argument("--val-labels", default=None, help="csv with complex_id,y_true")
    parser.add_argument("--out", required=True,
                        help="output directory; model.pt, history.csv and summary.json go here")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"default: the authors' {AUTHOR_EPOCHS}")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"default: the authors' {AUTHOR_BATCH_SIZE}")
    parser.add_argument("--lr", type=float, default=None,
                        help=f"default: the authors' {AUTHOR_LR}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-encoder", default=None,
                        help="encoder .pt from `gnnb encoder split`; makes this a fine-tune")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="train the three heads only, against a fixed representation")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    epochs = args.epochs if args.epochs is not None else AUTHOR_EPOCHS
    batch_size = args.batch_size if args.batch_size is not None else AUTHOR_BATCH_SIZE
    lr = args.lr if args.lr is not None else AUTHOR_LR

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu"
    train_ids = read_ids(args.train_labels)
    val_ids = read_ids(args.val_labels) if args.val_labels else None
    print(f"device {device} | epochs {epochs} | batch {batch_size} | lr {lr}")
    print(f"{len(train_ids)} training complexes"
          + (f", {len(val_ids)} validation" if val_ids else ", no validation split"))
    with_decoys = count_decoys(args.data_dir, train_ids)
    if with_decoys == 0:
        print("WARNING: no training complex carries a decoy ligand, so the authors' decoy "
              "objective is inert and every complex trains as a true binder — see the "
              "module docstring")
    else:
        print(f"{with_decoys} of {len(train_ids)} training complexes carry decoys")

    model = PLANET(AUTHOR_FEATURE_DIMS, AUTHOR_NHEADS, AUTHOR_KEY_DIMS, AUTHOR_VALUE_DIMS,
                   AUTHOR_PRO_UPDATE_ITERS, AUTHOR_LIG_UPDATE_ITERS,
                   AUTHOR_PRO_LIG_UPDATE_ITERS, device).to(device)
    authors_init(model)

    provenance: dict[str, object] = {}
    if args.init_encoder:
        provenance = transfer_encoder(model, args.init_encoder, freeze=args.freeze_encoder)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    val_loader = None
    if val_ids:
        val_loader = loader_for(ProLigDataset(args.data_dir, pdb_ids=val_ids,
                                              batch_size=batch_size, shuffle=False,
                                              decoy_flag=False))

    os.makedirs(args.out, exist_ok=True)
    history: list[dict] = []
    best = {"epoch": -1, "val_loss": float("inf"), "val_r": float("nan")}
    best_state = None
    total_step = 0

    for epoch in range(epochs):
        # Rebuilt every epoch, as theirs is, so each epoch draws fresh decoy ligands.
        train_loader = loader_for(ProLigDataset(args.data_dir, pdb_ids=train_ids,
                                                batch_size=batch_size, shuffle=True,
                                                decoy_flag=True))
        set_training_mode(model, True, frozen_encoder=args.freeze_encoder)
        losses, lig_losses, prolig_losses, aff_losses = [], [], [], []
        preds, trues = [], []

        for res_batch, mol_batch, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(res_batch, mol_batch)
            lig_loss, prolig_loss, aff_loss = model.compute_loss(
                predictions, targets, res_batch, mol_batch)
            beta = 0.0 if total_step <= AFFINITY_WARMUP_STEPS else 1.0
            objective = lig_loss + prolig_loss + beta * aff_loss
            objective.backward()
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), AUTHOR_CLIP_NORM)
            optimizer.step()
            total_step += 1

            # **Recorded unweighted, optimised weighted.** `objective` above is what the
            # gradient sees, warm-up and all; `train_loss` in history.csv is the full
            # lig + prolig + aff sum. They differ only during the first 500 steps — and if
            # the recorded number carried beta, `train_loss` would exclude the affinity term
            # while `val_loss` included it, so the two columns would not be the same
            # quantity exactly where a curve is read most closely.
            losses.append(float(lig_loss + prolig_loss + aff_loss))
            lig_losses.append(float(lig_loss))
            prolig_losses.append(float(prolig_loss))
            aff_losses.append(float(aff_loss))
            pKs, flags = targets[2].cpu().numpy(), targets[3].cpu().numpy()
            affinities = predictions[2].detach().cpu().numpy()
            preds.extend(affinities[flags > 0].tolist())
            trues.extend(pKs[flags > 0].tolist())

        train_loss = sum(losses) / max(len(losses), 1)
        train_r = pearson(trues, preds)
        val_loss = val_r = ""
        if val_loader is not None:
            val_loss, val_r = run_eval(model, val_loader, device)
            if val_loss < best["val_loss"]:
                best = {"epoch": epoch, "val_loss": val_loss, "val_r": val_r}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # The first five keys are the harness's standard columns; the three after them are
        # PLANET's own, because its loss is a sum of three objectives and a single number
        # hides which one moved.
        history.append({"epoch": epoch, "train_loss": train_loss, "train_r": train_r,
                        "val_loss": val_loss, "val_r": val_r,
                        "lig_loss": sum(lig_losses) / max(len(lig_losses), 1),
                        "prolig_loss": sum(prolig_losses) / max(len(prolig_losses), 1),
                        "aff_loss": sum(aff_losses) / max(len(aff_losses), 1)})
        print(f"epoch {epoch} | step {total_step} | train {train_loss:.4f} R {train_r:.4f}"
              + (f" | val {val_loss:.4f} R {val_r:.4f}" if val_loader is not None else ""))

    # Without a validation split there is nothing to select on, so the last epoch is the model —
    # stated rather than implied.
    state = best_state if best_state is not None else model.state_dict()
    torch.save({k: v.detach().cpu() for k, v in state.items()},
               os.path.join(args.out, "model.pt"))

    with open(os.path.join(args.out, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "model": "planet",
        "epochs": len(history),
        "selected_epoch": best["epoch"] if best_state is not None else len(history) - 1,
        "selected_by": "val_loss" if best_state is not None else "last epoch (no val split)",
        "best_val_loss": best["val_loss"] if best_state is not None else None,
        "best_val_r": best["val_r"] if best_state is not None else None,
        "final_train_loss": history[-1]["train_loss"],
        "optimizer_steps": total_step,
        "seed": args.seed, "epochs_requested": epochs, "batch_size": batch_size, "lr": lr,
        "train_labels": args.train_labels, "val_labels": args.val_labels,
        # Said out loud: the split chose the complexes, the HDF5 attributes gave the pK.
        "labels_from": "preprocess.py HDF5 attributes, not the split's y_true column",
        # 0 means the authors' negative-sampling objective did not run — see the docstring.
        "training_complexes_with_decoys": with_decoys,
        "freeze_encoder": bool(args.freeze_encoder),
        **provenance,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
