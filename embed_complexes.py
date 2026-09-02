"""Extract a complex-level embedding from PLANET, with the prediction heads removed.

PLANET has no graph-level vector either. `ProLig` builds
`linear_ligand_affinity(lig) * linear_pocket_affinity(poc)` across the whole atom × residue
grid and sums per-pair scalars into the affinity, so stripping the heads leaves a grid, not
a vector. A pooling the authors never had has to be introduced, which makes this embedding
*defined* rather than *native*.

What we take is the output of `prolig_attention` — the residue and atom representations
after they have updated each other by cross-attention, which is the last point where the
network still holds separate, interpretable encodings of the two sides. Each is pooled per
complex and the two are concatenated, giving 2 × feature_dims.

Rather than hooking, this replays `PLANET.forward` up to that point. The scopes that say
which rows belong to which complex are needed for pooling anyway, and an explicit replay
makes it obvious that nothing downstream of the heads is being touched.

usage:
    python embed_complexes.py --complexes DIR --model checkpoints_2020/PLANET.iter-145000 \
        --out /outputs/embeddings.npz --pool sum
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from planet.data import ProLigDataset
from planet.model import PLANET

POOLS = {
    "sum": lambda t: t.sum(dim=0),
    "mean": lambda t: t.mean(dim=0),
    "max": lambda t: t.max(dim=0).values,
}


def encode(model: PLANET, res_batch, mol_batch):
    """PLANET.forward up to the cross-attention output, heads left untouched."""
    fresidues, _res_map, res_scope, alpha_coordinates = res_batch
    fatoms, fbonds, agraph, bgraph, lig_scope = mol_batch

    device = model.device
    fresidues = fresidues.to(device)
    alpha_coordinates = alpha_coordinates.to(device)
    fatoms = fatoms.to(device)
    fbonds = fbonds.to(device)
    agraph = agraph.to(device)
    bgraph = bgraph.to(device)

    fresidues = model.proteinegnn(fresidues, alpha_coordinates, res_scope)
    fatoms = model.ligandgat(fatoms, fbonds, agraph, bgraph, lig_scope)
    fatoms, fresidues = model.prolig.prolig_attention(fatoms, fresidues, lig_scope, res_scope)
    return fatoms, fresidues, lig_scope, res_scope


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed complexes with the heads removed")
    parser.add_argument("--complexes", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool", default="sum", choices=sorted(POOLS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
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
    batch_paths = dataset.batches
    pool = POOLS[args.pool]

    ids: list[str] = []
    vectors: list[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, (res_batch, mol_batch, _targets) in enumerate(
                tqdm(loader, desc="Embedding")):
            try:
                fatoms, fresidues, lig_scope, res_scope = encode(model, res_batch, mol_batch)
            except Exception as e:
                print(f"batch {batch_idx} failed: {e}")
                continue

            paths = batch_paths[batch_idx]
            for path, (start_res, n_res), (start_atom, n_atom) in zip(paths, res_scope, lig_scope):
                ligand = pool(fatoms[start_atom:start_atom + n_atom])
                pocket = pool(fresidues[start_res:start_res + n_res])
                vectors.append(torch.cat([ligand, pocket]).cpu().numpy())
                ids.append(os.path.basename(os.path.dirname(path)))

    if not vectors:
        raise RuntimeError("no complexes embedded")

    matrix = np.stack(vectors)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, ids=np.array(ids), vectors=matrix)
    print(f"\n{len(ids)} embeddings of dimension {matrix.shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()
