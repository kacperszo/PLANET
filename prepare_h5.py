"""Featurise the complexes named in a split into a reusable HDF5 cache.

`preprocess.py` is the authors' entry point and it writes `<id>_pocket.h5` *inside each complex
directory*. The harness mounts the dataset read-only on purpose — a model must never be able to
mutate the corpus every other model is scored on — and the dataset is 11 GB over 17655 entries
while a split is a few thousand. Copying the corpus to preprocess it is the wrong trade twice.

So this reads the split, reads each complex straight out of the read-only `/data`, and writes
only the HDF5 into a directory of its own, laid out as `<out>/<id>/<id>_pocket.h5` — which is
exactly what `ProLigDataset` scans for. The featurisation itself is the authors': `ComplexPocket`
is imported unchanged, with the same four arguments `preprocess.py` passes it.

pK comes from the PDBbind index, as theirs does, not from the split's `y_true` column. See the
note in `train_contract.py` about why the two are kept separate.

usage:
    python prepare_h5.py --complexes /data --index /data/index/INDEX_general_PL_data.2019 \
        --labels /splits/train.csv --out /cache/planet_h5 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import os
from multiprocessing import Pool

from tqdm import tqdm

from planet.chem import ComplexPocket


def parse_index(index_path: str) -> dict[str, float]:
    """{pdb_code: pK} from a PDBbind INDEX file — preprocess.py:parse_index, unchanged."""
    pk_data: dict[str, float] = {}
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                pk_data[parts[0].lower()] = float(parts[3])
            except ValueError:
                continue
    return pk_data


def process_one(job):
    pdb_id, complexes, out_dir, pK, skip_existing = job
    target_dir = os.path.join(out_dir, pdb_id)
    h5_path = os.path.join(target_dir, f'{pdb_id}_pocket.h5')
    if skip_existing and os.path.exists(h5_path):
        return pdb_id, 'cached'

    source = os.path.join(complexes, pdb_id)
    ligand_sdf = os.path.join(source, f'{pdb_id}_ligand.sdf')
    protein_pdb = os.path.join(source, f'{pdb_id}_protein.pdb')
    decoy_sdf = os.path.join(source, f'{pdb_id}_decoy.sdf')
    try:
        pocket = ComplexPocket(protein_pdb, ligand_sdf, pK, decoy_sdf)
        os.makedirs(target_dir, exist_ok=True)
        pocket.save_h5(h5_path)
        return pdb_id, 'ok'
    except Exception as e:
        return pdb_id, f'{type(e).__name__}: {e}'


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PLANET's HDF5 cache for one split")
    parser.add_argument("--complexes", required=True, help="read-only directory of complexes")
    parser.add_argument("--index", required=True, help="PDBbind INDEX_general_PL_data file")
    parser.add_argument("--labels", required=True, help="csv with complex_id,y_true")
    parser.add_argument("--out", required=True, help="directory for <id>/<id>_pocket.h5")
    parser.add_argument("--workers", type=int, default=0, help="0 runs in this process")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    pk_data = parse_index(args.index)
    print(f"{len(pk_data)} pK entries in the index")

    with open(args.labels, newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["complex_id"].lower() for r in rows]

    # The pK that trains PLANET comes from the index, not from the split. Say out loud if the
    # two disagree, rather than letting the split's y_true column look authoritative.
    disagree = [(r["complex_id"], float(r["y_true"]), pk_data[r["complex_id"].lower()])
                for r in rows
                if r["complex_id"].lower() in pk_data
                and abs(float(r["y_true"]) - pk_data[r["complex_id"].lower()]) > 0.01]
    if disagree:
        print(f"{len(disagree)} complexes carry a different affinity in the split than in the "
              f"index; PLANET trains on the index value:")
        for cid, split_y, index_y in disagree[:10]:
            print(f"  {cid}: split {split_y}, index {index_y}")

    jobs = [(i, args.complexes, args.out, pk_data.get(i, 0), args.skip_existing)
            for i in ids if os.path.isdir(os.path.join(args.complexes, i))]
    absent = len(ids) - len(jobs)
    if absent:
        print(f"{absent} of {len(ids)} complexes in the split are not in {args.complexes}")
    missing_pk = sum(1 for j in jobs if j[3] == 0)
    if missing_pk:
        print(f"{missing_pk} have no pK in the index and will carry 0. In training they "
              f"still contribute the two interaction objectives, because pK_flags masks the "
              f"affinity term; in validation the authors set every flag to 1, so "
              f"train_contract.py drops pK == 0 from the correlation instead")

    os.makedirs(args.out, exist_ok=True)
    print(f"preprocessing {len(jobs)} complexes into {args.out}")

    if args.workers:
        with Pool(args.workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_one, jobs), total=len(jobs)))
    else:
        results = [process_one(j) for j in tqdm(jobs)]

    ok = sum(1 for _, s in results if s == 'ok')
    cached = sum(1 for _, s in results if s == 'cached')
    failed = [(i, s) for i, s in results if s not in ('ok', 'cached')]
    print(f"{ok} written, {cached} already cached, {len(failed)} failed")
    for pdb_id, why in failed[:10]:
        print(f"  {pdb_id}: {why}")
    if not ok and not cached:
        raise SystemExit("nothing was preprocessed")


if __name__ == "__main__":
    main()
