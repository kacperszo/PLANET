"""
Build train.pkl / valid.pkl / core.pkl for PLANET from PDBbind v2019 + CASF-2016.

Directory layout assumed:
  v2019_dir/
    <pdb>/
      <pdb>_ligand.sdf
      <pdb>_protein.pdb
      <pdb>_pocket.pkl          ← created by process_PDBBind.py

  casf_dir/   (CASF-2016/coreset)
    <pdb>/
      <pdb>_ligand.sdf
      <pdb>_protein.pdb
      <pdb>_pocket.pkl          ← created by process_PDBBind.py

  casf_scores/  (CASF-2016/power_scoring)
    CoreSet.dat

Outputs (in --out_dir):
  train.pkl, valid.pkl, core.pkl
  Each is a list of [pkl_path, pK] pairs.
"""
import os, pickle, random, json, argparse
import numpy as np


def parse_coreset_dat(dat_path):
    """Return dict {pdb_code: pK} from CoreSet.dat."""
    pk = {}
    with open(dat_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            pk[parts[0].lower()] = float(parts[3])
    return pk


def parse_pk_json(json_path):
    with open(json_path) as f:
        return json.load(f)


def collect_records(data_dir, pk_dict, exclude_set=None):
    """Walk data_dir, return list of [pkl_path, pK] for complexes with a pocket pkl."""
    records = []
    for pdb in os.listdir(data_dir):
        if exclude_set and pdb.lower() in exclude_set:
            continue
        pkl_path = os.path.join(data_dir, pdb, f'{pdb}_pocket.pkl')
        if not os.path.exists(pkl_path):
            continue
        pK = pk_dict.get(pdb.lower(), 0.0)
        records.append([pkl_path, pK])
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v2019_dir', required=True,
                        help='Path to PDBbind v2019 directory')
    parser.add_argument('--casf_dir', required=True,
                        help='Path to CASF-2016/coreset directory')
    parser.add_argument('--casf_scores', required=True,
                        help='Path to CASF-2016/power_scoring directory (contains CoreSet.dat)')
    parser.add_argument('--pk_json', required=True,
                        help='Path to pk_v2019.json (from make_pk_json.py)')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for train/valid/core pkl files')
    parser.add_argument('--valid_frac', type=float, default=0.1,
                        help='Fraction of non-CASF v2019 records used for validation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── core set (CASF-2016) ──────────────────────────────────────────────────
    coreset_pk = parse_coreset_dat(os.path.join(args.casf_scores, 'CoreSet.dat'))
    casf_ids = set(coreset_pk.keys())

    core_records = collect_records(args.casf_dir, coreset_pk)
    print(f"Core (CASF-2016): {len(core_records)} / {len(casf_ids)} complexes found")

    # ── train + valid (v2019 minus CASF-2016) ────────────────────────────────
    pk_json = parse_pk_json(args.pk_json)
    all_train_records = collect_records(args.v2019_dir, pk_json, exclude_set=casf_ids)
    print(f"v2019 (minus CASF-2016): {len(all_train_records)} records with pocket pkl")

    random.shuffle(all_train_records)
    n_valid = int(len(all_train_records) * args.valid_frac)
    valid_records = all_train_records[:n_valid]
    train_records = all_train_records[n_valid:]
    print(f"Train: {len(train_records)}, Valid: {len(valid_records)}")

    # ── save ──────────────────────────────────────────────────────────────────
    for name, records in [('train', train_records),
                          ('valid', valid_records),
                          ('core', core_records)]:
        out_path = os.path.join(args.out_dir, f'{name}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(records, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
