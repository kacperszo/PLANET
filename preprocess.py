"""Preprocess PDBbind entries into per-complex HDF5 pocket files.

Basic usage:
    uv run preprocess.py -d /data/pdbbind -i /data/pdbbind/index/INDEX_general_PL_data.2019 -n 8
"""
import os, argparse, re
from planet.chem import ComplexPocket
from multiprocessing import Pool


def parse_index(index_path):
    """Parse PDBbind INDEX file into a {pdb_code: pK} dict."""
    pk_data = {}
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # format: PDB  resolution  year  -logKd/Ki  Kd/Ki  ...
            if len(parts) < 4:
                continue
            try:
                pk_data[parts[0].lower()] = float(parts[3])
            except ValueError:
                continue
    return pk_data


def process_one(args):
    record_dir, pk_data = args
    pdb_name = os.path.basename(record_dir)
    ligand_sdf  = os.path.join(record_dir, f'{pdb_name}_ligand.sdf')
    protein_pdb = os.path.join(record_dir, f'{pdb_name}_protein.pdb')
    decoy_sdf   = os.path.join(record_dir, f'{pdb_name}_decoy.sdf')
    pK = pk_data.get(pdb_name.lower(), 0)
    try:
        pocket = ComplexPocket(protein_pdb, ligand_sdf, pK, decoy_sdf)
        pocket.save_h5(os.path.join(record_dir, f'{pdb_name}_pocket.h5'))
    except Exception as e:
        print(f"Skipping {pdb_name}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess PDBbind entries into per-complex HDF5 files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-d', '--dir', required=True,
                        help='Path to PDBbind dataset directory')
    parser.add_argument('-i', '--index', required=True,
                        help='PDBbind INDEX file (e.g. INDEX_general_PL_data.2019)')
    parser.add_argument('-n', '--njobs', required=True, type=int,
                        help='Number of parallel workers')
    args = parser.parse_args()
    print(args)

    pk_data = parse_index(args.index)
    print(f"Parsed {len(pk_data)} pK entries from index.")

    records = [(os.path.join(args.dir, sub), pk_data)
               for sub in os.listdir(args.dir)
               if os.path.isdir(os.path.join(args.dir, sub))]
    with Pool(args.njobs) as pool:
        pool.map(process_one, records)
