"""Preprocess PDBbind entries into per-complex HDF5 pocket files."""
import os, argparse, json
from planet.chem import ComplexPocket
from multiprocessing import Pool


def process_one(args):
    record_dir, pK_data_path = args
    pdb_name = os.path.basename(record_dir)
    ligand_sdf  = os.path.join(record_dir, f'{pdb_name}_ligand.sdf')
    protein_pdb = os.path.join(record_dir, f'{pdb_name}_protein.pdb')
    decoy_sdf   = os.path.join(record_dir, f'{pdb_name}_decoy.sdf')
    with open(pK_data_path, 'r') as f:
        pK_data = json.load(f)
    pK = pK_data.get(pdb_name, 0)
    try:
        pocket = ComplexPocket(protein_pdb, ligand_sdf, pK, decoy_sdf)
        pocket.save_h5(os.path.join(record_dir, f'{pdb_name}_pocket.h5'))
    except Exception as e:
        print(f"Skipping {pdb_name}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dir', required=True,
                        help='Path to PDBbind dataset directory')
    parser.add_argument('-n','--njobs', required=True, type=int,
                        help='Number of parallel workers')
    parser.add_argument('-k','--pk_data', required=True,
                        help='Path to pk_v2019.json')
    args = parser.parse_args()
    print(args)

    records = [(os.path.join(args.dir, sub), args.pk_data)
               for sub in os.listdir(args.dir)]
    with Pool(args.njobs) as pool:
        pool.map(process_one, records)
