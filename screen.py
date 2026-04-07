"""Virtual screening entry point — scores molecules against a protein pocket."""
from rdkit import RDLogger
import argparse
from planet.screening import workflow, result_to_csv_sdf

if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')
    parser = argparse.ArgumentParser(description='PLANET virtual screening')
    parser.add_argument('-p','--protein', required=True,
                        help='Protein structure file (.pdb)')
    parser.add_argument('-l','--ligand', default=None,
                        help='Crystal ligand SDF — defines pocket centre')
    parser.add_argument('-x','--center_x', default=None, type=float)
    parser.add_argument('-y','--center_y', default=None, type=float)
    parser.add_argument('-z','--center_z', default=None, type=float)
    parser.add_argument('-m','--mol_file', required=True,
                        help='Molecules to score (.sdf or .smi)')
    parser.add_argument('--prefix', default='result',
                        help='Output file prefix (default: result)')
    args = parser.parse_args()

    predicted_affinities, mol_names, smis = workflow(
        args.protein, args.mol_file, args.ligand,
        args.center_x, args.center_y, args.center_z)
    result_to_csv_sdf(predicted_affinities, mol_names, smis, args.prefix)
