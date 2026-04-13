"""Compare two preprocessed pocket .h5 files for the same PDB complex.

Useful for diagnosing preprocessing differences — e.g. why the same complex
processed from the CASF-2013 directory vs PLANET_dataset gives different
model predictions.

Usage:
    uv run scripts/compare_h5.py <pdb> <dir1> <dir2> [<dir3> ...]

Example:
    uv run scripts/compare_h5.py 10gs \\
        /home/kszot/gnns/PLANET_dataset/PDBbind2020-PLANET \\
        /home/kszot/gnns/CASF-2013/coreset
"""
import argparse, os, sys
import h5py
import numpy as np


def summarise(path: str) -> dict:
    with h5py.File(path, 'r') as f:
        res = f['res_features'][:]
        alpha = f['alpha_coordinates'][:]
        prolig = f['pro_lig_interaction'][:]
        lig = f['ligand_mol'][:]
        return {
            'pK': float(f.attrs['pK']),
            'n_residues': int(res.shape[0]),
            'n_ligand_atoms': int(prolig.shape[0]),
            'res_sum': float(res.sum()),
            'res_mean': float(res.mean()),
            'alpha_min': alpha.min(axis=0),
            'alpha_max': alpha.max(axis=0),
            'alpha_zero_rows': int((alpha == 0).all(axis=1).sum()),
            'contacts': int(prolig.sum()),
            'lig_bytes': int(lig.nbytes),
        }


def print_summary(label: str, s: dict) -> None:
    print(f'\n[{label}]')
    print(f"  pK              : {s['pK']}")
    print(f"  n_residues      : {s['n_residues']}")
    print(f"  n_ligand_atoms  : {s['n_ligand_atoms']}")
    print(f"  res_features    : sum={s['res_sum']:.3f}  mean={s['res_mean']:.4f}")
    amin, amax = s['alpha_min'], s['alpha_max']
    print(f"  alpha range     : x[{amin[0]:.2f},{amax[0]:.2f}] "
          f"y[{amin[1]:.2f},{amax[1]:.2f}] z[{amin[2]:.2f},{amax[2]:.2f}]")
    print(f"  alpha zero rows : {s['alpha_zero_rows']}/{s['n_residues']}")
    print(f"  contacts        : {s['contacts']} (pairs within 4 Å)")
    print(f"  ligand_mol size : {s['lig_bytes']} bytes")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('pdb', help='PDB code (e.g. 10gs)')
    p.add_argument('dirs', nargs='+', help='Two or more directories holding <pdb>/<pdb>_pocket.h5')
    args = p.parse_args()

    summaries = []
    for d in args.dirs:
        h5_path = os.path.join(d, args.pdb, f'{args.pdb}_pocket.h5')
        if not os.path.exists(h5_path):
            print(f'[skip] {h5_path} not found', file=sys.stderr)
            continue
        label = os.path.basename(os.path.normpath(d)) or d
        s = summarise(h5_path)
        print_summary(label, s)
        summaries.append((label, s))

    if len(summaries) < 2:
        return 1

    # pair-wise diff header
    print('\n' + '=' * 60)
    print('Differences (first vs others)')
    print('=' * 60)
    base_label, base = summaries[0]
    for label, s in summaries[1:]:
        print(f'\n[{base_label}] vs [{label}]')
        for key in ['pK', 'n_residues', 'n_ligand_atoms', 'contacts',
                    'res_sum', 'alpha_zero_rows', 'lig_bytes']:
            if base[key] != s[key]:
                print(f'  {key:16s}: {base[key]}  ->  {s[key]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
