"""Parse PDBbind v2019 INDEX_general_PL_data.2019 into a JSON {pdb_code: pK} file."""
import json, argparse, re

def parse_index(index_path):
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
            pdb_code = parts[0].lower()
            try:
                pk_val = float(parts[3])
            except ValueError:
                continue
            pk_data[pdb_code] = pk_val
    return pk_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', required=True,
                        help='Path to INDEX_general_PL_data.2019')
    parser.add_argument('-o', '--output', required=True,
                        help='Output JSON path, e.g. pk_v2019.json')
    args = parser.parse_args()

    pk_data = parse_index(args.index)
    with open(args.output, 'w') as f:
        json.dump(pk_data, f, indent=2)
    print(f"Saved {len(pk_data)} entries to {args.output}")
