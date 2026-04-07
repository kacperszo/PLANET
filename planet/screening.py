import sys,os,torch
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from torch.utils.data import Dataset, DataLoader
from planet.model import PLANET
from planet.chem import ProteinPocket, mol_batch_to_graph


class PlanetEstimator:
    def __init__(self, device):
        self.model = PLANET(300, 8, 300, 300, 3, 10, 1, device=device)
        self.model.load_parameters()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def set_pocket_from_ligand(self, protein_pdb, ligand_sdf):
        try:
            self.pocket = ProteinPocket(protein_pdb=protein_pdb, ligand_sdf=ligand_sdf)
        except Exception as e:
            raise RuntimeError(f'Failed to parse protein pocket: {e}') from e
        self.res_features = self.model.cal_res_features_helper(
            self.pocket.res_features, self.pocket.alpha_coordinates)

    def set_pocket_from_coordinate(self, protein_pdb, centeriod_x, centeriod_y, centeriod_z):
        try:
            self.pocket = ProteinPocket(protein_pdb, centeriod_x, centeriod_y, centeriod_z)
        except Exception as e:
            raise RuntimeError(f'Failed to parse protein pocket: {e}') from e
        self.res_features = self.model.cal_res_features_helper(
            self.pocket.res_features, self.pocket.alpha_coordinates)


class VS_SDF_Dataset(Dataset):
    def __init__(self, sdf_file, batch_size=32):
        self.batch_size = batch_size
        self.sdf_supp = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)
        self.data_index = self._build_index()

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        try:
            mol_batch_idx = self.data_index[idx]
            mol_batch = [self.sdf_supp[i] for i in mol_batch_idx]
            mol_names = [mol.GetProp('_Name') for mol in mol_batch if mol is not None]
            mol_batch = [Chem.AddHs(mol) for mol in mol_batch if mol is not None]
            mol_feature_batch = mol_batch_to_graph(mol_batch)
            mol_smiles = [Chem.MolToSmiles(Chem.RemoveHs(mol)) for mol in mol_batch if mol is not None]
            return (mol_feature_batch, mol_smiles, mol_names)
        except Exception as e:
            print(f"Skipping SDF batch {idx}: {e}")
            return (None, None, None)

    def _build_index(self):
        index_list = [i for i, mol in enumerate(self.sdf_supp) if mol is not None]
        index_list = [index_list[i:i + self.batch_size] for i in range(0, len(index_list), self.batch_size)]
        if len(index_list) >= 2 and len(index_list[-1]) <= 5:
            index_list[-1] = index_list.pop(-2) + index_list[-1]
        return index_list


class VS_SMI_Dataset(Dataset):
    def __init__(self, smi_file, batch_size=32):
        self.batch_size = batch_size
        self.contents = self._read_smi(smi_file)

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        mol_batch_contents = self.contents[idx]
        mol_batch_contents = [
            (Chem.AddHs(Chem.MolFromSmiles(smi)), smi, name)
            for (smi, name) in mol_batch_contents
            if Chem.MolFromSmiles(smi, sanitize=True) is not None
        ]
        mol_feature_batch = mol_batch_to_graph([c[0] for c in mol_batch_contents], auto_detect=False)
        return (mol_feature_batch, [c[1] for c in mol_batch_contents], [c[2] for c in mol_batch_contents])

    def _read_smi(self, smi_file):
        with open(smi_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        try:
            contents = [(line.split()[0], line.split()[1]) for line in lines]
        except IndexError:
            contents = [(line.split()[0], "UNKNOWN") for line in lines]
        return [contents[i:i + self.batch_size] for i in range(0, len(contents), self.batch_size)]


def workflow(protein_pdb, mol_file, ligand_sdf=None, centeriod_x=None, centeriod_y=None, centeriod_z=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    estimator = PlanetEstimator(device)
    estimator.model.to(device)

    if ligand_sdf is not None:
        estimator.set_pocket_from_ligand(protein_pdb, ligand_sdf)
    elif centeriod_x is not None and centeriod_y is not None and centeriod_z is not None:
        estimator.set_pocket_from_coordinate(protein_pdb, centeriod_x, centeriod_y, centeriod_z)
    else:
        raise ValueError("Provide either --ligand or all three of --center_x/y/z")

    suffix = os.path.basename(mol_file).rsplit('.', 1)[-1].lower()
    if suffix == 'smi':
        dataset = VS_SMI_Dataset(mol_file)
    elif suffix == 'sdf':
        dataset = VS_SDF_Dataset(mol_file)
    else:
        raise NotImplementedError(f"Unsupported molecule format: .{suffix} (use .sdf or .smi)")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
                            drop_last=False, collate_fn=lambda x: x[0])
    predicted_affinities, mol_names, smis = [], [], []
    with torch.no_grad():
        for (mol_feature_batch, smi_batch, mol_name) in dataloader:
            try:
                batch_size = len(smi_batch)
                fresidues_batch, res_scope = estimator.model.cal_res_features(estimator.res_features, batch_size)
                predicted_affinity = estimator.model.screening(fresidues_batch, res_scope, mol_feature_batch)
                predicted_affinities.append(predicted_affinity.view([-1]).cpu().numpy())
                smis.extend(smi_batch)
                mol_names.extend(mol_name)
            except Exception as e:
                print(f"Skipping batch: {e}")
                continue
    return np.concatenate(predicted_affinities), mol_names, smis


def main_cli():
    """Entry point registered as `planet-screen` in pyproject.toml."""
    from rdkit import RDLogger
    import argparse
    RDLogger.DisableLog('rdApp.*')
    parser = argparse.ArgumentParser(description='PLANET virtual screening')
    parser.add_argument('-p','--protein', required=True)
    parser.add_argument('-l','--ligand', default=None)
    parser.add_argument('-x','--center_x', default=None, type=float)
    parser.add_argument('-y','--center_y', default=None, type=float)
    parser.add_argument('-z','--center_z', default=None, type=float)
    parser.add_argument('-m','--mol_file', required=True)
    parser.add_argument('--prefix', default='result')
    args = parser.parse_args()
    predicted_affinities, mol_names, smis = workflow(
        args.protein, args.mol_file, args.ligand,
        args.center_x, args.center_y, args.center_z)
    result_to_csv_sdf(predicted_affinities, mol_names, smis, args.prefix)


def result_to_csv_sdf(predicted_affinities, mol_names, smis, prefix='result'):
    writer = Chem.SDWriter(prefix + '.sdf')
    writer.SetProps(['PLANET_affinity'])
    for aff, name, smi in zip(predicted_affinities, mol_names, smis):
        try:
            mol = Chem.MolFromSmiles(smi)
            mol.SetProp('PLANET_affinity', f'{aff:.3f}')
            mol.SetProp('_Name', name)
            writer.write(mol)
        except Exception:
            continue
    writer.close()
    pd.DataFrame({
        'mol_name': mol_names,
        'SMILES': smis,
        'PLANET_affinity': predicted_affinities,
    }).to_csv(prefix + '.csv', index=False)
