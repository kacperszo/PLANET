import json, os, pickle, random
from typing import List, Optional, Set
from torch.utils.data import Dataset
from chemutils import ComplexPocket, tensorize_all
import numpy as np
from itertools import chain


class ProLigDataset(Dataset):
    """
    Dataset for PLANET training/evaluation.

    Scans data_dir for <pdb>/<pdb>_pocket.pkl files at runtime —
    no intermediate index pkl needed, paths are never stored.

    Args:
        data_dir   : directory with <pdb>/<pdb>_pocket.pkl structure
        pk_json    : path to {pdb_code: pK} JSON (e.g. pk_v2019.json)
        split      : 'train' | 'valid' | 'all'
        exclude_ids: PDB codes to skip (e.g. CASF test set)
        valid_frac : fraction of data used for validation split
        seed       : random seed for reproducible train/valid split
        batch_size : complexes per batch
        shuffle    : shuffle records before batching (training only)
        decoy_flag : use random decoy ligands during training
    """

    def __init__(self, data_dir: str, pk_json: str,
                 split: str = 'all',
                 exclude_ids: Optional[Set[str]] = None,
                 valid_frac: float = 0.1,
                 seed: int = 42,
                 batch_size: int = 16,
                 shuffle: bool = True,
                 decoy_flag: bool = True):

        with open(pk_json) as f:
            pk_data = json.load(f)

        exclude_ids = {x.lower() for x in (exclude_ids or set())}

        records = []
        for pdb in os.listdir(data_dir):
            if pdb.lower() in exclude_ids:
                continue
            pkl_path = os.path.join(data_dir, pdb, f'{pdb}_pocket.pkl')
            if not os.path.exists(pkl_path):
                continue
            pK = pk_data.get(pdb.lower(), 0.0)
            records.append((pkl_path, pK))

        # deterministic train/valid split
        rng = random.Random(seed)
        rng.shuffle(records)
        n_valid = int(len(records) * valid_frac)
        if split == 'train':
            records = records[n_valid:]
        elif split == 'valid':
            records = records[:n_valid]
        # 'all' → keep everything

        if shuffle:
            while True:
                random.shuffle(records)
                batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
                if self._check(batches):
                    break
        else:
            batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]

        self.batches = batches
        self.decoy_flag = decoy_flag

    def _check(self, batches):
        for batch in batches:
            if np.sum([float(r[1]) for r in batch]) == 0:
                return False
        return True

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self._tensorize(idx)

    def _tensorize(self, idx):
        pocket_batch = []
        for (pkl_path, _) in self.batches[idx]:
            with open(pkl_path, 'rb') as f:
                pocket_batch.append(pickle.load(f))
        res_feature_batch, mol_feature_batch, mol_interactions, pro_lig_interactions, pKs, pK_flags, complex_labels = \
            tensorize_all(pocket_batch, self.decoy_flag)
        return res_feature_batch, mol_feature_batch, \
               (mol_interactions, pro_lig_interactions, pKs, pK_flags, complex_labels)

    def get_bonded_atom_pairs(self) -> List[List[tuple]]:
        bonded_pairs = []
        for (pkl_path, _) in chain(*self.batches):
            with open(pkl_path, 'rb') as f:
                pocket: ComplexPocket = pickle.load(f)
                bonded_pairs.append(pocket.ligand.get_bonded_atoms())
        return bonded_pairs
