import os, random
from typing import List, Optional, Set
from torch.utils.data import Dataset
from planet.chem import ComplexPocket, tensorize_all
import numpy as np
from itertools import chain


class ProLigDataset(Dataset):
    """
    Dataset for PLANET training/evaluation.

    Scans data_dir for <pdb>/<pdb>_pocket.h5 files at runtime.
    pK values are read directly from the stored HDF5 attributes.

    Two split modes:
    - Explicit: pass ``pdb_ids`` — only those PDB codes are used, no random split.
      Used when pre-defined train/valid/test index files are available.
    - Random: omit ``pdb_ids`` — directory is scanned, ``exclude_ids`` are dropped,
      and the remainder is split randomly into train/valid by ``valid_frac``.

    Args:
        data_dir   : directory with <pdb>/<pdb>_pocket.h5 structure
        pdb_ids    : explicit set of PDB codes to use (overrides split/exclude/valid_frac)
        split      : 'train' | 'valid' | 'all' — used only when pdb_ids is None
        exclude_ids: PDB codes to skip — used only when pdb_ids is None
        valid_frac : fraction of data for validation — used only when pdb_ids is None
        seed       : random seed for reproducible train/valid split
        batch_size : complexes per batch
        shuffle    : shuffle records before batching (set True for training)
        decoy_flag : use random decoy ligands during training
    """

    def __init__(self, data_dir: str,
                 pdb_ids: Optional[Set[str]] = None,
                 split: str = 'all',
                 exclude_ids: Optional[Set[str]] = None,
                 valid_frac: float = 0.1,
                 seed: int = 42,
                 batch_size: int = 16,
                 shuffle: bool = True,
                 decoy_flag: bool = True):

        if pdb_ids is not None:
            # explicit mode: use exactly the given PDB codes
            pdb_ids = {x.lower() for x in pdb_ids}
            records = []
            for pdb in os.listdir(data_dir):
                if pdb.lower() not in pdb_ids:
                    continue
                h5_path = os.path.join(data_dir, pdb, f'{pdb}_pocket.h5')
                if not os.path.exists(h5_path):
                    continue
                records.append(h5_path)
        else:
            # random split mode
            exclude_ids = {x.lower() for x in (exclude_ids or set())}
            records = []
            for pdb in os.listdir(data_dir):
                if pdb.lower() in exclude_ids:
                    continue
                h5_path = os.path.join(data_dir, pdb, f'{pdb}_pocket.h5')
                if not os.path.exists(h5_path):
                    continue
                records.append(h5_path)

            rng = random.Random(seed)
            rng.shuffle(records)
            n_valid = int(len(records) * valid_frac)
            if split == 'train':
                records = records[n_valid:]
            elif split == 'valid':
                records = records[:n_valid]

        rng = random.Random(seed)
        if shuffle:
            max_attempts = 1000
            for attempt in range(max_attempts):
                rng.shuffle(records)
                batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
                if self._check(batches):
                    break
            else:
                raise RuntimeError(
                    f"Could not build valid batches after {max_attempts} shuffle attempts. "
                    "Too many records with pK=0?"
                )
        else:
            batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]

        self.batches = batches
        self.decoy_flag = decoy_flag

    def _check(self, batches):
        for batch in batches:
            pks = [ComplexPocket.read_pk(p) for p in batch]
            if np.sum(pks) == 0:
                return False
        return True

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self._tensorize(idx)

    def _tensorize(self, idx):
        pocket_batch = [ComplexPocket.load_h5(p) for p in self.batches[idx]]
        res_feature_batch, mol_feature_batch, mol_interactions, pro_lig_interactions, pKs, pK_flags, complex_labels = \
            tensorize_all(pocket_batch, self.decoy_flag)
        return res_feature_batch, mol_feature_batch, \
               (mol_interactions, pro_lig_interactions, pKs, pK_flags, complex_labels)

    def get_bonded_atom_pairs(self) -> List[List[tuple]]:
        bonded_pairs = []
        for h5_path in chain(*self.batches):
            pocket = ComplexPocket.load_h5(h5_path)
            bonded_pairs.append(pocket.ligand.get_bonded_atoms())
        return bonded_pairs
