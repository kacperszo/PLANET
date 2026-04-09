# PLANET

**P**rotein-**L**igand **A**ffinity prediction **NET**work — a graph neural network that predicts binding affinity from a protein pocket graph and a 2D ligand graph, without requiring exhaustive docking conformational sampling.

PLANET was trained on PDBbind v2019 with three simultaneous objectives: binding affinity regression, protein-ligand contact map prediction, and intra-ligand distance matrix prediction. On the CASF-2016 benchmark it matches state-of-the-art 3D complex-based models while running at a fraction of the compute cost — making it practical for large-scale virtual screening.

Original paper: [PLANET: A Multi-Objective Graph Neural Network Model for Protein–Ligand Binding Affinity Prediction](https://doi.org/10.1021/acs.jcim.2c01085)

---

## How it works

PLANET takes as input a **protein pocket** (residue sequence + Cα coordinates only — no side chains, no docked pose) and a **2D ligand graph** (atoms + bonds), and predicts binding affinity in pK units.

```
Protein pocket                       Ligand (2D graph)
  residues × BLOSUM62 (20-dim)         atoms × physicochemical features
  + Cα coordinates                     + bond features
        │                                     │
   ProteinEGNN                           LigandGAT
  (E(n)-equivariant                  (graph attention,
   message passing,                   bond-message
   3 iterations)                      passing, 10 iters)
        │                                     │
        └──────── ProteinLigandAttention ──────┘
                  (bidirectional cross-attention
                   between residues and atoms,
                   1 iteration)
                         │
                       ProLig
                  (element-wise product
                   of protein × ligand features)
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   intra-ligand    protein-ligand    binding
   distance map    contact map       affinity (pK)
```

**ProteinEGNN** encodes the pocket using Cα–Cα squared distances as edge features, making it invariant to rotation and translation. **LigandGAT** encodes the ligand using a bond-centric message passing scheme. **ProteinLigandAttention** lets residue and atom representations update each other via cross-attention. The final **ProLig** head predicts all three outputs from element-wise products of the cross-attended features.

The model is trained with three simultaneous objectives:
1. **Affinity regression** — MSE loss on pK values (masked where pK = 0)
2. **Protein-ligand contact prediction** — BCE loss on atom–residue contact labels (4 Å threshold)
3. **Intra-ligand distance prediction** — BCE loss on atom–atom distance labels

The auxiliary tasks act as structural regularisers, forcing the model to learn geometrically meaningful representations without requiring a docked pose as input.

---

## Installation

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

By default this pulls PyTorch with **CUDA 12.1** support. Edit the index URL in `pyproject.toml` before running `uv sync` to change this:

| Target | URL |
|--------|-----|
| CPU only | `https://download.pytorch.org/whl/cpu` |
| CUDA 12.4 | `https://download.pytorch.org/whl/cu124` |

---

## Training on PDBbind

### Prerequisites

The recommended dataset is the **PLANET_dataset** provided by the original authors (PDBbind 2020 with pre-generated decoy ligands and pre-defined train/valid/test splits). It is available on the PDBbind website alongside the paper.

Alternatively, any PDBbind release can be used with a random train/valid split.

### Step 1 — preprocess structures

Converts each PDBbind entry into a self-contained HDF5 pocket file (`<pdb>_pocket.h5`).
pK values are parsed directly from the PDBbind INDEX file.
If a ligand SDF fails to parse (e.g. invalid valence), the script automatically falls back to the `.mol2` file.

```bash
# Combine all index files into one for preprocessing
cat PLANET_dataset/index/PDBbind_PLANET_TrainSet.2020 \
    PLANET_dataset/index/PDBbind_PLANET_ValidSet.2020 \
    PLANET_dataset/index/PDBbind_PLANET_TestSet.2020 \
    > PLANET_dataset/index/ALL.2020

uv run preprocess.py \
    -d PLANET_dataset/PDBbind2020-PLANET \
    -i PLANET_dataset/index/ALL.2020 \
    -n 16
```

Each entry directory must contain `<pdb>_ligand.sdf` (or `<pdb>_ligand.mol2` as fallback) and `<pdb>_protein.pdb`. Decoy ligands (`<pdb>_decoy.sdf`) are loaded automatically when present — they are part of the multi-objective training signal.

Add `--skip-existing` to resume an interrupted preprocessing run.

### Step 2 — train

**Explicit split mode** (recommended — uses pre-defined index files, reproduces paper setup):

```bash
uv run train.py \
    -d PLANET_dataset/PDBbind2020-PLANET \
    -c PLANET_dataset/PDBbind2020-PLANET \
    --train_index PLANET_dataset/index/PDBbind_PLANET_TrainSet.2020 \
    --valid_index PLANET_dataset/index/PDBbind_PLANET_ValidSet.2020 \
    --test_index  PLANET_dataset/index/PDBbind_PLANET_TestSet.2020 \
    -s checkpoints/
```

**Random split mode** (fallback when no index files are available — excludes CASF entries from training):

```bash
uv run train.py \
    -d $PDBBIND_DIR \
    -c $CASF_DIR \
    -s checkpoints/
```

To resume from a checkpoint:

```bash
uv run train.py \
    -d PLANET_dataset/PDBbind2020-PLANET \
    -c PLANET_dataset/PDBbind2020-PLANET \
    --train_index PLANET_dataset/index/PDBbind_PLANET_TrainSet.2020 \
    --valid_index PLANET_dataset/index/PDBbind_PLANET_ValidSet.2020 \
    --test_index  PLANET_dataset/index/PDBbind_PLANET_TestSet.2020 \
    -s checkpoints/ \
    --checkpoint checkpoints/PLANET.iter-50000 \
    --initial_step 50000
```

Key training arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--epoch` | 250 | Number of epochs |
| `--batch_size` | 16 | Complexes per batch |
| `--lr` | 1e-4 | Initial learning rate |
| `--anneal_iter` | 20000 | LR decay interval in steps (after step 60k) |
| `--save_iter` | 5000 | Checkpoint + eval interval (steps) |
| `--print_iter` | 200 | Log interval (steps) |
| `--train_index` | — | Index file for training set (explicit split mode) |
| `--valid_index` | — | Index file for validation set (explicit split mode) |
| `--test_index` | — | Index file for test set (explicit split mode) |
| `--valid_frac` | 0.1 | Fraction held out for validation (random split mode) |
| `--checkpoint` | — | Path to checkpoint to resume from |
| `--initial_step` | 0 | Global step to start from (set when resuming) |

### Step 3 — evaluate

```bash
uv run evaluate.py \
    -f checkpoints/PLANET.iter-XXXXX \
    -c $CASF_DIR \
    -o results/casf
```

Prints MAE, RMSE, Pearson R, Spearman ρ, and Concordance Index (CI).
Saves predictions to `results/casf.h5` (arrays) and `results/casf_meta.json` (scopes + bonded pairs).

---

## Virtual screening

Score a library of molecules against a protein pocket using a trained checkpoint.
The pocket can be defined either by a crystal ligand SDF or by explicit coordinates.

```bash
# pocket defined by crystal ligand
uv run screen.py \
    -p protein.pdb \
    -l crystal_ligand.sdf \
    -m library.sdf \
    -w checkpoints/PLANET.iter-100000 \
    --prefix result

# pocket defined by centre coordinates
uv run screen.py \
    -p protein.pdb \
    -x 12.3 -y 45.6 -z 78.9 \
    -m library.smi \
    -w checkpoints/PLANET.iter-100000 \
    --prefix result
```

Outputs `result.csv` and `result.sdf` with predicted affinities (pK units).

`screen.py` is also installed as a `planet-screen` entry point:

```bash
planet-screen -p protein.pdb -l ligand.sdf -m library.sdf \
    -w checkpoints/PLANET.iter-100000
```

| Flag | Description |
|------|-------------|
| `-p / --protein` | Protein structure file (`.pdb`) |
| `-l / --ligand` | Crystal ligand SDF — defines pocket centre |
| `-x/-y/-z` | Pocket centre coordinates (alternative to `-l`) |
| `-m / --mol_file` | Molecules to score (`.sdf` or `.smi`) |
| `-w / --checkpoint` | Trained model checkpoint |
| `--prefix` | Output file prefix (default: `result`) |

---

## Data format

Each preprocessed complex is stored as `<pdb>_pocket.h5` (gzip-compressed HDF5):

| Key | Shape | Description |
|-----|-------|-------------|
| `res_features` | `[n_res, 20]` | BLOSUM62 residue features |
| `alpha_coordinates` | `[n_res, 3]` | Cα positions |
| `pro_lig_interaction` | `[n_atoms, n_res]` | Contact labels (4 Å threshold) |
| `ligand_mol` | `[N]` uint8 | RDKit molecule binary |
| `decoys/0..k` | `[M]` uint8 | Decoy molecule binaries |
| attrs: `pK` | float | Binding affinity |
| attrs: `decoys_count` | int | Number of decoys |

---

## Repository layout

```
train.py          — training loop
evaluate.py       — CASF-2016 evaluation
screen.py         — virtual screening inference
preprocess.py     — preprocessing: PDBbind → per-complex HDF5
planet/
  model.py        — PLANET nn.Module
  data.py         — ProLigDataset (on-the-fly HDF5 loader)
  chem.py         — featurisation, ComplexPocket, HDF5 I/O
  layers.py       — ProteinEGNN, LigandGAT, ProLig layers
  utils.py        — tensor utilities
```
