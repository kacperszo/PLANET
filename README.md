# PLANET

**P**rotein-**L**igand **A**ffinity prediction **NET**work — a graph neural network that predicts binding affinity from a protein pocket graph and a 2D ligand graph, without requiring exhaustive docking conformational sampling.

PLANET was trained on PDBbind v2019 with three simultaneous objectives: binding affinity regression, protein-ligand contact map prediction, and intra-ligand distance matrix prediction. On the CASF-2016 benchmark it matches state-of-the-art 3D complex-based models while running at a fraction of the compute cost — making it practical for large-scale virtual screening.

Original paper: [PLANET: A Multi-Objective Graph Neural Network Model for Protein–Ligand Binding Affinity Prediction](https://doi.org/10.1021/acs.jcim.2c01085)

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

- PDBbind general set (v2019 recommended) — available at <http://pdbbind.org.cn/>
- CASF-2016 core set (used as the held-out test set throughout training)

### Step 1 — preprocess structures

Converts each PDBbind entry into a self-contained HDF5 pocket file (`<pdb>_pocket.h5`).
pK values are parsed directly from the PDBbind INDEX file — no intermediate JSON needed.

```bash
uv run preprocess.py \
    -d $PDBBIND_DIR \
    -i $PDBBIND_DIR/index/INDEX_general_PL_data.2019 \
    -n 16
```

Do the same for the CASF-2016 core set:

```bash
uv run preprocess.py \
    -d $CASF_DIR \
    -i $PDBBIND_DIR/index/INDEX_general_PL_data.2019 \
    -n 8
```

Each entry directory should contain `<pdb>_ligand.sdf`, `<pdb>_protein.pdb`, and optionally `<pdb>_decoy.sdf`.
After preprocessing, each directory will also contain `<pdb>_pocket.h5`.

- **Protein:** fix broken residues and assign protonation states (e.g. Maestro `prepwizard`). **α-carbon of every residue must be present** — PLANET uses Cα positions for the protein graph.
- **Ligands:** add hydrogens and assign ionisation states (e.g. Maestro `epik`, or RDKit).

### Step 2 — train

The dataset is built on-the-fly from the HDF5 files — no intermediate index files needed.
CASF entries are automatically excluded from train/valid and used as the running test set.

```bash
uv run train.py \
    -d $PDBBIND_DIR \
    -c $CASF_DIR \
    -s checkpoints/
```

To resume from a checkpoint:

```bash
uv run train.py \
    -d $PDBBIND_DIR \
    -c $CASF_DIR \
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
| `--valid_frac` | 0.1 | Fraction of PDBbind held out for validation |
| `--checkpoint` | — | Path to checkpoint to resume from |
| `--initial_step` | 0 | Global step to start from (set when resuming) |

### Step 3 — evaluate on CASF-2016

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
