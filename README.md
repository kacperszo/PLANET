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

## Quick start — virtual screening

A demo is included (`demo/`): an ADRB2 receptor (`adrb2.pdb`), its crystal ligand (`adrb2_ligand.sdf`) used to define the binding pocket, and a set of molecules to score (`mols.sdf`).

```bash
cd demo
uv run ../PLANET_run.py -p adrb2.pdb -l adrb2_ligand.sdf -m mols.sdf
```

Outputs `result.csv` and `result.sdf` with predicted affinities (pK units).

### Inference arguments

| Flag | Description |
|------|-------------|
| `-p / --protein` | Protein structure file (`.pdb`) |
| `-l / --ligand` | Crystal ligand SDF — defines pocket centre; overrides `-x/-y/-z` |
| `-x/-y/-z` | Pocket centre coordinates (alternative to `-l`) |
| `-m / --mol_file` | Molecules to score (`.sdf` or `.smi`) |
| `--prefix` | Output file prefix (default: `result`) |

### Preparing input files

- **Protein:** fix broken residues and assign protonation states (e.g. Maestro `prepwizard`). **α-carbon of every residue must be present** — PLANET uses Cα positions for the protein graph.
- **Ligands:** add hydrogens and assign ionisation states (e.g. Maestro `epik`, or RDKit).

---

## Training on PDBbind

### Prerequisites

- PDBbind general set (v2019 recommended) — available at <http://pdbbind.org.cn/>
- CASF-2016 core set (used as held-out test set throughout training)
- A JSON file mapping PDB codes to pK values — generate with:

```bash
uv run scripts/make_pk_json.py --index $PDBBIND/index/INDEX_general_PL_data.2019 \
    --out pk_v2019.json
```

### Step 1 — preprocess structures

Converts each PDBbind entry into a self-contained HDF5 pocket file (`<pdb>_pocket.h5`).
Run in parallel with `-n` workers:

```bash
uv run process_PDBBind.py \
    -d $PDBBIND_DIR \
    -n 16 \
    -k pk_v2019.json
```

Do the same for the CASF-2016 core set directory:

```bash
uv run process_PDBBind.py \
    -d $CASF_DIR \
    -n 8 \
    -k pk_v2019.json
```

Each entry directory should contain `<pdb>_ligand.sdf`, `<pdb>_protein.pdb`, and optionally `<pdb>_decoy.sdf`.  
After preprocessing, each directory will also contain `<pdb>_pocket.h5`.

> **Note:** The HDF5 format (via `h5py`) replaced the previous pickle-based format. Existing `_pocket.pkl` files are no longer compatible — rerun `process_PDBBind.py` to regenerate.

### Step 2 — train

The dataset is built on-the-fly from the directory structure — no intermediate index files needed.
CASF entries are automatically excluded from train/valid and used as the running test set.

```bash
uv run PLANET_train.py \
    -d $PDBBIND_DIR \
    -c $CASF_DIR \
    -k pk_v2019.json \
    -s checkpoints/
```

Key training arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--epoch` | 250 | Number of epochs |
| `--batch_size` | 16 | Complexes per batch |
| `--lr` | 1e-4 | Learning rate |
| `--anneal_iter` | 20000 | LR decay interval (after step 60k) |
| `--save_iter` | 5000 | Checkpoint + eval interval (steps) |
| `--print_iter` | 200 | Log interval (steps) |
| `--initial_step` | 0 | Resume from step (pair with `--load_epoch`) |

### Step 3 — evaluate on CASF-2016

```bash
uv run PLANET_test.py \
    -f checkpoints/PLANET.iter-XXXXX \
    -c $CASF_DIR \
    -k pk_v2019.json \
    -o results/casf
```

Prints MAE, RMSE, Pearson R, Spearman ρ, and Concordance Index (CI).  
Saves predictions to `results/casf.h5` (arrays) and `results/casf_meta.json` (scopes + bonded pairs).

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
PLANET_run.py          — virtual screening inference
PLANET_train.py        — training loop
PLANET_test.py         — CASF-2016 evaluation
PLANET_model.py        — model definition (PLANET nn.Module)
PLANET_datautils.py    — ProLigDataset (on-the-fly HDF5 loader)
chemutils.py           — featurisation, ComplexPocket, HDF5 I/O
layers.py              — ProteinEGNN, LigandGAT, ProLig layers
nnutils.py             — tensor utilities
process_PDBBind.py     — preprocessing (pkl → h5 per complex)
scripts/
  make_pk_json.py      — build pk_v2019.json from PDBbind index
  build_datasets_v2019.py — (legacy) static dataset builder, superseded
demo/                  — example protein + ligands for quick test
PLANET.param           — pretrained weights
```
