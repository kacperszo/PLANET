# PLANET — affinity from cross-attention between two encoders

> A maintained fork. The authors' own README is kept as
> [README.upstream.md](README.upstream.md) for attribution and for their description of
> the method — **its build and run instructions are not current for this fork.**

## What it is

An E(n)-equivariant ProteinEGNN over pocket residues and a LigandGAT over the ligand
graph, joined by cross-attention. Trained on three objectives at once: affinity, a contact map, and
intra-ligand distances. The heads run per atom-residue pair and pooling collapses them to scalars, so
no complex-level vector exists in the architecture and the embedding is ours.


## Weights and reported performance

No usable checkpoint ships here. The authors' `PLANET.param` was removed from this fork
(commit `2e9bea8`) and `.gitignore` excludes `*.param` and `checkpoints_2020/`, so you must
supply one: recover the authors' file from their repository
(https://github.com/ComputArtCMCG/PLANET) or train your own with `train_contract.py` (the
`train` command below writes `model.pt`; point `--model` at it).

A retrain from this fork exists, but its protocol differs from the authors': the authors used a
pre-cleaned PDBbind with decoy non-binders and their own splits, whereas this fork trained on
PDBbind 2020 with its own splits, and the multi-objective training with decoys is only partially
reproduced. A figure from that retrain would not be comparable with the paper's, so none is
claimed here.

Anything scored with a PDBbind-2020-trained checkpoint has to come from the same preprocessing
provenance (see Porting notes).

## Build

```bash
podman build --format=docker -t planet:latest .
```

## Run it

Every command below runs with `--network=none` and a read-only root filesystem. `CKPT_DIR`
is a directory you supply holding a checkpoint (see above); `train` writes `model.pt` into its
output directory, which can serve as `CKPT_DIR`.

Input is one directory per complex:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf      # or .mol2

Ligand 3D coordinates are read during preprocessing but only to build training targets; the authors trained with `use_rdkit_coords: true`, so the ligand enters as an RDKit conformer.

```bash
# predict   (CKPT_DIR/model.pt; use your own file name)
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$CKPT_DIR:/ckpt:ro" \
    localhost/planet:latest \
    sh -c 'set -e; mkdir -p /outputs/_staged; cp -r /data/. /outputs/_staged/; : > /outputs/_empty_index; python preprocess.py -d /outputs/_staged -i /outputs/_empty_index -n 8; python predict_complexes.py --complexes /outputs/_staged --model /ckpt/model.pt --out /outputs/predictions.csv --device cpu; rm -rf /outputs/_staged /outputs/_empty_index'

# embed
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$CKPT_DIR:/ckpt:ro" \
    localhost/planet:latest \
    sh -c 'set -e; mkdir -p /outputs/_staged; cp -r /data/. /outputs/_staged/; : > /outputs/_empty_index; python preprocess.py -d /outputs/_staged -i /outputs/_empty_index -n 8; python embed_complexes.py --complexes /outputs/_staged --model /ckpt/model.pt --out /outputs/embeddings.npz --pool sum --device cpu; rm -rf /outputs/_staged /outputs/_empty_index'

# train   (writes model.pt, history.csv, summary.json to /outputs)
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/planet:latest \
    sh -c 'cd /work && python prepare_h5.py --complexes /data --index /data/index/INDEX_general_PL_data.2019 --labels /splits/train.csv --out /cache/planet_h5 --skip-existing && python prepare_h5.py --complexes /data --index /data/index/INDEX_general_PL_data.2019 --labels /splits/val.csv --out /cache/planet_h5 --skip-existing && python train_contract.py --data-dir /cache/planet_h5 --train-labels /splits/train.csv --out /outputs --seed 0 --val-labels /splits/val.csv --epochs 30 --device cpu'

# finetune  (encoder frozen; drop --freeze-encoder to tune all of it)
# ENC_DIR holds encoder.pt: a state dict of encoder tensors only (see Porting notes)
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$ENC_DIR:/ckpt:ro" \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/planet:latest \
    sh -c 'cd /work && python prepare_h5.py --complexes /data --index /data/index/INDEX_general_PL_data.2019 --labels /splits/train.csv --out /cache/planet_h5 --skip-existing && python prepare_h5.py --complexes /data --index /data/index/INDEX_general_PL_data.2019 --labels /splits/val.csv --out /cache/planet_h5 --skip-existing && python train_contract.py --data-dir /cache/planet_h5 --train-labels /splits/train.csv --out /outputs --seed 0 --val-labels /splits/val.csv --epochs 30 --init-encoder /ckpt/encoder.pt --freeze-encoder --device cpu'
```

## What comes out

| file | holds |
|---|---|
| `predictions.csv` | `complex_id,y_pred` |
| `embeddings.npz` | `ids` and `vectors`, 600-dim: pooled cross-attention output for each side, concatenated. The pooling is this fork's, not the authors': the heads run per atom-residue pair and pooling collapses them to scalars, so no complex vector exists in the architecture |

## Porting notes

What this fork changed: the environment moved from conda/Python 3.6 to uv/Python 3.13 with
torch 2.5.1; `os.system()` calls were removed; `torch.load` uses `weights_only=True`; the code
now lives in the `planet/` package. `train_contract.py` and `prepare_h5.py` are new.

Things to know before reusing it:

- **The same PDB entry from two sources is not the same input, and the model notices.**
  Preprocessing CASF-2013 straight from its own `coreset/` directory scored MAE 2.87, R 0.22
  with a PDBbind-2020-trained checkpoint. The `_pocket.h5` files differ from the PDBbind-2020 ones
  for the same complexes (`res_features`, `alpha_coordinates`, `pro_lig_interaction` and
  `ligand_mol` hash differently; 3su2 has 43 vs 41 pocket residues, 3myg 57 vs 56 ligand atoms).
  Routing the same complexes through the PDBbind-2020 files restored R to 0.753. Score a dataset
  from the same preprocessing provenance the checkpoint was trained on. CASF-2016 was close enough
  that the problem did not appear there.
- **`preprocess.py` writes `<id>_pocket.h5` next to its inputs**, so it cannot run on a read-only
  mount. The run commands stage a copy into `/outputs` first. `prepare_h5.py` avoids the copy by
  reading from the read-only directory and writing a separate cache.
- **Roughly 43% of PDBbind v2019 will not preprocess.** About 10,100 of 17,600 general-set entries
  survive; the rest are SDFs RDKit refuses over valence errors. This is a property of PDBbind, but
  it silently shrinks any training set built from it. The training contract now refuses a split
  containing complexes that failed to preprocess, rather than shrinking it.
- **`CA` is selected by atom name, and only `ATOM` records are parsed.** `planet/chem.py` picks
  alpha carbons with `line[12:16].strip() == "CA"`, which would also match calcium ions, except that
  only lines starting with `ATOM` are kept and calcium sits in `HETATM`. Loosening that filter to
  include `HETATM` (say, to keep cofactors) would silently add calcium ions to the pocket as
  residues with a fake alpha carbon. No error, just phantom residues.
- **Missing pK is 0.** `preprocess.py` writes pK=0 for complexes absent from its index, which is
  what makes label-free scoring possible; `evaluate.py` cannot do that, since it computes metrics
  against ground truth. The supervised training contract rejects zero, missing, duplicate and
  non-finite labels, labels that disagree between the CSV, the index and the cached HDF5, and
  complexes that were never prepared.
- **Training contract (repairs of 2026-10-20).** Discovered HDF5 records are sorted and shuffled
  with `seed + epoch`, so the order is reproducible and not reset to the same order each epoch.
  Validation propagates failures, rejects non-finite loss and empty results, restores model mode
  on error, and cannot select a zero-loss "best" checkpoint after a failed run. Invalid settings
  and non-finite objectives or gradient norms are rejected. The architecture, inference
  featurisation, pooling, loss weights, the 500-step affinity warm-up and the author
  initialisation are unchanged. Behaviour is covered by `test_training_contract.py`.
- **Encoder boundary.** The checkpoint holds `proteinegnn`, `ligandgat` and `prolig` and nothing
  else. The embedding is taken after `prolig.prolig_attention`, so the cross-attention counts as
  encoder; the three prediction heads (affinity, contact map, intra-ligand distances) are the
  `HEAD_PREFIXES` in `planet/transfer.py`. To make an `encoder.pt`, load a checkpoint's state dict
  and drop every key for which `planet.transfer.is_head` is true (29 encoder tensors remain, 20
  head tensors are removed). Prefixes match at module boundaries: `prolig.linear_lig_interaction`
  must not claim `prolig.prolig_attention`.
- **The embedding is this fork's.** `embed_complexes.py` replays the forward to
  `prolig_attention` and pools `fatoms` and `fresidues` per complex (2 x 300 = 600 dims). Another
  pooling point is possible; this one is a choice, not the authors'.
- **Training is a restricted protocol.** Decoy non-binders, which the authors trained with, are
  absent, so a model trained here is not a faithful retraining of the paper's.
- **Containers.** `COPY . /work` bakes in whatever is not in `.containerignore`; weights
  directories are excluded and must be mounted at run time. Podman caps `/dev/shm` at 64 MB, so
  anything spawning data-loading workers needs `--shm-size 4g` or it fails partway through with
  "No space left on device".
