# PLANET — affinity from cross-attention between two encoders

> A fork maintained for [gnn-benchmark](../../README.md). The authors' own README is
> kept as [README.upstream.md](README.upstream.md) for attribution and for their
> description of the method — **its build and run instructions are not current for
> this fork.**

## What it is

An E(n)-equivariant ProteinEGNN over pocket residues and a LigandGAT over the ligand
graph, joined by cross-attention. Trained on three objectives at once: affinity, a contact map, and
intra-ligand distances. The heads run per atom-residue pair and pooling collapses them to scalars, so
no complex-level vector exists in the architecture and the embedding is ours.

## State

| | |
|---|---|
| CASF-2016 scoring | **R 0.765**, RMSE 1.667, n=285 |
| embedding | 600d, **ours**, probe R 0.682 — 89% of its own head |
| `gnnb verify` | 285/285, 3.8e-06 |
| invariance | the best-conditioned model here — 1.4e-06 under an exact rotation |

Trained on PDBbind 2020, so anything scored with it has to come from the same preprocessing
provenance. The checkpoint is iter-145000: best by validation Pearson R, and also best on the
authors' test set and on CASF.

## Build

```bash
podman build --format=docker -t planet:latest .
```

## Run it, without the harness

Generated from this model's adapter by `gnnb howto`, so these are the exact commands
the benchmark issues — regenerate with `python tools/sync_model_readmes.py`. Every one
runs with `--network=none` and a read-only root filesystem.

Input is one directory per complex:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf      # or .mol2; several models try both

Ligand 3D coordinates are read during preprocessing but only to build training targets; the authors trained with `use_rdkit_coords: true`, so the ligand enters as an RDKit conformer.

```bash
# planet.modern — localhost/planet:latest
# source: models/planet

# predict
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/checkpoints_2020:/ckpt:ro" \
    localhost/planet:latest \
    sh -c 'set -e; mkdir -p /outputs/_staged; cp -r /data/. /outputs/_staged/; : > /outputs/_empty_index; python preprocess.py -d /outputs/_staged -i /outputs/_empty_index -n 8; python predict_complexes.py --complexes /outputs/_staged --model /ckpt/PLANET.iter-145000 --out /outputs/predictions.csv --device cpu; rm -rf /outputs/_staged /outputs/_empty_index'

# embed
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/checkpoints_2020:/ckpt:ro" \
    localhost/planet:latest \
    sh -c 'set -e; mkdir -p /outputs/_staged; cp -r /data/. /outputs/_staged/; : > /outputs/_empty_index; python preprocess.py -d /outputs/_staged -i /outputs/_empty_index -n 8; python embed_complexes.py --complexes /outputs/_staged --model /ckpt/PLANET.iter-145000 --out /outputs/embeddings.npz --pool sum --device cpu; rm -rf /outputs/_staged /outputs/_empty_index'

# train
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/checkpoints_2020:/ckpt:ro" \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/planet:latest \
    sh -c 'cd /work && python prepare_h5.py --complexes /data --index /data/index/INDEX_general_PL_data.2019 --labels /splits/train.csv --out /cache/planet_h5 --skip-existing && python prepare_h5.py --complexes /data --index /data/index/INDEX_general_PL_data.2019 --labels /splits/val.csv --out /cache/planet_h5 --skip-existing && python train_contract.py --data-dir /cache/planet_h5 --train-labels /splits/train.csv --out /outputs --seed 0 --val-labels /splits/val.csv --epochs 30 --device cpu'

# finetune  (encoder frozen; drop --freeze-encoder to tune all of it)
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/checkpoints_2020:/ckpt:ro" \
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
| `embeddings.npz` | `ids` and `vectors`, 600-dim — pooled cross-attention output for each side, concatenated. Ours: the heads run per atom-residue pair and pooling collapses them to scalars, so no complex vector exists in the architecture |

## Before you trust the numbers

Trained on PDBbind 2020, so anything scored with it has to come from the same preprocessing provenance. The checkpoint used here is iter-145000, best by validation Pearson R and also best on the authors' test set and on CASF — chosen without cherry-picking.

## Maintainer notes

`CLAUDE.md` in this directory holds what breaks if it is changed back.
