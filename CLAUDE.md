# CLAUDE.md — PLANET

## What this is

Affinity prediction from a **protein pocket (residue sequence + Cα coordinates only)** and a
**2D ligand graph**. No docked pose, no side chains. Fork of the authors' repo, reproduced
for the gnn-benchmark harness, where it runs as `planet.modern`.

Trained with three simultaneous objectives — affinity, protein–ligand contact map, and
intra-ligand distance map — the last two acting as structural regularisers.

## Current state

Runs end to end through the harness. On the CASF-2016 core set with
`checkpoints_2020/PLANET.iter-145000`: **RMSE 1.667, MAE 1.303, R 0.765, CI 0.803**, all 285
complexes preprocessed without failures.

`iter-145000` was chosen by Pearson R on the validation set and is also best on the authors'
test set and on CASF, so it is not a cherry-pick. Its own reference numbers, from
2026-04-13: CASF-2013 R 0.753, authors' PDBbind-2020 test set R 0.791.

Two weight sets are present. `checkpoints_2020/` is the one to use — trained on the authors'
PDBbind-2020 dataset. `checkpoints/` is an earlier run on PDBbind v2019 and tops out around
iteration 141000 at R 0.654 on CASF-2013.

## Hard-won facts (do NOT regress these)

- **The same PDB entry from two sources is not the same input, and the model notices.**
  Preprocessing CASF-2013 straight from `CASF-2013/coreset/` scored **MAE 2.87, R 0.22**.
  The `_pocket.h5` files differ byte-wise from the PDBbind-2020 ones for the same complexes —
  `res_features`, `alpha_coordinates`, `pro_lig_interaction` and `ligand_mol` all hash
  differently, and some complexes differ in size outright (3su2: 43 vs 41 pocket residues;
  3myg: 57 vs 56 ligand atoms). Routing the same complexes through the PDBbind-2020 files
  restored R to 0.753. **A dataset scored with this checkpoint must come from the same
  preprocessing provenance it was trained on.** CASF-2016 happens to be close enough that
  the problem does not appear there.
- **`preprocess.py` writes `<id>_pocket.h5` next to its inputs**, so it cannot run against
  the harness's read-only `/data`. The adapter stages a copy into `/outputs` first. That is
  deliberate — no model may mutate the corpus every other model is scored on.
- **Roughly 43% of PDBbind v2019 will not preprocess.** About 10,100 of 17,600 general-set
  entries survive; the rest are SDFs RDKit refuses over valence errors. This is a known
  property of PDBbind, not a bug here, but it silently shrinks any training set built from it.
- **`CA` is selected by atom name, and only `ATOM` records are parsed.** `chem.py:100` picks
  alpha carbons with `line[12:16].strip() == "CA"`, which would also match calcium ions —
  except that `chem.py:119` and `:151` keep only lines starting with `ATOM`, and calcium sits
  in `HETATM`. Loosening that filter to include `HETATM`, an obvious-looking change if someone
  ever wants cofactors, would silently add calcium ions to the pocket as residues with a fake
  alpha carbon. No error, just a few phantom residues.
- **`preprocess.py` defaults pK to 0** for complexes missing from its index, which is what
  makes label-free scoring possible at all — `evaluate.py` cannot do it, since it computes
  metrics against ground truth and writes them beside the predictions.
- **The affinity head has no graph-level vector**, same as SS-GNN. `ProLig` computes
  `linear_ligand_affinity(lig) * linear_pocket_affinity(poc)` over the whole atom×residue
  grid and sums per-pair scalars. Pooling for an embedding has to be introduced by us.
- Environment migrated from conda/Py3.6 to uv/Py3.13, torch 2.5.1. Security fixes already
  applied upstream in this fork: `os.system()` removed, `torch.load` uses `weights_only=True`.

## Where the reproduction stands

The paper's own figures were not the target. Our reference numbers are the 2026-04-13 run,
because that is the pipeline we can document and repeat. Known deviations from the authors:
they trained on a pre-cleaned PDBbind with decoy non-binders and their own splits, and the
multi-objective training with decoys was only partially reproduced.

## Build & run

```bash
podman build --format=docker -t planet:latest .
gnnb run --variant planet.modern --capability predict --dataset <complexes> --gpu
```

## Embedding

Done: `embed_complexes.py` replays the forward to `prolig_attention`, pools `fatoms` and
`fresidues` per complex and concatenates them — 2 × 300 = 600 dims. **Probe R 0.682**, retaining
89% of the model's own Pearson R, against the size baseline's 0.486.

Recorded as `defined`, not `native`: the pooling is ours. That distinction turned out to matter
more than expected — every model where we chose the aggregation sits eight to ten points below
the ones where the authors' own formula was available. Worth a second look for a pooling point
in their code before treating this number as the ceiling.

## Next

- Scaffold or similarity-aware split, to separate what the model knows from what it memorised.
  266 of the 285 core-set complexes sit inside PDBbind refined, and PLANET's training list is
  not published.
