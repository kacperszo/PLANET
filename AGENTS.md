# PLANET working rules

Read `CLAUDE.md` for the model and its preprocessing/paper-reproduction limits.
Use the benchmark's existing PodmanRunner: offline, read-only source/inputs,
separate outputs. Do not change active experiments or shared caches.

- Protein inputs are residue features and C-alpha coordinates; ligand inputs
  are a 2D graph. Preserve the architecture and the three author objectives.
- Supervised CSV labels, the PDBbind index and cached HDF5 pK must agree.
  Do not silently replace targets or rewrite shared cached labels.
- Zero is the model's missing-label sentinel. The supervised contract rejects it,
  as well as missing/non-finite labels and missing complexes. Preprocessing
  failures must be reflected explicitly in the split.
- Unlabelled inference still permits pK=0; the stricter contract is for training.
- Training shuffling starts from sorted records and uses `seed + epoch`.
  The underlying dataset's fixed-seed default remains for legacy callers.
- Validation must fail on an exception, non-finite objective or empty result.
  Such a run must not select a zero-loss "best" checkpoint.
- The 500-step affinity warm-up and author initialisation are unchanged.
  Missing decoys still mean a restricted training protocol, not faithful retraining.
- Tests in `test_training_contract.py` cover shuffle reproducibility, labels,
  cache refusal, and failed/empty validation.

Make repairs in a separate checkout; integrate only after tests.
