# Training-contract repairs — 2026-10-20

- Sort discovered HDF5 records and pass `seed + epoch` from the harness trainer,
  so shuffling is reproducible and is not reset to the same order every epoch.
- Validate supervised labels against both index and HDF5, with absolute
  tolerance 1e-5; reject missing, duplicate, zero-sentinel or non-finite labels,
  missing prepared complexes and stale cached target values.
- Validation propagates failures, rejects non-finite loss and empty results,
  and restores model mode even on errors; it cannot report zero for failed batches.
- Reject invalid training settings and non-finite objectives/gradient norms.

No network architecture, inference featurisation, pooling, loss weighting or
warm-up change. No shared cache migration. Large splits with preprocessing
failures now require an explicitly cleaned split instead of silently shrinking.
The original decoy-data and full paper-reproduction gaps remain.
