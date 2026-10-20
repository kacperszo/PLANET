"""Reject silent changes of supervised targets across CSV, index and HDF5."""
import csv
import math
from pathlib import Path

LABEL_ATOL = 1e-5


def read_labels(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not {"complex_id", "y_true"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: needs complex_id,y_true")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty split")
    labels = {}
    for row in rows:
        cid = row["complex_id"].strip().lower()
        if not cid or Path(cid).name != cid or cid in (".", "..") or cid in labels:
            raise ValueError(f"{path}: invalid or duplicate complex ID {cid!r}")
        value = float(row["y_true"])
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite label for {cid}")
        if value == 0:
            raise ValueError(f"{path}: pK=0 for {cid} is PLANET's missing-label sentinel; "
                             "this supervised contract cannot represent it")
        labels[cid] = value
    return labels


def validate_pk(cid, expected, actual, source):
    if not math.isfinite(actual) or actual == 0:
        raise ValueError(f"{source}: missing/non-finite pK for {cid}")
    if not math.isclose(expected, actual, rel_tol=0, abs_tol=LABEL_ATOL):
        raise ValueError(f"{source}: label mismatch for {cid}: CSV {expected}, stored {actual}; "
                         "use a consistent split/cache, do not silently reuse different labels")


def validate_h5(data_dir, labels):
    import h5py

    for cid, expected in labels.items():
        path = Path(data_dir) / cid / f"{cid}_pocket.h5"
        if not path.is_file():
            raise ValueError(f"{path}: missing prepared complex; make exclusions explicit in the split")
        with h5py.File(path, "r") as f:
            if "pK" not in f.attrs:
                raise ValueError(f"{path}: missing pK attribute")
            validate_pk(cid, expected, float(f.attrs["pK"]), path)
