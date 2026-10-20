"""PLANET label, shuffle and validation regressions; run in its offline image."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from label_contract import read_labels, validate_h5
from planet.data import ProLigDataset
from prepare_h5 import process_one
from train_contract import run_eval

torch.set_num_threads(1)


class EvalModel(torch.nn.Module):
    def __init__(self, fail=False, nonfinite=False):
        super().__init__()
        self.fail = fail
        self.nonfinite = nonfinite

    def forward(self, *args):
        if self.fail:
            raise RuntimeError("bad validation batch")
        return None, None, torch.tensor([2., 4.])

    def compute_loss(self, *args):
        return torch.tensor(float("nan") if self.nonfinite else 1.), torch.tensor(2.), torch.tensor(3.)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def h5(self, cid, pk=5.):
        path = self.root / cid / f"{cid}_pocket.h5"
        path.parent.mkdir(exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["pK"] = pk
        return path

    def test_reproducible_epoch_shuffle_independent_of_directory_order(self):
        for i in range(24):
            self.h5(f"id{i:02d}")
        names = [f"id{i:02d}" for i in range(24)]
        def order(seed):
            return ProLigDataset(str(self.root), pdb_ids=set(names), seed=seed,
                                 batch_size=4, shuffle=True).batches
        with patch("planet.data.os.listdir", return_value=names):
            first, next_epoch = order(7), order(8)
        with patch("planet.data.os.listdir", return_value=names[::-1]):
            repeated = order(7)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_epoch)
        self.assertEqual(sorted(sum(first, [])), sorted(sum(next_epoch, [])))

    def test_label_contract_refuses_mismatch_missing_zero_and_nonfinite(self):
        path = self.h5("abc", 5.)
        before = path.read_bytes()
        validate_h5(self.root, {"abc": 5.})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_h5(self.root, {"abc": 6.})
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "missing prepared"):
            validate_h5(self.root, {"missing": 5.})
        for pk in (0., float("nan"), float("inf")):
            self.h5("abc", pk)
            with self.assertRaises(ValueError):
                validate_h5(self.root, {"abc": 5.})

    def test_csv_contract(self):
        path = self.root / "labels.csv"
        for rows in ([("A", 5.), ("a", 5.)], [("a", 0.)], [("a", "nan")], [], [("", 5.)]):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["complex_id", "y_true"])
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                read_labels(path)
        path.write_text("complex_id,y_true\nabc,5.2\n")
        self.assertEqual(read_labels(path), {"abc": 5.2})

    def test_cached_h5_mismatch_is_not_silently_reused_or_rewritten(self):
        path = self.h5("abc", 5.)
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            process_one(("abc", "/unused", str(self.root), 6., True))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(process_one(("abc", "/unused", str(self.root), 5., True)),
                         ("abc", "cached"))

    def test_validation_failures_are_not_zero_loss_and_restore_mode(self):
        targets = (None, None, torch.tensor([1., 2.]), torch.tensor([1., 1.]))
        batch = (None, None, targets)
        for model, batches, error in [
            (EvalModel(), [], ValueError),
            (EvalModel(fail=True), [batch], RuntimeError),
            (EvalModel(nonfinite=True), [batch], FloatingPointError),
        ]:
            model.train()
            with self.assertRaises(error):
                run_eval(model, batches, "cpu")
            self.assertTrue(model.training)
        model = EvalModel()
        loss, r = run_eval(model, [batch], "cpu")
        self.assertEqual(loss, 6.)
        self.assertAlmostEqual(r, 1.)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
