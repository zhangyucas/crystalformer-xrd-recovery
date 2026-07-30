import argparse
import csv
import json
import pickle

import numpy as np

from issue68_xrd_recovery.scripts.xrd_prior_sample import (
    _load_params,
    _sample_row,
    extract_params,
)
from issue68_xrd_recovery.scripts.xrd_recovery import convert_samples


def _p1_row():
    target = np.zeros(119, dtype=int)
    target[14] = 1
    return _sample_row(
        0,
        0,
        target,
        (
            1,
            np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.asarray([14, 0]),
            np.asarray([1, 0]),
            np.asarray([1, 0]),
            np.asarray([3.5, 3.5, 3.5, 90.0, 90.0, 90.0]),
        ),
    )


def test_sample_row_records_exact_formula_and_geometry():
    row = _p1_row()
    assert row["formula_match"] is True
    assert row["geometry_precheck"] is True
    assert json.loads(row["A"])[:2] == [14, 0]


def test_sample_row_uses_crystalformer_reduced_formula_matching():
    target = np.zeros(119, dtype=int)
    target[14] = 1
    row = _sample_row(
        0,
        0,
        target,
        (
            1,
            np.asarray([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            np.asarray([14, 14]),
            np.asarray([1, 1]),
            np.asarray([8, 8]),
            np.asarray([5.0, 5.0, 5.0, 90.0, 90.0, 90.0]),
        ),
    )
    assert row["num_atoms"] == 16
    assert row["formula_match"] is True


def test_extract_params_strips_optimizer_state(tmp_path):
    source = tmp_path / "epoch_000001.pkl"
    output = tmp_path / "params.pkl"
    with source.open("wb") as handle:
        pickle.dump({"params": {"w": np.ones(2)}, "opt_state": {"large": [1, 2]}}, handle)
    extract_params(argparse.Namespace(checkpoint=str(source), output=str(output), force=False))
    with output.open("rb") as handle:
        payload = pickle.load(handle)
    assert "opt_state" not in payload
    np.testing.assert_array_equal(_load_params(output)["w"], np.ones(2))


def test_convert_streamed_sample_to_cif(tmp_path):
    input_path = tmp_path / "samples.csv"
    row = _p1_row()
    with input_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    output_dir = tmp_path / "cifs"
    convert_samples(argparse.Namespace(
        input=str(input_path),
        output_dir=str(output_dir),
        max_candidates=10,
        require_formula_match=True,
    ))
    assert (output_dir / "sample_0000.cif").is_file()
    conversion = list(csv.DictReader((output_dir / "conversion.csv").open()))
    assert conversion[0]["formula"] == "Si"
    assert conversion[0]["error"] == ""
