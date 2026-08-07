import csv
import json

from issue68_xrd_recovery.scripts.run_ppo_checkpoint_evaluation import (
    _complete_sampling_csv,
    _valid_json,
)


def _write_sampling_csv(path, indices):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_index", "value"))
        writer.writeheader()
        for index in indices:
            writer.writerow({"sample_index": index, "value": index})


def test_complete_sampling_csv_requires_exact_contiguous_pool(tmp_path):
    complete = tmp_path / "complete.csv"
    truncated = tmp_path / "truncated.csv"
    discontinuous = tmp_path / "discontinuous.csv"
    _write_sampling_csv(complete, range(3))
    _write_sampling_csv(truncated, range(2))
    _write_sampling_csv(discontinuous, (0, 2, 1))

    assert _complete_sampling_csv(complete, 3)
    assert not _complete_sampling_csv(truncated, 3)
    assert not _complete_sampling_csv(discontinuous, 3)
    assert not _complete_sampling_csv(tmp_path / "missing.csv", 3)


def test_valid_json_rejects_interrupted_output(tmp_path):
    complete = tmp_path / "complete.json"
    truncated = tmp_path / "truncated.json"
    complete.write_text(json.dumps({"status": "complete"}))
    truncated.write_text('{"status":')

    assert _valid_json(complete)
    assert not _valid_json(truncated)
