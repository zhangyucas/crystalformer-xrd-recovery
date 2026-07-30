import json

from issue68_xrd_recovery.scripts.xrd_study import (
    _group_rows,
    _summary_rows,
    make_commands,
)


def test_study_groups_tau_and_separates_match_rate(tmp_path):
    first = tmp_path / "tau0"
    second = tmp_path / "tau1"
    first.mkdir()
    second.mkdir()
    (first / "summary.json").write_text(json.dumps({
        "method": "ppo",
        "beta": 0.0,
        "best_score": 0.95,
        "structure_match": True,
    }))
    (second / "summary.json").write_text(json.dumps({
        "method": "ppo",
        "beta": 0.1,
        "best_score": 0.92,
        "structure_match": False,
    }))
    rows = _summary_rows(tmp_path, 0.9)
    grouped = _group_rows(rows, 0.9)
    assert len(grouped) == 2
    assert {row["tau"] for row in grouped} == {0.0, 0.1}
    assert sorted(row["structure_match_rate"] for row in grouped) == [0.0, 1.0]


def test_study_reads_xrd_ppo_log_without_loading_checkpoint(tmp_path):
    run = tmp_path / "ppo"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({
        "reward": "xrd",
        "beta": 0.25,
        "formula": "Si",
    }))
    (run / "data.txt").write_text(
        "epoch xrd_mean xrd_err xrd_max xrd_min attempt\n"
        "1 0.4 0.1 0.7 0.2 3\n"
        "2 0.5 0.1 0.8 0.3 3\n"
    )
    rows = _summary_rows(tmp_path, 0.75)
    assert len(rows) == 1
    assert rows[0]["method"] == "crystalformer_ppo"
    assert rows[0]["tau"] == 0.25
    assert rows[0]["best_similarity"] == 0.8


def test_make_commands_expands_seed_and_evaluation_steps(tmp_path):
    class Args:
        tau = "0,0.1"
        seeds = "3,5"
        formula = "Si"
        target = "target.csv"
        restore_path = "compact.pkl"
        output_root = str(tmp_path / "sweep")
        ground_truth = "truth.cif"
        evaluation_max_candidates = 12
        epochs = 1
        ppo_epochs = 1
        batchsize = 2
        sample_multiplier = 2
        max_sampling_attempts = 4
        h0_size = 64
        transformer_layers = 2
        num_heads = 4
        key_size = 16
        model_size = 64
        embed_size = 64
        sg_temperature = 0.8
        sg_epsilon = 0.1
        diversity_weight = 0.2

    make_commands(Args)
    script = (tmp_path / "sweep" / "run_tau_sweep.sh").read_text()
    assert script.count("crystalformer.cli.train_ppo") == 4
    assert script.count("issue68_xrd_recovery/scripts/xrd_recovery.py") == 4
    assert "--seed \\\n  5" in script
