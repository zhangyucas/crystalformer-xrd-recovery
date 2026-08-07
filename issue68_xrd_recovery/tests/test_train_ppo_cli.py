import argparse
import json

from crystalformer.cli.train_ppo import _resolve_xrd_args, _validate_safe_checkpoint
from issue68_xrd_recovery.scripts.run_improved_ppo_matrix import completed_run


def test_xrd_cli_values_prefer_explicit_values_and_read_metadata(tmp_path):
    target = tmp_path / "target.csv"
    target.write_text("two_theta,intensity\n20,1\n")
    (tmp_path / "target.csv.json").write_text(json.dumps({
        "wavelength": "Mo",
        "two_theta_min": 10,
        "two_theta_max": 80,
        "grid_step": 0.2,
        "profile": "pseudo-voigt",
        "fwhm": 0.3,
        "eta": 0.25,
        "target_is_peaks": True,
    }))
    args = argparse.Namespace(
        formula="Si",
        xrd_target=str(target),
        xrd_target_structure=None,
        xrd_wavelength=None,
        xrd_two_theta_min=5.0,
        xrd_two_theta_max=None,
        xrd_grid_step=None,
        xrd_profile=None,
        xrd_fwhm=None,
        xrd_eta=None,
        xrd_target_is_peaks=None,
    )
    _resolve_xrd_args(args, argparse.ArgumentParser())
    assert args.xrd_wavelength == "Mo"
    # An explicitly supplied numeric value wins over sidecar metadata.
    assert args.xrd_two_theta_min == 5.0
    assert args.xrd_two_theta_max == 80
    assert args.xrd_grid_step == 0.2
    assert args.xrd_target_is_peaks is True


def test_safe_checkpoint_directory_resolves_same_file_as_loader(tmp_path):
    older = tmp_path / "epoch_000002.pkl"
    latest = tmp_path / "epoch_000010.pkl"
    older.write_bytes(b"small")
    latest.write_bytes(b"small")
    args = argparse.Namespace(safe_cpu=True, restore_path=str(tmp_path))

    _validate_safe_checkpoint(args, argparse.ArgumentParser())

    assert args.restore_path == str(latest)


def test_improved_matrix_requires_new_ppo_diagnostics(tmp_path):
    run_dir = tmp_path / "run" / "config"
    run_dir.mkdir(parents=True)
    header = (
        "epoch reward_mean advantage_std ppo_objective clip_fraction ratio_max "
        "approx_kl_old grad_norm score_p50\n"
    )
    rows = "".join(f"{epoch} 0 1 0 0 1 0 1 0.5\n" for epoch in range(1, 6))
    (run_dir / "data.txt").write_text(header + rows)
    (run_dir / "epoch_010005.pkl").write_bytes(b"checkpoint")
    complete, reason, _ = completed_run(tmp_path / "run", epochs=5)
    assert complete
    assert reason == "complete"


def test_xrd_gamma_default_is_resolved_before_numeric_validation(tmp_path, monkeypatch):
    import pytest
    from crystalformer.cli import train_ppo

    monkeypatch.setattr(
        "sys.argv",
        ["train_ppo", "--reward", "xrd", "--formula", "Si"],
    )
    with pytest.raises(SystemExit) as exc:
        train_ppo.main()
    # The expected error is the missing XRD target, not a None comparison for gamma.
    assert exc.value.code == 2
