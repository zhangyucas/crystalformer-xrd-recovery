import argparse
import json

from crystalformer.cli.train_ppo import _resolve_xrd_args, _validate_safe_checkpoint


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
