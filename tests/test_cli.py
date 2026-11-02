import pytest

from insar_unwrap.cli import main

from .conftest import EXAMPLE_CONFIG


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "validate-config" in out and "doctor" in out


def test_validate_config_success(capsys):
    assert main(["validate-config", "--config", str(EXAMPLE_CONFIG)]) == 0
    assert capsys.readouterr().out.startswith("OK:")


def test_validate_config_failure_reports_field(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "experiment: {name: x}\n"
        "input: {kind: synthetic}\n"
        "problem: {sizes: [32], label_min: 2, label_max: 1}\n",
        encoding="utf-8",
    )
    assert main(["validate-config", "--config", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "label_min" in err


def test_doctor_runs_locally(capsys):
    rc = main(["doctor", "--snaphu-executable", "definitely-not-a-real-binary"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "SQA sampler" in out
    assert "definitely-not-a-real-binary" in out and "MISSING" in out
