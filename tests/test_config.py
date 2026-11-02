import pytest

from insar_unwrap.config import ConfigError, ExperimentConfig, load_config

from .conftest import EXAMPLE_CONFIG


def test_example_config_validates_without_data_present():
    cfg = load_config(EXAMPLE_CONFIG)
    assert cfg.input.kind == "nisar_gunw"
    assert cfg.problem.sizes == [32, 64, 128, 256]
    assert cfg.problem.num_labels == 4
    assert cfg.problem.bits_per_pixel == 2
    # The placeholder file must not exist; validation must not depend on it.
    assert cfg.input.path is not None and not cfg.input.path.exists()


def test_one_hot_bits_per_pixel(example_dict):
    example_dict["problem"]["encoding"] = "one_hot"
    cfg = ExperimentConfig.model_validate(example_dict)
    assert cfg.problem.bits_per_pixel == cfg.problem.num_labels


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda d: d["problem"].update(label_min=1, label_max=1), "strictly less"),
        (lambda d: d["problem"].update(sizes=[64, 32]), "ascending"),
        (lambda d: d["problem"].update(sizes=[32, 32]), "duplicates"),
        (lambda d: d["problem"].update(sizes=[0, 32]), "greater than or equal to 1"),
        (lambda d: d["input"]["mask"].update(coherence_threshold=1.5), "less than or equal to 1"),
        (lambda d: d["solvers"]["sa"].update(beta_range=[5.0, 1.0]), "beta_min < beta_max"),
        (lambda d: d["solvers"]["sqa"].update(Hp_field=[1.0]), "given together"),
        (
            lambda d: d["solvers"]["sqa"].update(Hp_field=[1.0], Hd_field=[1.0]),
            "not both",
        ),
        (
            lambda d: (
                d["solvers"]["sqa"].pop("schedule"),
                d["solvers"]["sqa"].update(Hp_field=[1.0, 2.0], Hd_field=[0.0, 0.0]),
            ),
            "classical annealing",
        ),
        (lambda d: d["solvers"]["sqa"]["schedule"].update(gamma_end=5.0), "gamma_end must be"),
        # Gamma and the linear/geometric schedules are deliberately not accepted.
        (lambda d: d["solvers"]["sqa"].update(Gamma=1.0), "Extra inputs"),
        (lambda d: d["solvers"]["sqa"].update(beta_schedule_type="geometric"), "Extra inputs"),
        (lambda d: d["experiment"].update(seed=2**31), "less than or equal to 2147483647"),
        (lambda d: d["solvers"]["sqa"].update(bogus_argument=3), "Extra inputs"),
        (lambda d: d["input"].pop("layers"), "input.layers is required"),
        (lambda d: d["problem"]["reference"].update(mode="pixel"), "pixel is required"),
        (
            lambda d: d["solvers"].update(
                sa={"enabled": False}, sqa={"enabled": False}, snaphu={"enabled": False}
            ),
            "at least one solver",
        ),
    ],
)
def test_invalid_configs_are_rejected(example_dict, write_yaml, mutate, fragment):
    mutate(example_dict)
    with pytest.raises(ConfigError) as excinfo:
        load_config(write_yaml(example_dict))
    assert fragment in str(excinfo.value)


def test_synthetic_input_does_not_need_path_or_layers(example_dict):
    example_dict["input"] = {"kind": "synthetic"}
    example_dict["problem"]["edge_weighting"] = "uniform"
    cfg = ExperimentConfig.model_validate(example_dict)
    assert cfg.input.path is None


def test_non_mapping_yaml_is_rejected(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(p)


def test_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "nope.yaml")


def test_sqa_explicit_fields_replace_generated_schedule(example_dict):
    sqa = example_dict["solvers"]["sqa"]
    sqa.pop("schedule")
    sqa.update(Hp_field=[5.0, 5.0, 5.0], Hd_field=[10.0, 5.0, 0.0], num_sweeps_per_beta=2)
    cfg = ExperimentConfig.model_validate(example_dict)
    assert cfg.solvers.sqa.total_sweeps == 6


def test_sqa_generated_schedule_sweeps(example_dict):
    cfg = ExperimentConfig.model_validate(example_dict)
    assert cfg.solvers.sqa.total_sweeps == cfg.solvers.sqa.schedule.num_steps
