from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "experiment.example.yaml"


@pytest.fixture
def example_dict() -> dict:
    with EXAMPLE_CONFIG.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def write_yaml(tmp_path: Path):
    def _write(data: dict, name: str = "cfg.yaml") -> Path:
        p = tmp_path / name
        p.write_text(yaml.safe_dump(data), encoding="utf-8")
        return p

    return _write
