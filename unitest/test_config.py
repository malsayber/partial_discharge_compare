"""Tests for configuration loading."""

from pathlib import Path

import config


def test_load_config() -> None:
    cfg = config.load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    assert cfg.project.root_dir == "partial_discharge_project"
    assert isinstance(config.ROOT_DIR, Path)
