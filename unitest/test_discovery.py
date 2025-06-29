"""Tests for dataset discovery including sklearn fallback."""

from pathlib import Path

from preprocess import discovery


def test_discover_sklearn_dataset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(discovery.config, "RAW_DIR", tmp_path)
    sessions = discovery.discover_sessions("iris")
    assert sessions
    assert (tmp_path / "iris").exists()

