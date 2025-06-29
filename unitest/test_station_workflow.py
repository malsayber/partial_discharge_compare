"""Tests for station-based preprocessing workflow."""

from pathlib import Path
import numpy as np

from preprocess import discovery, io
import types
import sys
import config


def test_discover_npy_files(tmp_path: Path) -> None:
    station1 = tmp_path / "station_1"
    station1.mkdir()
    np.save(station1 / "a.npy", np.array([1]))
    station2 = tmp_path / "station_2"
    station2.mkdir()
    np.save(station2 / "b.npy", np.array([2, 3]))

    records = sorted(discovery.discover_npy_files(tmp_path), key=lambda r: r.file_path)

    assert len(records) == 2
    assert records[0].station_id == "station_1"
    assert Path(records[0].file_path).name == "a.npy"
    assert records[1].station_id == "station_2"


def test_load_pd_npy(tmp_path: Path) -> None:
    arr = np.array([0.0, 1.0, 2.0])
    p = tmp_path / "sig.npy"
    np.save(p, arr)

    loaded = io.load_pd_npy(p)
    assert np.array_equal(loaded, arr)


def test_process_npy_file(tmp_path: Path, monkeypatch) -> None:
    data = np.arange(5, dtype=float)
    station = tmp_path / "station_123"
    station.mkdir()
    sig = station / "sig.npy"
    np.save(sig, data)
    record = discovery.FileRecord(station_id="station_123", file_path=str(sig))

    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(
        config.CONFIG.preprocessing_options,
        "window_length_ms",
        1000,
    )
    monkeypatch.setattr(
        config.CONFIG.preprocessing_options,
        "bandpass_hz",
        [0.1, 0.4],
    )

    fake = types.ModuleType("tsaug")
    fake.TimeWarp = lambda *a, **k: type("D", (), {"augment": lambda self, x: x})()
    fake.AddNoise = lambda *a, **k: type("D", (), {"augment": lambda self, x: x})()
    sys.modules.setdefault("tsaug", fake)
    from preprocess import run_preprocess

    monkeypatch.setattr(
        run_preprocess.cleaning,
        "bandpass_filter",
        lambda x, low, high, fs: x,
    )

    run_preprocess.process_npy_file(
        record,
        force=True,
        adv_denoise=False,
        augment=False,
    )

    out_dir = config.PROCESSED_DIR / record.station_id
    assert any(out_dir.glob("sig_*.npy"))
