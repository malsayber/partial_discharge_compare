"""Tests for I/O helpers."""

from pathlib import Path
import numpy as np
import h5py

from partial_discharge_compare.preprocess import io


def test_load_pd_csv(tmp_path: Path) -> None:
    p = tmp_path / "sig.csv"
    p.write_text("1\n2\n3\n")
    data = io.load_pd_csv(p)
    assert np.array_equal(data, np.array([1.0, 2.0, 3.0]))


def test_load_pd_hdf5(tmp_path: Path) -> None:
    p = tmp_path / "sig.h5"
    with h5py.File(p, "w") as h5:
        h5.create_dataset("data", data=[0, 1, 2])
    data = io.load_pd_hdf5(p)
    assert np.array_equal(data, np.array([0.0, 1.0, 2.0]))
