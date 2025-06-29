"""Tests for signal cleaning functions."""

import numpy as np

from preprocess import cleaning
from unitest.fixtures.synthetic_pd import generate_synthetic_partial_discharge


def test_bandpass_filter_shape() -> None:
    df = generate_synthetic_partial_discharge(num_good=1, num_fault=0, length=100)
    signal = df.iloc[0, :-1].to_numpy(float)
    filtered = cleaning.bandpass_filter(signal, 1.0, 30.0, fs=100.0)
    assert filtered.shape == signal.shape
    assert np.isfinite(filtered).all()


def test_zscore_normalize() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    norm = cleaning.zscore_normalize(arr)
    assert np.isclose(np.mean(norm), 0.0)
    assert np.isclose(np.std(norm), 1.0)


def test_advanced_denoise_identity() -> None:
    arr = np.sin(np.linspace(0, 2 * np.pi, 100))
    den = cleaning.advanced_denoise(arr, method="ewt")
    assert den.shape == arr.shape
