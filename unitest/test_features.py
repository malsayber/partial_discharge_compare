"""Tests for feature extractors."""

import numpy as np

from unitest.fixtures.synthetic_pd import generate_synthetic_partial_discharge
from features import extractors


def _get_window() -> np.ndarray:
    df = generate_synthetic_partial_discharge(num_good=1, num_fault=0, length=64)
    return df.iloc[0, :-1].to_numpy(float)


def test_time_skewness_small() -> None:
    w = _get_window()
    val = extractors.compute_time_skewness(w)
    assert abs(val) < 1.0


def test_spectral_centroid_range() -> None:
    w = _get_window()
    val = extractors.compute_spectral_centroid(w, fs=64.0)
    assert 0.0 <= val <= 32.0


def test_mne_line_length_positive() -> None:
    w = _get_window()
    val = extractors.compute_line_length(w, fs=64.0)
    assert val > 0.0


def test_librosa_mfcc_finite() -> None:
    w = _get_window()
    val = extractors.compute_mfcc(w, fs=64.0)
    assert np.isfinite(val)
