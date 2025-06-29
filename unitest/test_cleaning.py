"""Tests for signal cleaning functions."""

import numpy as np

from partial_discharge_compare.preprocess import cleaning
from .fixtures.synthetic_pd import generate_synthetic_partial_discharge


def test_bandpass_filter_shape() -> None:
    df = generate_synthetic_partial_discharge(num_good=1, num_fault=0, length=100)
    signal = df.iloc[0, :-1].to_numpy(float)
    filtered = cleaning.bandpass_filter(signal, 1.0, 30.0, fs=100.0)
    assert filtered.shape == signal.shape
    assert np.isfinite(filtered).all()
