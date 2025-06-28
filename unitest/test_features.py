from feature_extraction import time_skewness, time_kurtosis, wavelet_energy
import random


def test_time_skewness_zero_for_symmetric_data():
    data = [-1, 0, 1]
    assert abs(time_skewness(data)) < 1e-9


def test_time_kurtosis_known_value():
    data = [-1, 0, 1]
    expected = 1.5
    assert abs(time_kurtosis(data) - expected) < 1e-9


def test_wavelet_energy_conservation():
    random.seed(0)
    data = [random.random() for _ in range(16)]
    energies = wavelet_energy(data, level=2)
    total = sum(energies.values())
    expected = sum(x * x for x in data)
    assert abs(total - expected) < 1e-6
