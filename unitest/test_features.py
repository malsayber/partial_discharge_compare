from feature_extraction import time_skewness, time_kurtosis, wavelet_energy
import random
import unittest
import numpy as np

from feature_extraction.time_mean import time_mean
from feature_extraction.time_std import time_std
from feature_extraction.time_rms import time_rms
from feature_extraction.dominant_frequency import dominant_frequency
from feature_extraction.spectral_entropy import spectral_entropy
from unitest.fixtures.synthetic_pd import generate_synthetic_partial_discharge

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

class TestFeatures(unittest.TestCase):
    def setUp(self):
        df = generate_synthetic_partial_discharge(num_good=1, num_fault=0)
        self.signal = df.iloc[0, :-1].to_numpy()

    def test_time_mean(self):
        self.assertAlmostEqual(time_mean(self.signal), np.mean(self.signal))

    def test_time_std(self):
        self.assertAlmostEqual(time_std(self.signal), np.std(self.signal))

    def test_time_rms(self):
        expected = np.sqrt(np.mean(self.signal ** 2))
        self.assertAlmostEqual(time_rms(self.signal), expected)

    def test_dominant_frequency(self):
        freq = dominant_frequency(self.signal, fs=1.0)
        self.assertTrue(isinstance(freq, float))

    def test_spectral_entropy(self):
        ent = spectral_entropy(self.signal)
        self.assertTrue(isinstance(ent, float))


if __name__ == "__main__":
    unittest.main()