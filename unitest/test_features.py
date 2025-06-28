import unittest
import numpy as np

from feature_extraction.time_mean import time_mean
from feature_extraction.time_std import time_std
from feature_extraction.time_rms import time_rms
from feature_extraction.dominant_frequency import dominant_frequency
from feature_extraction.spectral_entropy import spectral_entropy
from unitest.fixtures.synthetic_pd import generate_synthetic_partial_discharge


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
