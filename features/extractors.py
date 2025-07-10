"""Base feature extractor functions for PD windows."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import stats, signal
import pywt
import pandas as pd
import inspect

try:
    from mne_features.feature_extraction import extract_features
    HAVE_MNE = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_MNE = False

try:
    import librosa
    HAVE_LIBROSA = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_LIBROSA = False


# ---------------------------------------------------------------------------
# Custom time-domain statistics
# ---------------------------------------------------------------------------

def compute_time_skewness(window: np.ndarray) -> float:
    """Return the skewness of the signal."""
    return float(stats.skew(window))


def compute_time_kurtosis(window: np.ndarray) -> float:
    """Return the kurtosis of the signal."""
    return float(stats.kurtosis(window))


def compute_time_rms(window: np.ndarray) -> float:
    """Return the root-mean-square of the signal."""
    return float(np.sqrt(np.mean(np.square(window))))


def compute_time_variance(window: np.ndarray) -> float:
    """Return the variance of the signal."""
    return float(np.var(window))


def compute_peak_to_peak(window: np.ndarray) -> float:
    """Return the peak-to-peak amplitude."""
    return float(np.ptp(window))


def compute_zero_cross_rate(window: np.ndarray) -> float:
    """Return the rate of zero crossings."""
    signs = np.sign(window)
    return float(np.mean(signs[:-1] != signs[1:]))


# ---------------------------------------------------------------------------
# Frequency-domain statistics
# ---------------------------------------------------------------------------

def compute_spectral_centroid(window: np.ndarray, fs: float) -> float:
    """Return the spectral centroid in Hz."""
    if not HAVE_LIBROSA:
        freqs, psd = signal.welch(window, fs=fs)
        return float(np.sum(freqs * psd) / np.sum(psd))
    centroid = librosa.feature.spectral_centroid(y=window, sr=int(fs))
    return float(np.mean(centroid))


def compute_spectral_bandwidth(window: np.ndarray, fs: float) -> float:
    """Return the spectral bandwidth in Hz."""
    if not HAVE_LIBROSA:
        freqs, psd = signal.welch(window, fs=fs)
        centroid = np.sum(freqs * psd) / np.sum(psd)
        return float(np.sqrt(np.sum(psd * (freqs - centroid) ** 2) / np.sum(psd)))
    bw = librosa.feature.spectral_bandwidth(y=window, sr=int(fs))
    return float(np.mean(bw))


def compute_spectral_entropy(window: np.ndarray, fs: float) -> float:
    """Return the Shannon entropy of the power spectral density."""
    freqs, psd = signal.welch(window, fs=fs)
    psd_norm = psd / np.sum(psd)
    return float(-np.sum(psd_norm * np.log(psd_norm + 1e-10)))


def compute_dominant_frequency(window: np.ndarray, fs: float) -> float:
    """Return the dominant frequency component in Hz."""
    freqs, psd = signal.welch(window, fs=fs)
    return float(freqs[np.argmax(psd)])


# ---------------------------------------------------------------------------
# Wavelet-based statistics
# ---------------------------------------------------------------------------

def compute_wavelet_energy(window: np.ndarray, wavelet: str = "db4", level: int = 3) -> float:
    """Return the total wavelet energy across decomposition levels."""
    coeffs = pywt.wavedec(window, wavelet, level=level)
    energy = np.sum([np.sum(c ** 2) for c in coeffs])
    return float(energy / len(window))


def compute_wavelet_entropy(window: np.ndarray, wavelet: str = "db4", level: int = 3) -> float:
    """Return the entropy of wavelet energies."""
    coeffs = pywt.wavedec(window, wavelet, level=level)
    energies = np.array([np.sum(c ** 2) for c in coeffs])
    energies /= np.sum(energies)
    return float(-np.sum(energies * np.log(energies + 1e-12)))


def compute_wavelet_symlets_energy(window: np.ndarray, level: int = 3) -> float:
    """Wavelet energy using Symlet basis."""
    return compute_wavelet_energy(window, wavelet="sym5", level=level)


def compute_multiscale_entropy(window: np.ndarray, scales: Iterable[int] | None = None) -> float:
    """Approximate multiscale entropy using coarse-graining."""
    if scales is None:
        scales = range(1, 4)
    entropies = []
    for s in scales:
        if s <= 1:
            coarse = window
        else:
            coarse = window[: len(window) // s * s].reshape(-1, s).mean(axis=1)
        entropies.append(compute_spectral_entropy(coarse, fs=1.0))
    return float(np.mean(entropies))


# ---------------------------------------------------------------------------
# Librosa features
# ---------------------------------------------------------------------------

def compute_mfcc(window: np.ndarray, fs: float, n_mfcc: int = 13) -> float:
    """Return the mean MFCC across coefficients."""
    if not HAVE_LIBROSA:
        raise ImportError("librosa is required for MFCC computation")
    mfcc = librosa.feature.mfcc(y=window, sr=int(fs), n_mfcc=n_mfcc)
    return float(np.mean(mfcc))


# ---------------------------------------------------------------------------
# Wrappers for mne-features
# ---------------------------------------------------------------------------

def _compute_mne_feature(window: np.ndarray, fs: float, feature: str) -> float:
    """General helper to compute a feature via mne-features."""
    if not HAVE_MNE:
        raise ImportError("mne-features is required for this extractor")
    data = window[np.newaxis, np.newaxis, :]
    values = extract_features(data, fs, selected_funcs=[feature])
    return float(values[0, 0])


def compute_line_length(window: np.ndarray, fs: float) -> float:
    return _compute_mne_feature(window, fs, "line_length")


def compute_zero_crossings(window: np.ndarray, fs: float) -> float:
    return _compute_mne_feature(window, fs, "zero_crossings")


def compute_kurtosis(window: np.ndarray, fs: float) -> float:
    return _compute_mne_feature(window, fs, "kurtosis")


def compute_rms(window: np.ndarray, fs: float) -> float:
    return _compute_mne_feature(window, fs, "rms")


def compute_samp_entropy(window: np.ndarray, fs: float) -> float:
    return _compute_mne_feature(window, fs, "samp_entropy")


class FeatureExtractor:
    """Simple wrapper used in tutorial scripts to compute features."""

    def __init__(self, fs: float = 1.0) -> None:
        from .catalog import load_feature_catalog

        self.fs = fs
        self.catalog = load_feature_catalog()

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all enabled features for ``df``.

        Parameters
        ----------
        df:
            DataFrame containing a ``signal`` column.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame of extracted features.
        """

        arr = df.iloc[:, 0].to_numpy()
        rows = {}
        for name, func in self.catalog.items():
            sig = inspect.signature(func)
            kwargs = {}
            if "fs" in sig.parameters:
                kwargs["fs"] = self.fs
            rows[name] = func(arr, **kwargs)
        return pd.DataFrame([rows])

