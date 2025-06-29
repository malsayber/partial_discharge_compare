from __future__ import annotations

"""Signal cleaning utilities used in the preprocessing stage.

This module implements simple cleaning helpers such as band-pass filtering,
advanced denoising via VMD or wavelet thresholding and amplitude normalisation.
Each helper returns the cleaned signal as a ``numpy.ndarray``.
"""

from pathlib import Path
from typing import List

import numpy as np
from scipy.signal import butter, filtfilt

import config


def save_cleaned_signal(x: np.ndarray, clean_type: str, dataset: str, window_id: str) -> Path:
    """Save a cleaned signal to the processed data directory.

    Parameters
    ----------
    x:
        Array containing the cleaned signal.
    clean_type:
        Name of the cleaning method (e.g. ``bandpass`` or ``denoise``).
    dataset:
        Dataset name used to create the subdirectory.
    window_id:
        Identifier for the sample/window.

    Returns
    -------
    Path
        Location of the saved file.
    """
    out_dir = config.PROCESSED_DIR / clean_type / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{window_id}.npy"
    np.save(out_path, x)
    return out_path


def bandpass_filter(x: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    """Apply a fourth-order Butterworth band-pass filter.

    Parameters
    ----------
    x:
        Input signal.
    low:
        Low cut frequency in Hz.
    high:
        High cut frequency in Hz.
    fs:
        Sampling rate in Hz of ``x``.

    Returns
    -------
    np.ndarray
        Filtered signal of the same shape as ``x``.
    """
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def vmd_denoise(x: np.ndarray, **kwargs) -> np.ndarray:
    """Denoise signal using Variational Mode Decomposition (VMD).

    Parameters
    ----------
    x:
        Signal to denoise.
    **kwargs:
        Additional keyword arguments passed to :func:`vmdpy.VMD`.

    Returns
    -------
    np.ndarray
        The reconstructed signal from the decomposed modes.
    """
    from vmdpy import VMD

    u, _, _ = VMD(x, **kwargs)
    return np.sum(u, axis=0)


def ewt_denoise(x: np.ndarray, wavelet: str = "db4", level: int = 1) -> np.ndarray:
    """Denoise a signal using wavelet thresholding as a proxy for EWT.

    Parameters
    ----------
    x:
        Input signal.
    wavelet:
        Wavelet family used for decomposition.
    level:
        Decomposition level.

    Returns
    -------
    np.ndarray
        Denoised signal reconstructed from thresholded coefficients.
    """
    import pywt

    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeffs = [pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs]
    rec = pywt.waverec(coeffs, wavelet)
    return rec[: len(x)]


def advanced_denoise(x: np.ndarray, method: str = "vmd", **kwargs) -> np.ndarray:
    """Apply advanced denoising using either VMD or wavelets.

    Parameters
    ----------
    x:
        Input signal to denoise.
    method:
        ``"vmd"`` to apply Variational Mode Decomposition or ``"ewt"`` to use
        a wavelet-based approach.
    **kwargs:
        Extra parameters forwarded to the underlying denoising function.

    Returns
    -------
    np.ndarray
        The denoised signal.
    """
    if method == "vmd":
        return vmd_denoise(x, **kwargs)
    return ewt_denoise(x, **kwargs)


def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Return a z-score normalised copy of ``x``."""
    mean = np.mean(x)
    std = np.std(x) if np.std(x) > 0 else 1.0
    return (x - mean) / std
