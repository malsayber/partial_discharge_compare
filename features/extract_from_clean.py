"""Feature extraction from cleaned npy files.

This module scans the project `root_dir` for folders named ``data_clean`` under
individual stations. Each subdirectory represents a cleaning strategy (e.g.
``standard_denoising_normalisation`` or ``advanced_denoising/VMD``). Every ``.npy``
file contained within is loaded and processed by the feature catalog defined in
:mod:`features.catalog`.

For each cleaned file we save multiple Parquet files under ``2_feature_engineering``
inside the ``outputs/features`` directory. The output hierarchy mirrors the cleaning
strategy and groups features by theme, for example::

    <root_dir>/outputs/features/<station>/classic_stats/advanced_denoising/VMD/<file>.parquet

This design allows different preprocessing experiments to coexist while keeping
features organised by theme and method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict
import inspect

import pandas as pd
import numpy as np
from tqdm import tqdm

import config
from .catalog import load_feature_catalog

# Map feature names to a thematic folder.
_THEME_MAP: dict[str, list[str]] = {
    "classic_stats": [
        "time_skewness",
        "time_kurtosis",
        "time_rms",
        "time_variance",
        "peak_to_peak",
        "zero_cross_rate",
        "line_length",
        "kurtosis",
        "rms",
        "zero_crossings",
    ],
    "time_frequency": [
        "spectral_centroid",
        "spectral_bandwidth",
        "dominant_frequency",
        "spectral_entropy",
        "mfcc",
    ],
    "wavelet_cwt": [
        "wavelet_energy",
        "wavelet_entropy",
        "wavelet_symlets_energy",
    ],
    "entropy_fractal": ["multiscale_entropy"],
}


def _call_extractor(func: Callable, window: np.ndarray, fs: float) -> float:
    """Call ``func`` with ``window`` and ``fs`` when required."""
    sig = inspect.signature(func)
    kwargs = {}
    if "fs" in sig.parameters:
        kwargs["fs"] = fs
    return float(func(window, **kwargs))


def _process_file(npy_path: Path, catalog: Dict[str, Callable], fs: float) -> None:
    """Compute features for one ``.npy`` file and save to Parquet.

    Parameters
    ----------
    npy_path:
        Path to a cleaned window stored under ``data_clean``.
    catalog:
        Mapping of feature names to extractor callables.
    fs:
        Sampling rate of the signal in Hz.
    """
    window = np.load(npy_path)
    station_dir = npy_path.parents[2]
    method_rel = npy_path.parent.relative_to(station_dir / "data_clean")
    for theme, feature_names in _THEME_MAP.items():
        rows = {}
        for name in feature_names:
            func = catalog.get(name)
            if not func:
                continue
            try:
                rows[name] = _call_extractor(func, window, fs)
            except Exception:  # pragma: no cover - extractor failure
                rows[name] = np.nan
        if rows:
            out_root = Path(config.CONFIG.project.features_dir)
            out_dir = out_root / station_dir.name / theme / method_rel
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{npy_path.stem}.parquet"
            pd.DataFrame([rows]).to_parquet(out_path, index=False)


def run(fs: float = 1.0) -> None:
    """Extract features from all cleaned windows.

    The function recursively scans ``CONFIG.project.root_dir`` for stations that
    contain a ``data_clean`` folder. For every ``.npy`` file found, the feature
    catalog is applied and results are written as Parquet files under
    ``outputs/features/<station>`` grouped by theme and cleaning method.

    Parameters
    ----------
    fs:
        Sampling rate used for feature extractors requiring it.
    """
    catalog = load_feature_catalog()
    root = Path(config.CONFIG.project.root_dir)
    for station in root.iterdir():
        data_clean = station / "data_clean"
        if not data_clean.is_dir():
            continue
        npy_files = list(data_clean.rglob("*.npy"))
        for npy_path in tqdm(npy_files, desc=station.name):
            _process_file(npy_path, catalog, fs)


if __name__ == "__main__":
    run()
