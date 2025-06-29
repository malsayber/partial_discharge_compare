"""Utilities to map feature names to extractor callables."""
from __future__ import annotations

from typing import Callable, Dict

import config
from . import extractors


_EXTRACTOR_MAP = {
    **{name: getattr(extractors, name) for name in dir(extractors) if name.startswith("compute_")},
    "line_length": extractors.compute_line_length,
    "zero_crossings": extractors.compute_zero_crossings,
    "kurtosis": extractors.compute_kurtosis,
    "rms": extractors.compute_rms,
    "samp_entropy": extractors.compute_samp_entropy,
}


def load_feature_catalog() -> Dict[str, Callable]:
    """Load enabled features from ``config.yaml``.

    Returns
    -------
    Dict[str, Callable]
        Mapping of feature name to extractor function.
    """
    feats_cfg = config.CONFIG.features
    catalog: Dict[str, Callable] = {}

    def _add_feats(names: list[str]) -> None:
        for n in names:
            func = _EXTRACTOR_MAP.get(f"compute_{n}", _EXTRACTOR_MAP.get(n))
            if func:
                catalog[n] = func

    if feats_cfg.enable_all or feats_cfg.custom.enabled:
        _add_feats(feats_cfg.custom.selected_features)
    if feats_cfg.enable_all or feats_cfg.librosa.enabled:
        _add_feats(feats_cfg.librosa.selected_features)
    if feats_cfg.enable_all or feats_cfg.mne_features.enabled:
        _add_feats(feats_cfg.mne_features.selected_features)

    return catalog

