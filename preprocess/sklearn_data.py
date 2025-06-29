from __future__ import annotations

"""Utility to materialize scikit-learn toy datasets as raw CSV files."""

from pathlib import Path
import json

import numpy as np
from sklearn import datasets

import config

_SUPPORTED = {
    "iris": datasets.load_iris,
    "wine": datasets.load_wine,
    "breast_cancer": datasets.load_breast_cancer,
}


def ensure_dataset(dataset: str) -> None:
    """Generate CSV files under ``data/raw/<dataset>`` if not already present."""
    if dataset not in _SUPPORTED:
        return

    base = config.RAW_DIR / dataset
    if base.exists():
        return

    base.mkdir(parents=True)
    data = _SUPPORTED[dataset]()
    X = np.asarray(data.data, dtype=float)
    y = np.asarray(data.target, dtype=int)

    for idx, (row, label) in enumerate(zip(X, y)):
        session_dir = base / f"sample_{idx:04d}"
        session_dir.mkdir()
        np.savetxt(session_dir / "HFCT.csv", row, delimiter=",")
        with open(session_dir / "labels.json", "w", encoding="utf-8") as fh:
            json.dump({"label": int(label)}, fh)

