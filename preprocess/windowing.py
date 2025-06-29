from __future__ import annotations

"""Segmentation utilities for PD signals."""

from pathlib import Path
from typing import List
import json
import numpy as np


def segment_signal(x: np.ndarray, window_ms: float, fs: float) -> List[np.ndarray]:
    """Split ``x`` into non-overlapping windows."""
    win_len = int(fs * window_ms / 1000)
    n = len(x) // win_len
    return [x[i * win_len : (i + 1) * win_len] for i in range(n)]


def load_window_labels(label_file: str | Path | None, n_windows: int) -> List[int | None]:
    """Load labels for each window from ``label_file``."""
    if label_file is None or not Path(label_file).exists():
        return [None] * n_windows
    with open(label_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    label = data.get("label")
    return [label] * n_windows
