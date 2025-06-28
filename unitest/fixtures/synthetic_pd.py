"""Synthetic partial discharge signal generator for testing."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_partial_discharge(
    num_good: int = 20,
    num_fault: int = 20,
    length: int = 40,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Create a simple dataset of synthetic partial discharge signals.

    Parameters
    ----------
    num_good : int
        Number of good (non-discharge) samples.
    num_fault : int
        Number of partial discharge samples.
    length : int
        Number of data points in each signal.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame where each row is a signal and ``target`` column holds labels
        (0 for good, 1 for partial discharge).
    """
    rng = np.random.default_rng(seed)

    good = rng.normal(0.0, 0.05, size=(num_good, length))
    fault = rng.normal(0.0, 0.05, size=(num_fault, length))
    pulse_positions = rng.integers(5, length - 5, size=(num_fault, 3))
    for i, positions in enumerate(pulse_positions):
        for pos in positions:
            fault[i, pos] += rng.uniform(0.5, 1.0)

    data = np.vstack([good, fault])
    labels = np.array([0] * num_good + [1] * num_fault)
    columns = [f"s{i}" for i in range(length)]
    df = pd.DataFrame(data, columns=columns)
    df["target"] = labels
    return df
