"""
Synthetic Partial Discharge Signal Generator and Loader.

This script provides tools for generating and analyzing synthetic partial discharge
signals, mimicking those found in high-voltage equipment monitoring. It is intended
for testing and evaluation purposes in simplified settings.

- Data is generated as numpy arrays simulating good and faulty signals.
- Each signal consists of a 1D array with synthetic features.
- This script mimics a scenario where data from multiple stations (e.g., 52009, 52008)
  is collected, though only a small subset is generated here for testing.

Example station data (real scenario):
    idStation   total_count
    52009       23641
    52008       23626
    52007       22969
    52012       22762
    52014       21608
    52013       18588
    52011       10016
    52010        8224
"""


import numpy as np
import pandas as pd


def generate_synthetic_partial_discharge(
        num_good: int = 20,
        num_fault: int = 20,
        length: int = 40,
        seed: int | None = 0,
) -> pd.DataFrame:
    """
    Generate synthetic signals representing partial discharge events.

    This function creates a toy dataset consisting of two classes:
    - "Good" signals: normal background noise.
    - "Fault" signals: background noise with superimposed synthetic pulses
      representing discharge activity.

    Each signal is a 1D array of fixed length with added Gaussian noise. Faulty signals
    include additional sharp pulses at random positions to simulate discharge.

    Parameters
    ----------
    num_good : int, default=20
        Number of non-discharge (normal) samples to generate.
    num_fault : int, default=20
        Number of discharge (faulty) samples to generate.
    length : int, default=40
        Length of each signal (number of time steps).
    seed : int or None, default=0
        Seed for the random number generator for reproducibility.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing generated signals.
        - Each row represents a signal.
        - Columns s0 to s{length-1} are signal values.
        - Column 'target' contains labels: 0 (good), 1 (fault).
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
