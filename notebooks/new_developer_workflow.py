"""Developer workflow demo for PD cleaning and feature engineering.

This example walks through the minimal pipeline so new contributors can
experiment step by step.  The workflow is intentionally linear:

1. **Load** the sample signal ``unitest/data/748987.npy``.
2. **Preprocess** it with :func:`denoise_signal` (band-pass filter and
   normalisation).
3. **Extract features** using :class:`~features.extractors.FeatureExtractor`.
4. **Compute** a pairwise distance matrix between those features.
5. **Select features** via ``featurewiz`` (SULOV + XGB ranking).

INFO logs report how many features exist at each stage so you can observe
the effect of selection.  Run the file line by line in a Python session to
adapt the workflow for your own data.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from featurewiz import featurewiz

# Add project root to ``sys.path`` when running from ``notebooks/``
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from preprocess.cleaning import denoise_signal
from features.extractors import FeatureExtractor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def load_data(file_path: Path) -> np.ndarray:
    """Load a sample PD signal from disk.

    Args:
        file_path: Path to the ``.npy`` file containing the signal.

    Returns:
        Loaded 1D numpy array.

    Other tracks to explore:
        * Loading data from CSV or Parquet files.
        * Handling larger-than-memory datasets with Dask or Vaex.
    """
    logger.info("Loading sample signal from %s", file_path)
    return np.load(file_path)

def preprocess_data(signal: np.ndarray) -> np.ndarray:
    """Apply standard denoising to the raw signal.

    Args:
        signal: Raw signal array.

    Returns:
        Cleaned and normalised signal.

    Other tracks to explore:
        * Advanced denoising such as wavelets or Kalman filters.
        * Synthetic augmentation (noise, shifts, scaling).
        * Outlier detection and removal.
    """
    logger.info("Preprocessing signal")
    # Assume an example sampling rate of 100 MHz so the default
    # band-pass frequencies from ``config.yaml`` are valid.
    return denoise_signal(signal, fs=100_000_000)

def extract_features(denoised_signal: np.ndarray) -> pd.DataFrame:
    """Extract basic statistical features from the signal.

    Args:
        denoised_signal: Cleaned signal from :func:`preprocess_data`.

    Returns:
        DataFrame with one row of extracted features.

    Other tracks to explore:
        * Time-frequency or non-linear features.
        * Automated feature engineering libraries such as ``featuretools``.
    """
    logger.info("Extracting features")
    # Pass the same sampling rate used during preprocessing
    extractor = FeatureExtractor(fs=100_000_000)
    # The extractor expects a DataFrame, so we convert the signal
    signal_df = pd.DataFrame({'signal': denoised_signal})
    features = extractor.extract_features(signal_df)
    return features

def apply_pairwise_math(features: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise distances between features.

    Args:
        features: DataFrame returned by :func:`extract_features`.

    Returns:
        DataFrame of pairwise Euclidean distances between feature columns.

    Other tracks to explore:
        * Cosine similarity or correlation matrices.
        * Graph-based analysis on the resulting distance matrix.
    """
    logger.info("Computing pairwise distances")
    return pd.DataFrame(pairwise_distances(features.T, metric='euclidean'),
                        index=features.columns, columns=features.columns)

def select_features(features: pd.DataFrame, target: np.ndarray) -> list[str]:
    """Perform feature selection using ``featurewiz``.

    Args:
        features: DataFrame of features with one row per sample.
        target: Array of target labels corresponding to ``features``.

    Returns:
        List of selected feature names.

    Other tracks to explore:
        * Recursive feature elimination or permutation importance.
        * Dimensionality reduction via PCA, t-SNE or UMAP.
    """
    # ``featurewiz`` expects the target column to be present in the
    # input DataFrame. In this example we generate a dummy binary target
    # array because the sample signal has no labels. Replace
    # ``dummy_target`` in :func:`main` with your own target values.
    features['target'] = target
    try:
        fwiz = featurewiz(features, 'target', corr_limit=0.7, verbose=0)
        selected_features, _ = fwiz.feature_selection()
        return selected_features
    except Exception as exc:  # pragma: no cover - handle unexpected errors
        logger.warning("featurewiz failed: %s", exc)
        return list(features.columns)

def main() -> None:
    """Run the end-to-end developer workflow example.

    The function sequentially loads a sample signal, preprocesses it,
    extracts a feature vector, computes a pairwise distance matrix and
    finally performs feature selection.  The results are reported via
    INFO logs so you can track the number of features at each stage.
    """
    logger.info("Starting developer workflow example")
    # 1. Load data
    # Always resolve the sample path from the project root so this
    # script works no matter the current working directory.
    file_path = ROOT / 'unitest' / 'data' / '748987.npy'
    raw_signal = load_data(file_path)
    logger.debug("Raw signal length: %d", len(raw_signal))

    # 2. Preprocess data
    denoised_signal = preprocess_data(raw_signal)
    logger.debug("Denoised signal length: %d", len(denoised_signal))

    # 3. Extract features
    # For demonstration, we'll treat each data point in the signal as a sample
    # and extract features for each. In a real scenario, you might use windowing.
    features = extract_features(denoised_signal)
    feature_count = features.shape[1]
    logger.info("Extracted %d base features", feature_count)
    logger.debug("Features DataFrame shape: %s", features.shape)
    logger.debug("Example features:\n%s", features.head())


    # 4. Apply pairwise math
    pairwise_matrix = apply_pairwise_math(features)
    logger.info("Pairwise distance matrix shape: %s", pairwise_matrix.shape)
    logger.debug("Pairwise distance matrix head:\n%s", pairwise_matrix.head())

    # 5. Select features
    # ``featurewiz`` needs a target column in the DataFrame. Because our
    # example signal has no labels, we create a random binary target purely
    # to satisfy the API. Replace this with your actual labels when adapting
    # the workflow.
    np.random.seed(42)
    dummy_target = np.random.randint(0, 2, size=features.shape[0])
    selected_features = select_features(features.copy(), dummy_target)
    logger.info("Selected %d of %d features", len(selected_features), feature_count)
    logger.debug("Selected features: %s", selected_features)
    logger.info(
        "Workflow complete. Retained %d of %d base features",
        len(selected_features),
        feature_count,
    )


if __name__ == '__main__':
    main()
