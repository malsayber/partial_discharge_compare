"""Minimal script walking through PD cleaning and feature extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

try:  # featurewiz is optional and heavy
    from featurewiz import featurewiz
except Exception as exc:  # pragma: no cover - optional dependency
    featurewiz = None
    print("featurewiz unavailable:", exc)

# Add project root to ``sys.path`` when running from ``notebooks/``
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from preprocess.cleaning import denoise_signal
from features.extractors import FeatureExtractor

def load_data(file_path):
    """
    Loads raw time-series PD signals from a .npy file.

    This is the first step in the workflow, where we load the raw data.
    The data is expected to be a numpy array.

    Other tracks to explore:
    - Loading data from other formats like CSV, Parquet, or directly from a database.
    - Handling larger-than-memory datasets using libraries like Dask or Vaex.
    """
    return np.load(file_path)

def preprocess_data(signal):
    """
    Applies standard denoising to the raw signal.

    This step is crucial for removing noise from the signal, which can
    significantly improve the quality of the extracted features. We use a
    standard denoising technique here.

    Other tracks to explore:
    - Advanced denoising techniques: Wavelet denoising, Kalman filters, etc.
    - Synthetic data augmentation: Creating more data by adding noise, shifting, or scaling the signal.
    - Outlier detection and removal: Identifying and removing anomalous data points.
    """
    # Assume an example sampling rate of 100 MHz so the default
    # band-pass frequencies from ``config.yaml`` are valid.
    return denoise_signal(signal, fs=100_000_000)

def extract_features(denoised_signal):
    """
    Extracts classic statistical features from the denoised signal.

    Feature extraction is the process of creating new features from the existing
    data. These features can help machine learning models to better understand
    the data. Here, we extract some classic statistical features.

    Other tracks to explore:
    - More advanced features: Time-frequency domain features (e.g., from CWT or STFT),
      non-linear features, or features from pre-trained models.
    - Automated feature engineering: Using libraries like featuretools to automatically
      generate features.
    """
    # Pass the same sampling rate used during preprocessing
    extractor = FeatureExtractor(fs=100_000_000)
    # The extractor expects a DataFrame, so we convert the signal
    signal_df = pd.DataFrame({'signal': denoised_signal})
    features = extractor.extract_features(signal_df)
    return features

def apply_pairwise_math(features):
    """
    Applies pairwise math to the extracted features.

    This step can be used to compute relationships between features or samples.
    Here we compute the pairwise distances between features as an example.

    Other tracks to explore:
    - Other pairwise metrics: Cosine similarity, correlation, etc.
    - Graph-based analysis: Constructing a graph from the pairwise matrix and
      analyzing its properties.
    """
    return pd.DataFrame(pairwise_distances(features.T, metric='euclidean'),
                        index=features.columns, columns=features.columns)

def select_features(features, target):
    """
    Performs feature selection using featurewiz.

    Feature selection is the process of selecting a subset of relevant features
    for use in model construction. This can help to improve model performance,
    reduce overfitting, and decrease training time.

    Other tracks to explore:
    - Other feature selection methods: Recursive Feature Elimination (RFE),
      L1-based feature selection, or permutation importance.
    - Dimensionality reduction techniques: Principal Component Analysis (PCA),
      t-SNE, or UMAP.
    """
    # featurewiz expects the target variable to be in the same dataframe
    # Here we create a dummy target for demonstration purposes.
    # In a real scenario, you would use your actual target variable.
    if featurewiz is None:
        # Fallback to returning all feature names
        return list(features.columns)
    try:
        features['target'] = target
        fwiz = featurewiz(features, 'target', corr_limit=0.7, verbose=0)
        selected_features, _ = fwiz.feature_selection()
        return selected_features
    except Exception as exc:  # pragma: no cover - fallback path
        print("featurewiz failed:", exc)
        return list(features.columns)

def main():
    """
    Main function to run the example workflow.
    """
    # 1. Load data
    # The path is relative to the root of the project
    file_path = 'unitest/data/748987.npy'
    raw_signal = load_data(file_path)

    # 2. Preprocess data
    denoised_signal = preprocess_data(raw_signal)

    # 3. Extract features
    # For demonstration, we'll treat each data point in the signal as a sample
    # and extract features for each. In a real scenario, you might use windowing.
    features = extract_features(denoised_signal)
    print("Extracted Features:")
    print(features.head())


    # 4. Apply pairwise math
    pairwise_matrix = apply_pairwise_math(features)
    print("\nPairwise Distance Matrix:")
    print(pairwise_matrix.head())

    # 5. Select features
    # Create a dummy target variable for demonstration
    np.random.seed(42)
    dummy_target = np.random.randint(0, 2, size=features.shape[0])
    selected_features = select_features(features.copy(), dummy_target)
    print(f"\nSelected features: {selected_features}")


if __name__ == '__main__':
    main()
