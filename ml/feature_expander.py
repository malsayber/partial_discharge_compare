"""Feature expansion using featurewiz."""
from __future__ import annotations

import logging
import pandas as pd
from featurewiz import featurewiz

logger = logging.getLogger(__name__)


def expand_features_featurewiz(
    df: pd.DataFrame,
    target_column: str,
    corr_limit: float = 0.7,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Generate additional features using featurewiz.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing ``target_column``.
    target_column : str
        Name of the target column.
    corr_limit : float
        Correlation limit for featurewiz feature engineering.
    n_jobs : int
        Number of parallel workers.

    Returns
    -------
    pandas.DataFrame
        DataFrame with engineered features selected by featurewiz.
    """
    logger.info("Expanding features with featurewiz...")
    features, train_df = featurewiz(
        df,
        target=target_column,
        corr_limit=corr_limit,
        verbose=2,
        feature_engg=True,
        category_encoders=None,
        n_jobs=n_jobs,
    )
    logger.info("Feature expansion completed: %s features", len(features))
    return train_df[features + [target_column]]
