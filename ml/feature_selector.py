from featurewiz import featurewiz
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def select_features_featurewiz(df, target_column, corr_limit=0.7, feature_engg=False, category_encoders=None, n_jobs=-1):
    """
    Selects features using featurewiz.
    """
    logger.info("Starting feature selection using featurewiz...")
    features, trainm = featurewiz(
        df,
        target=target_column,
        corr_limit=corr_limit,
        verbose=2,
        feature_engg=feature_engg,
        category_encoders=category_encoders,
        n_jobs=n_jobs
    )
    logger.info("Feature selection using featurewiz complete.")
    logger.info(f"Selected features: {features}")
    return features

if __name__ == '__main__':
    from .data_loader import load_data
    df = load_data()
    target_column = 'target' # Assuming 'target' column exists
    selected_features = select_features_featurewiz(df.copy(), target_column) # Use .copy() to avoid modifying original df
    print("\nSelected Features by featurewiz:")
    print(selected_features)
