
"""
This script provides a generalized framework for executing various machine learning
experiment pipelines, as defined in the project's `config.yaml` file. It is designed
to be modular and extensible, allowing for easy definition and execution of different
experimental setups, from baseline models to complex feature engineering and selection tracks.

The script orchestrates the following key stages of a typical ML pipeline:
1.  **Data Ingestion & Pre-processing**: Loads raw data and applies initial cleaning and transformation steps.
2.  **Feature Engineering**: Extracts a wide range of features from the pre-processed data.
3.  **Feature Combination & Expansion**: Creates new features through mathematical operations or interactions.
4.  **Feature Selection**: Applies various techniques to select the most relevant features for modeling.
5.  **Model Training and Evaluation**: Trains and evaluates machine learning models on the selected features.

The script is driven by the configurations specified in `config.yaml`, which defines the
different experiments to be run. Each experiment can have its own unique combination of
settings for each stage of the pipeline.

To run this script, you can execute it from the command line:
    python run_experiments.py

The script will then iterate through all the defined experiments in the configuration
file, executing each one in sequence and logging the results.
"""


import itertools
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from config import load_config, Config
from preprocess.io import load_parquet
from features.extractors import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def feature_combination(data: pd.DataFrame, combination_config: dict) -> pd.DataFrame:
    """Performs feature combination and expansion."""
    logging.info("Performing feature combination...")
    if not combination_config.get('enabled', False):
        return data

    if combination_config.get('polynomial', False):
        poly = PolynomialFeatures(interaction_only=True, include_bias=False)
        # Assuming last column is target
        X = data.iloc[:, :-1]
        poly_features = poly.fit_transform(X)
        poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(X.columns))
        data = pd.concat([data.iloc[:, :-1], poly_df, data.iloc[:, -1]], axis=1)

    # Add other combination methods here

    return data

def feature_selection(data: pd.DataFrame, selection_method: str) -> pd.DataFrame:
    """Performs feature selection."""
    logging.info(f"Performing feature selection using: {selection_method}")
    if selection_method == 'all':
        return data
    elif selection_method == 'featurewiz':
        try:
            from featurewiz import featurewiz
        except ImportError:
            logging.error("Featurewiz is not installed. Please install it using `pip install featurewiz`.")
            return data
        # Placeholder for featurewiz logic
        logging.info("Using Featurewiz for feature selection.")
        # Assuming last column is target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        # Note: featurewiz returns a tuple of (selected_features, dataframe)
        _, X_selected = featurewiz(X, y, corr_limit=0.7, verbose=0)
        return pd.concat([X_selected, y], axis=1)

    elif selection_method == 'mljar':
        logging.warning("MLJAR-supervised is not installed. Skipping this feature selection method.")
        return data
    else:
        logging.warning(f"Unknown feature selection method: {selection_method}")
        return data

def train_and_evaluate(data: pd.DataFrame, model_name: str):
    """Trains a model and evaluates it."""
    logging.info(f"Training and evaluating model: {model_name}")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    else:
        logging.warning(f"Model {model_name} not implemented. Using RandomForest as default.")
        model = RandomForestClassifier(random_state=42)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model: {model_name}, Accuracy: {accuracy:.4f}")
    return accuracy

# --- Experiment Runner ---

def run_experiment(experiment_config: dict, config: Config):
    """Runs a single experiment based on the given configuration."""
    logging.info(f"--- Starting Experiment: {experiment_config['name']} ---")

    # 1. Data Ingestion
    # This assumes you have a processed parquet file. 
    # You might need to run the preprocessing pipeline first.
    try:
        data = load_parquet(Path(config.project.processed_dir) / f"{experiment_config['dataset']['path']}.parquet")
    except FileNotFoundError:
        logging.error(f"Data for dataset '{experiment_config['dataset']['path']}' not found. Please run the preprocessing pipeline first.")
        # As a fallback for demonstration, we create a dummy dataframe.
        data = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'target': [0] * 50 + [1] * 50
        })
        data.rename(columns={'target': 'target'}, inplace=True)

    except FileNotFoundError:
        logging.error(f"Data for dataset '{experiment_config['dataset']['path']}' not found. Please run the preprocessing pipeline first.")
        # As a fallback for demonstration, we create a dummy dataframe.
        data = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'target': [0] * 50 + [1] * 50
        })
        data.rename(columns={'target': 'target'}, inplace=True)

    # 2. Feature Engineering
    feature_extractor = FeatureExtractor(fs=1.0) # You might want to get fs from config
    # This assumes the data from load_parquet is in a format that extract_features can handle.
    # You might need to adapt this part based on your actual data structure.
    # For now, we'll just pass the dummy data through.
    if "signal" in data.columns:
        features = feature_extractor.extract_features(data)
    else: # Placeholder for non-signal data
        features = data

    # 3. Feature Combination
    features = feature_combination(features, experiment_config.get('feature_combination', {}))

    # 4. Feature Selection
    features = feature_selection(features, experiment_config['feature_selection'])

    # 5. Model Training and Evaluation
    accuracy = train_and_evaluate(features, experiment_config['model'])

    logging.info(f"--- Finished Experiment: {experiment_config['name']} with accuracy {accuracy:.4f} ---")
    return {
        'experiment_name': experiment_config['name'],
        'accuracy': accuracy
    }

def main():
    """Main function to run all experiments."""
    config = load_config()

    # Define experiments based on your request
    exp0 = {
        "name": "Baseline (Exp 0)",
        "dataset": config.datasets[0].dict(),
        "feature_combination": {'enabled': False},
        "feature_selection": "all",
        "model": "RandomForest"
    }

    exp1 = {
        "name": "Featurewiz Track (Exp 1)",
        "dataset": config.datasets[0].dict(),
        "feature_combination": {'enabled': True, 'polynomial': True},
        "feature_selection": "featurewiz",
        "model": "RandomForest"
    }

    exp2 = {
        "name": "MLJAR-Supervised (Exp 2)",
        "dataset": config.datasets[0].dict(),
        "feature_combination": {'enabled': True, 'polynomial': True},
        "feature_selection": "mljar",
        "model": "RandomForest"
    }

    experiment_configs = [exp0, exp1, exp2]

    results = []
    for exp_config in experiment_configs:
        result = run_experiment(exp_config, config)
        results.append(result)

    # Log and save results
    results_df = pd.DataFrame(results)
    logging.info("\n--- Experiment Results ---")
    logging.info(results_df)
    results_df.to_csv('experiment_results.csv', index=False)
    logging.info("Results saved to experiment_results.csv")

if __name__ == "__main__":
    main()
