import json
import logging
import os

def load_config(config_path='config.json'):
    """
    Loads configuration from a JSON file. If the file is not found or JSON is invalid,
    returns a default configuration and logs a warning.
    """
    default_config = {
        "dataset_name": "iris",
        "model_types": ["RandomForest", "XGBoost", "LogisticRegression", "SVM"],
        "scaling_methods": ["standard"],
        "feature_selections": ["featurewiz"],
        "hyperparameter_tuning": True,
        "n_trials_optuna": 10,
        "cv_folds": 3,
        "test_size": 0.2,
        "val_size": 0.2,
        "random_state": 42,
        "target_column": "target"
    }
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logging.info(f"Configuration loaded from '{config_path}'")
            return config
        else:
            logging.warning(f"Configuration file '{config_path}' not found. Using default configuration.")
            return default_config
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from '{config_path}'. Using default configuration.")
        return default_config
    except Exception as e:
        logging.error(f"Unexpected error loading configuration from '{config_path}': {e}. Using default configuration.")
        return default_config
