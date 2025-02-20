from data_loader import load_data
from data_processor import preprocess_data
from model_runner import train_model, evaluate_model
from parameter_tuner import tune_hyperparameters
from feature_selector import select_features_featurewiz
import logging
import argparse
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="ML Workflow Script")
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration JSON file')
    args = parser.parse_args()

    # Load configuration from JSON file
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = { # Default configuration if config.json is not found
            "dataset_name": "iris",
            "model_type": "RandomForest",
            "scaling_method": "standard",
            "feature_selection": "featurewiz",
            "hyperparameter_tuning": True,
            "n_trials_optuna": 10,
            "cv_folds": 3,
            "test_size": 0.2,
            "val_size": 0.2,
            "random_state": 42,
            "target_column": "target" # Assuming 'target' column exists after loading
        }
        logging.warning(f"Configuration file '{args.config}' not found. Using default configuration.")

    logging.info("Starting ML Workflow...")
    logging.info(f"Configuration: {config}")

    # 1. Load Data
    df = load_data(dataset_name=config.get('dataset_name'))
    target_column = config.get('target_column')

    # 2. Feature Selection (optional)
    if config.get('feature_selection') == 'featurewiz':
        selected_features = select_features_featurewiz(df.copy(), target_column) # Use .copy()
    else:
        selected_features = df.columns.drop(target_column).tolist() # Use all features if no feature selection

    # 3. Data Preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        df,
        features=selected_features,
        target=target_column,
        scaling_method=config.get('scaling_method'),
        test_size=config.get('test_size'),
        val_size=config.get('val_size'),
        random_state=config.get('random_state')
    )

    # 4. Hyperparameter Tuning (optional)
    if config.get('hyperparameter_tuning'):
        best_params = tune_hyperparameters(X_train, y_train, model_type=config.get('model_type'), n_trials=config.get('n_trials_optuna'), cv=config.get('cv_folds'))
    else:
        best_params = None # Use default parameters in train_model

    # 5. Model Training
    model, cv_score = train_model(X_train, y_train, model_type=config.get('model_type'), params=best_params, cv=config.get('cv_folds'))

    # 6. Model Evaluation
    evaluate_model(model, X_test, y_test)

    logging.info("ML Workflow completed.")

if __name__ == "__main__":
    main()
