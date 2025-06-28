import optuna
from .model_runner import train_model  # Import train_model
from .data_processor import preprocess_data  # Import preprocess_data
from .data_loader import load_data  # Import load_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def objective(trial, model_type, X_train, y_train, cv):
    """
    Objective function for Optuna optimization.
    Defines the hyperparameter search space for each model type.
    """
    logging.info(f"Starting Optuna trial for {model_type}...")
    if model_type == 'DecisionTree':
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
    elif model_type == 'SVM':
        params = {
            'C': trial.suggest_float('C', 1e-5, 1e2, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
        }
    elif model_type == 'LogisticRegression':
        params = {
            'C': trial.suggest_float('C', 1e-5, 1e2, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
            'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 300)
        }
    elif model_type == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
        }
    elif model_type == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
    else:
        raise ValueError(f"Model type '{model_type}' not supported for tuning.")

    _, cv_score = train_model(X_train, y_train, model_type=model_type, params=params, cv=cv, scoring='accuracy')
    return cv_score # Optuna minimizes, so return negative accuracy if maximizing

def tune_hyperparameters(X_train, y_train, model_type, n_trials=10, cv=3):
    """
    Tunes hyperparameters for a given model type using Optuna.
    """
    logging.info(f"Starting hyperparameter tuning for {model_type} using Optuna...")
    study = optuna.create_study(direction='maximize') # We want to maximize accuracy
    study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, cv), n_trials=n_trials)

    logging.info("Hyperparameter tuning finished.")
    logging.info("  Best trial:")
    trial = study.best_trial
    logging.info(f"    Value (Mean CV Accuracy): {trial.value:.4f}")
    logging.info("    Params: ")
    for key, value in trial.params.items():
        logging.info(f"      {key}: {value}")

    return study.best_params

if __name__ == '__main__':
    df = load_data()
    X_train, _, _, y_train, _, _ = preprocess_data(df, target='target')  # Only need training data for tuning

    best_params = tune_hyperparameters(X_train, y_train, model_type='RandomForest', n_trials=5) # Example for RandomForest
    print("\nBest Hyperparameters for RandomForest:")
    print(best_params)
