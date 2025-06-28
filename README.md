# Partial Discharge Compare


Utilities and workflows for comparing partial discharge models.
=======
This project provides a modular machine learning pipeline for experimenting with different models, feature selection techniques and hyperparameter tuning. It was initially created to compare approaches for partial discharge classification, but the code is generic enough to be used on datasets such as the Iris dataset.

## Folder Structure

```
.
├── ml_flow/            # Main library containing the workflow scripts
│   ├── data_loader.py  # Load datasets
│   ├── data_processor.py
│   ├── feature_selector.py
│   ├── helper.py
│   ├── model_runner.py
│   ├── parameter_tuner.py
│   ├── main.py         # Entry point for running experiments
│   └── config.json     # Example configuration
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

The workflow writes intermediate data under `data/`, saved models under `model/` and logs under `log/`. These directories are created at runtime.

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the project requirements:
=======
# Partial Discharge Feature Comparison

This project evaluates different feature extraction methods and machine learning models for partial discharge classification.

## Installation

Create a Python environment and install dependencies:


```bash
pip install -r requirements.txt
```

## Example Usage

The workflow is configured via a configuration file. Although the repository ships with a JSON file, it can easily be represented in YAML as shown below:

```yaml
# config.yaml
datasets:
  - path: iris
    label_mapping: {}
model_types: [RandomForest, XGBoost, LogisticRegression, SVM]
scaling_methods: [standard]
feature_selections: [null, featurewiz]
hyperparameter_tuning: true
n_trials_optuna: 10
cv_folds: 3
test_size: 0.2
val_size: 0.2
random_state: 42
target_column: target
```

Each entry under `datasets` specifies the dataset `path` and an optional
`label_mapping` used to remap values in the target column.

Run the main script using your configuration file:

```bash
python ml_flow/main.py --config config.yaml
```

## Running Tests

If unit tests are added to the project, they can be executed with [pytest](https://docs.pytest.org/):

```bash
pytest
```

## Contributing

1. Fork the repository and create a new branch for your feature or fix:

```bash
git checkout -b feature/my-new-feature
```

2. Commit your changes and push the branch to your fork.
3. Open a pull request targeting the `main` branch. Include a clear description of your changes and reference any related issues.

=======
## Running Experiments

Experiments are configured through `ml/config.yaml`. To execute the workflow:

```bash
python -m ml.main --config ml/config.yaml
```

Results and models are saved in the `ml/` directory.

## Testing

Run unit tests with:

```bash
pytest -q
```

## Folder Overview

- `feature_extraction/` – individual feature functions
- `ml/` – data loading, preprocessing and model training utilities
- `reports/` – MNE report generation
- `unitest/` – unit tests and test fixtures
- `notebooks/` – tutorial notebooks
- `docs/` – additional documentation and design notes

## Citation

If you use this repository in academic work, please cite accordingly.