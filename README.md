# Partial Discharge Feature Comparison

This project evaluates different feature extraction methods and machine learning models for partial discharge classification.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

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
