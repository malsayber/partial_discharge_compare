# Partial Discharge Compare

This project contains a modular pipeline for partial discharge (PD) signal classification.
Stages of the pipeline are invoked via `ml/main.py` and configured through `config.yaml`.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run a specific stage with:

```bash
python -m ml.main --stage preprocess
```

Use `--stage full-run` to execute all stages sequentially. Flags such as
`--advanced-denoise` and `--augment` toggle optional preprocessing steps.

## Directory Structure

```
project_pd/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── features/
│   ├── models/
│   └── reports/
└── logs/
```

Synthetic test signals are located under `unitest/fixtures/` for unit testing.
