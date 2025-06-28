## 📌 Role: Enhanced Code Quality for `partial_discharge_compare`

This document guides AI agents in generating consistently high-quality, production-ready code for evaluating feature extraction and machine learning methods on partial discharge signals.

---

## 🎯 Goals

* ✅ Accelerate AI onboarding into this repository
* ✅ Ensure consistency and adherence to project standards
* ✅ Minimize manual review
* ✅ Improve overall reliability, maintainability, and testability

---

## 🧠 Project Overview

* **Purpose**: Compare handcrafted and automated features with various classifiers for partial discharge classification.
* **Structure**: Organized into feature extraction, machine learning utilities, reports, and unit tests.
* **Primary Interface**: `ml.main` executes experiments defined in `config.yaml`.
* **Data Input**: Numpy arrays or pandas DataFrames of synthetic or real signals.
* **Output**: Performance metrics, trained models, and HTML reports.

---

## 📁 Folder Structure

```bash
feature_extraction/        # Individual feature functions
ml/                        # Data loading, preprocessing, and model training
reports/                   # MNE report generation
unitest/                   # Unit tests and fixtures
notebooks/                 # Tutorial notebooks
```

---

## 📌 Key Coding Practices

### 📄 Input Handling and Testing

* Synthetic partial discharge signals are provided in `unitest.fixtures.synthetic_pd`.
* Functions must clearly accept numpy arrays and return standard Python scalars or pandas objects.
* Provide thorough tests for both feature functions and ML utilities using the synthetic data.

### 📂 Feature Implementation Guidelines

* **Separate Python files** for each feature (e.g., `time_mean.py`, `spectral_entropy.py`).
* **Group related features** within the `feature_extraction` directory.
* **Dedicated unit tests** verify each feature calculation.
* Allow users to compute selected features or all by default.

### 🪵 Logging Standards

* Root logger configured in `ml/main.py`.
* Each module initializes `logger = logging.getLogger(__name__)`.
* Use `INFO` for major steps and `DEBUG` for internal details.

### ⏳ Progress Feedback

* Employ `tqdm` for lengthy iterations when applicable.

### 📄 Docstrings and Type Hints

* Use Google-style docstrings.
* Provide type annotations for all public functions and tests.
* Document parameters, returns, and exceptions.

### 🧪 Comprehensive Unit Testing

* Use `pytest` or Python's `unittest` in the `unitest/` folder.
* Tests rely on synthetic signals to avoid external data dependencies.

---

## ✅ Conventions for Consistency

| Aspect             | Standard                                          |
| ------------------ | ------------------------------------------------- |
| Function Naming    | `snake_case`                                      |
| Class Naming       | `PascalCase`                                      |
| Variable Naming    | `snake_case`                                      |
| Imports            | Standard → third-party → local                    |
| Data Storage       | Reports saved in `reports/`                       |
| DataFrame Outputs  | Clearly named columns                             |

---

## 🧩 Modularization Guidance

* Modules must be self-contained and testable.
* Minimize cross-module dependencies and avoid side effects.
* Accept raw arrays or DataFrames; return DataFrames or standard Python types.
