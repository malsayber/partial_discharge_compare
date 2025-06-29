
## **Modular Flowchart – Partial Discharge (PD) Classification Pipeline**

This pipeline is broken into **independently executable stages** controlled by a command-line argument `--stage`.
Run the entire workflow (`full-run`) **or** jump straight to *pre-processing*, *feature extraction*, *analysis* or *reporting* as needed.

---

#### **`START`**

##### **Phase 0 : Initialisation & Dispatch**

1. **`[Process]`** **Execute `main.py`**
2. **`[Process]`** **Parse Command-Line Arguments**

    * `--stage` (controller)

        * `full-run`  → Ph 0 → 1 → 2 → 3
        * `preprocess` → only Phase 0
        * `extract`    → only Phase 1
        * `analyze`    → only Phase 2
        * `report`     → only Phase 3
    * `--force`    (overwrite any existing master‐feature file when used with `--stage extract`)
    * `--jobs`     (parallel workers for CPU-bound loops)
    * Optional flags (examples):

        * `--advanced-denoise`  `--augment`  `--wavelet-feats`
        * `--optuna` (use Bayesian HPO instead of GridSearch)
3. **`[Process]`** **Load `config.yaml` & Set Up Logging**

    * Contains sensor directories, feature sets, model grids, experiment matrix…
4. **`[Decision]`** **Route to the Requested Stage**

    * `preprocess` → **Phase 0** ↴
    * `extract`    → **Phase 1** ↴
    * `analyze`    → **Phase 2** ↴
    * `report`     → **Phase 3** ↴
    * `full-run`   → continue sequentially (Ph 0 → 1 → 2 → 3)

---

## 🔵 **Phase 0 – Pre-processing & Windowing** (`--stage preprocess`)

*Goal:* From every **raw PD measurement file** (e.g. CSV/HDF5 recorded via HFCT/UHF sensors) create **clean, normalised, windowed segments** ready for feature extraction.

5. **`[Data Input]`** Discover Measurement Sessions

    * Scan `dataset/raw_pd/*/*.csv` (one folder = one cable test).
    * Build a list of `session_record`s:

      ```python
      {
        'cable_id': 'Cable_01',
        'sensor_files': {
            'HFCT': '/path/Cable_01/HFCT.csv',
            'UHF' : '/path/Cable_01/UHF.csv'   # optional
        },
        'label_file': '/path/Cable_01/labels.json'  # ground-truth PD type
      }
      ```

6. **`[Loop START]`** *FOR each* `session_record`:

   a. **`[Process]`** **Load Raw PD Signals**

   ```python
   raw_hfct = load_pd_csv(sensor_files['HFCT'])
   raw_uhf  = load_pd_csv(sensor_files.get('UHF'))
   ```

   b. **`[Process]`** **Signal Cleaning**

    * Band-pass filter (100 kHz – 30 MHz)
    * Wavelet or VMD denoising ***(if `--advanced-denoise` flag)***
    * Amplitude normalisation (z-score or min-max)
    * Optional data augmentation (time-warp/jitter) ***(if `--augment`)***

   c. **`[Process]`** **Segmentation / Windowing**

    * Split into fixed-length windows (e.g. 5 ms) **or** phase-locked windows (based on 50/60 Hz mains)
    * Attach labels from `label_file` (e.g. `internal`, `surface`, `corona`, `no_pd`)

   d. **`[Data Output]`** Save cleaned windowed arrays →

   ```
   dataset/processed/{cable_id}/{sensor}/{window_id}.npy
   ```

7. **`[Loop END]`** (all cables processed)

8. **`[Decision]`** If initial stage == `full-run` → **Phase 1** else **END**

---

## 🟡 **Phase 1 – Feature Engineering & Combination** (`--stage extract`)

*Goal:* Compute **base** and **expanded** features for every window; consolidate into one master feature table.

9. **`[Process]`** For each windowed file:

    * **Base Feature Extraction** (time, frequency, entropy, PRPD metrics)
    * **Wavelet-based & multiscale entropy** ***(if `--wavelet-feats`)***

10. **`[Process]`** **Feature Expansion**

    * Pairwise arithmetic ops (+ − × ÷)
    * `PolynomialFeatures(interaction_only=True)`
    * `feature-engine` mathematical combinations
    * `autofeat` / symbolic regression ***(optional)***

11. **`[Data Output]`** Append to `features_master.parquet`

    * Overwritten only if `--force` is supplied.

12. **`[Decision]`** If initial stage == `full-run` → **Phase 2** else **END**

---

## 🟢 **Phase 2 – Feature Selection & Model Analysis** (`--stage analyze`)

*Goal:* Select optimal feature subsets, train a suite of ML models, and evaluate them.

13. **`[Process]`** **Load Master Feature Table**
14. **`[Process]`** **Instantiate Selection Tracks**

    * **Track A (Baseline):** use all features
    * **Track B (Featurewiz):** SULOV + XGB ranking
    * **Track C (MLJAR):** internal selector
    * **Track D (Optuna subset)** ***(if `--optuna`)***
15. **`[Loop START]`** *FOR each* selection track:
    a. Apply selector → `X_selected`
    b. **`[Loop]`** over model zoo (ridge, LDA, SVM, RF, Extra, GBDT, XGB, LGBM, CatBoost, …)

    * Hyper-parameter search:

        * **GridSearchCV** (default)
        * **OptunaSearchCV** ***(if `--optuna`)***
    * Cross-validated scoring (Accuracy, F1, Recall)
    * Save best estimator (`joblib`) and CV metrics.
16. **`[Data Output]`** `results/summary_metrics.csv` (all experiments)
17. **`[Decision]`** If initial stage == `full-run` → **Phase 3** else **END**

---

## 🟣 **Phase 3 – Reporting & Packaging** (`--stage report`)

*Goal:* Produce human-readable results, model explainability plots, and a deployable artifact.

18. **`[Process]`** Generate Markdown / PDF report

    * Confusion matrices
    * Precision-Recall & ROC curves
    * SHAP global & local explanations (top model)
    * Feature importance tables
19. **`[Process]`** Export winning model to **ONNX** and register in `model_registry/`
20. **`[Process]`** (Optional) Push run artefacts to **MLflow / W\&B**
21. **`[END]`** — Pipeline complete.

---

## 📌 Key Coding Practices

### 📄 Input Handling and Testing

* Synthetic partial discharge signals are provided in `unitest.fixtures.synthetic_pd`.
* Functions must clearly accept numpy arrays and return standard Python scalars or pandas objects.
* Provide thorough tests for both feature functions and ML utilities using the synthetic data.

### 📂 Feature Implementation Guidelines

* For every major pipeline, save the results in `outputs/features/` or `output/preprocessing`.
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
