Below is a **developer‐oriented task breakdown**, organized by pipeline phase. Each item describes exactly what to code or configure so that a programmer can implement the PD-classification pipeline end-to-end.

---

## 🔵 Phase 0 – Pre-processing & Signal Conditioning (`--stage preprocess`)

### 0.1 CLI & Logging Setup

1. **File:** `ml/main.py`

    * Add `argparse` support for `--stage`, `--force`, `--jobs`, plus flags (`--advanced-denoise`, `--augment`).
    * Initialize a root logger (Python `logging`) that writes to `logs/main.log`.

        * Log level is driven by `config.runtime.verbosity`.
        * Use a log filename that includes a timestamp and `--stage` name.
    * At startup, log:

      ```text
      --- Experiment Start ---
      Config: <absolute path to config.yaml>
      Git commit: <git rev-parse HEAD>
      Stage: `<stage>`
      Flags: {...}
      ------------------------
      ```

2. **File:** `config.py`

    * Write a helper to load and validate `config.yaml` using `pydantic` or `ruamel.yaml`.
    * Expose global constants `ROOT_DIR`, `RAW_DIR`, etc., from the `project:` block.

### 0.2 Raw Data Discovery & I/O

3. **File:** `preprocess/discovery.py`

    * Function `discover_sessions(dataset: str) -> List[Session]` that scans `data/raw/{dataset}` for `.csv/.h5` sensor files and returns named tuples with paths and labels.

4. **File:** `preprocess/io.py`

    * Implement `load_pd_csv(path: str) -> np.ndarray` and `load_pd_hdf5(path: str) -> np.ndarray`.
    * Ensure these return a 1D NumPy array of the raw time-series.

### 0.3 Signal Cleaning & Windowing

5. **File:** `preprocess/cleaning.py`
    * Make sure there is detail docstring for each function, and docstring at top of function
    * For each of the technique below, save the cleaned signal to `data/processed/{clean_type}/{dataset}/{window_id}.npy`:
    * Do a unit test about the cleaning functions to ensure they work as expected.
    * Do a tutorial in a Jupyter notebook for all the cleaning type, to show how to use the cleaning functions.
    * Implement `bandpass_filter(x: np.ndarray, low: float, high: float, fs: float) -> np.ndarray`.
    * If `config.preprocessing_options.advanced_denoise == True`, call either:

        * `vmd_denoise(x, ...)` (via `vmdpy`), or
        * `ewt_denoise(x, ...)` (via `pywt`).

6. **File:** `preprocess/augmentation.py`

    * If `config.preprocessing_options.augment == True`, implement `time_warp(x)`, `add_jitter(x)`, etc., using `tsaug`.

7. **File:** `preprocess/windowing.py`

    * Implement `segment_signal(x, window_ms: float, fs: float) -> List[np.ndarray]` that cuts `x` into non-overlapping windows.
    * Attach labels by reading `label_file` and generating a corresponding list of window‐level labels.

8. **File:** `preprocess/run_preprocess.py`

    * Orchestrate: for each session, load → clean → augment → window → save each window to

      ```
      data/processed/{dataset}/{sensor}/{window_id}.npy
      ```
    * If clean file already exist skip, but respect the `--force` flag to overwrite existing files.

---

## 🟡 Phase 1 – Feature Engineering & Expansion (`--stage extract`)

### 1.1 Feature Catalog Loader

9. **File:** `features/catalog.py`

    * Read the yaml for list of features
    * and produce a dict  `{ feature_name: extractor_fn }`.
    * Support toggling an entire block off when `enable_all: false`.

### 1.2 Base Feature Extractors

10. **File:** `features/extractors.py`

    * By default we will use mne_features to extract the features
      *     List of features available via mne_features:
        * pow_freq_bands
        * spect_edge_freq
        * energy_freq_bands
        * wavelet_coef_energy
        * kurtosis
        * skewness
        * line_length
        * mean
        * ptp_amp
        * quantile
        * rms
        * std
        * variance
        * spect_slope
        * zero_crossings
        * higuchi_fd
        * hjorth_complexity
        * hjorth_complexity_spect
        * hjorth_mobility
        * hjorth_mobility_spect
        * teager_kaiser_energy
        * phase_lock_val
        * spect_corr
        * time_corr
        * decorr_time
        * katz_fd
        * samp_entropy   # can cause memory issue, so we can disable it in config.
        * svd_entropy
        * svd_fisher_info
        * max_cross_corr
        * spect_entropy
    * But we can also use other libraries like librosa, tsfresh, etc.But avoid duplicating features.
    * If dedicated library is not available, then implement custom  function per feature:

        * `compute_time_skewness(window: np.ndarray) -> float`
        * `compute_spectral_centroid(window: np.ndarray, fs: float) -> float`
        * …and so on for kurtosis, RMS, entropy, MFCC, wavelet energies, etc.
    * Write unit tests to verify expected ranges on synthetic signals.
    * Eventhough come from different package or custome,but seperate the features based on the theme and library.
    
    ```yaml
            ├── 📁 2_feature_engineering/
        │   │   │   ├── 📁 classic_stats/                # RMS, kurtosis…
        │   │   │   ├── 📁 time_frequency/               # FFT bands, STFT
        │   │   │   ├── 📁 wavelet_cwt/                  
        │   │   │   ├── 📁 entropy_fractal/              
        │   │   │   ├── 📁 dfs_featuretools/            
        │   │   │   └── 📁 image_representations/       
        │   │   │
    ```
    * List the features from each library or custom library in the config.yaml. For example
    ```yaml
      mne_features:
          freq_bands:
          delta: [0.5, 4.5]
          theta: [4.5, 8.5]
          alpha: [8.5, 11.5]
          sigma: [11.5, 15.5]
          beta:  [15.5, 30.0]
          selected_features:
            - pow_freq_bands
            - spect_edge_freq
            - energy_freq_bands
            - wavelet_coef_energy
            - kurtosis
            - skewness
            - line_length
      librosa:
          selected_features:
            - spectral_centroid
            - spectral_bandwidth
            - spectral_contrast
            - mfcc
    ```

### 1.3 Feature Expansion

11. **File:** `features/expansion.py`

    * Function `pairwise_combine(df: pd.DataFrame) -> pd.DataFrame` that adds columns for `f_i + f_j`, `f_i * f_j`, `|f_i-f_j|`, `f_i/(f_j+ε)`.
    * Function `polynomial_interactions(df: pd.DataFrame) -> pd.DataFrame` that uses `sklearn.preprocessing.PolynomialFeatures(interaction_only=True)`.
    * If `config.features.enable_autofeat: true`, wrap `AutoFeatRegressor` to generate additional features.

### 1.4 Master Feature Table Assembly

12. **File:** `features/build_master.py`

    * Scan `data/processed/.../*.npy`, load windows, apply extractors and expansions.
    * Concatenate results into a single `pd.DataFrame` with columns:

      ```
      ['dataset', 'window_id', 'label', feat_1, feat_2, …]
      ```
    * Write to disk as Parquet:

      ```
      outputs/features/features_master.parquet
      ```
    * Honor `--force` for overwrite.

---

## 🟢 Phase 2 – Feature Selection & Model Training (`--stage analyze`)

### 2.1 Experiment Matrix Generator

13. **File:** `experiments/matrix.py`

    * Read lists under `datasets`, `preprocessing_options`, `scaling_methods`, `feature_selections`, `model_types`.
    * If `experiments.generator == cartesian_product`, produce a DataFrame with one row per combination.

### 2.2 Feature Selection Wrappers

14. **File:** `experiments/selection.py`

    * For each method:

        * `select_null(X, y) -> X`
        * `select_featurewiz(X, y) -> X_sub`
        * `select_boruta(X, y) -> X_sub` (via `BorutaPy`)
        * `select_rfe(X, y) -> X_sub` (via `sklearn.feature_selection.RFE`)
        * `select_mrmr(X, y) -> X_sub` (via `pymrmr`)

### 2.3 Model Zoo & Hyperparameter Configuration

15. **File:** `models/zoo.py`

    * Map `config.model_types` strings to instantiated sklearn/CatBoost/XGB objects and their default param grids.
    * Provide a function `get_model_and_grid(name: str)`.

16. **File:** `experiments/hpo.py`

    * If `config.hyperparameter_tuning.tool == 'optuna'`, set up `OptunaSearchCV` wrappers for each model and grid.
    * Else, use `sklearn.model_selection.GridSearchCV`.

### 2.4 Training Loop & CV

17. **File:** `experiments/train.py`

    * For each row in the experiment matrix:

        1. Load master features DataFrame → `X, y`.
        2. Apply `scaling_method` (via a `Pipeline` with scaler + dummy if `none`).
        3. Apply feature selection method to get `X_sel`.
        4. Instantiate model + HPO object.
        5. Fit with `cv=StratifiedKFold(config.cv_strategy.cv_folds)`.
        6. Record best params, CV scores (accuracy, recall, F1).
        7. Save best estimator to:

           ```
           outputs/models/{run_id}.joblib
           ```
        8. Append metrics to a running list.

18. **File:** `experiments/save_results.py`

    * After loop, write all metrics to:

      ```
      outputs/reports/summary_metrics.csv
      ```
    * If `config.runtime.save_confusion_matrices`, dump each model’s confusion matrix as an image under:

      ```
      outputs/reports/confusion_{run_id}.png
      ```
    * Similarly for ROC curves if enabled.

---

## 🟣 Phase 3 – Reporting & Export (`--stage report`)

### 3.1 Explainability & Plots

19. **File:** `reporting/plots.py`

    * Functions to generate:

        * Confusion matrix plot (`sklearn.metrics.ConfusionMatrixDisplay`)
        * ROC curve (`sklearn.metrics.RocCurveDisplay`)
        * SHAP summary & waterfall plots for top model (using `shap.TreeExplainer`)

20. **File:** `reporting/generate_report.py`

    * Read `summary_metrics.csv` and gather plot paths.
    * Render a Markdown template (e.g., using Jinja2) with sections for:

        1. **Overview** (config snapshot, git commit)
        2. **Top Models** (table of best runs)
        3. **Feature Importance & SHAP**
        4. **Performance Plots**
    * Write out:

      ```
      outputs/reports/final_report.md
      ```
    * (Optional) Convert to PDF via `pandoc` if required.

### 3.2 Model Export & Drift Stub

21. **File:** `deployment/export.py`

    * For each best `.joblib`, convert to ONNX:

      ```python
      from skl2onnx import convert_sklearn
      ```
    * Save under:

      ```
      outputs/models/{run_id}.onnx
      ```

22. **File:** `deployment/drift_monitor.py`

    * Stub out a drift detection script using `evidently`:

        * Load recent feature streams, compare distributions to training baseline.
        * If drift > threshold, emit a warning log or alert file.

---

### 🔧 Final Touches

* **Testing:**

    * Add unit tests under `tests/` for each module: I/O, feature extractors, selection methods, and the training loop.
    * Use small synthetic signals to verify feature values and pipeline integration.

* **CI Integration:**

    * Ensure a GitHub Actions workflow runs linting, type‐checks, and tests on each PR.
    * Optionally, run a smoke test of a single `full-run --jobs 1 --dataset iris` to sanity-check the end-to-end flow.

* **Documentation:**

    * Write a `README.md` that outlines:

        * How to install dependencies (`requirements.txt`)
        * CLI usage examples for each stage
        * Directory structure and config options
    * Embed the modular flowchart (ASCII or image) into the docs.
