# 📂 Folder Structure
How the folder looks like
```plaintext
dataset/
├── contactless_pd_detection/                   # Main dataset folder
│   ├── station_52009/                          # Station ID
│   │   ├── 748987.npy                          # Raw signal (20 ms , about 800000 samples)
│   │   ├── 748988.npy
│   │   ├── ...
│   ├── station_52008/
│   │   ├── ...
│   ├── station_52007/
│   │   ├── ...
│   ├── ...

```
```plaintext

📁 partial_discharge_project/
│
├── 📁 station_52009/                           # ➊ One folder per measurement station
│   │
│   ├── 📁 data_clean/                      # ──── 1  Cleaning & Denoising
│   │   ├── 📁 standard_denoising_normalisation/
│   │   ├── 📁 advanced_denoising/
│   │   │   ├── 📁 VMD/
│   │   │   └── 📁 EWT/
│   │   ├── 📁 synthetic_augmentation/
│   │   │   ├── 📁 tsaug/
│   │   │   ├── 📁 jitter/
│   │   │   └── 📁 PRPD_simulations/
│   │   └── 📁 outlier_detection/
│   │       └── 📁 PyOD/
│   │
│   ├── 📁 features/                        # ➋ 2 – 4  Features & Selection
│   │   │
│   │   ├── 📁 2_feature_engineering/
│   │   │   ├── 📁 classic_stats/                # RMS, kurtosis…
│   │   │   ├── 📁 time_frequency/               # FFT bands, STFT
│   │   │   ├── 📁 wavelet_cwt/                  
│   │   │   ├── 📁 entropy_fractal/              
│   │   │   ├── 📁 dfs_featuretools/             
│   │   │   └── 📁 image_representations/        
│   │   │
│   │   ├── 📁 3_feature_comb_expansion/
│   │   │   ├── 📁 pairwise_math_ops/
│   │   │   ├── 📁 polynomial_features/
│   │   │   ├── 📁 mathematical_combination/
│   │   │   ├── 📁 autofeat/                     # (optional)
│   │   │   └── 📁 symbolic_regression/          # (optional)
│   │   │
│   │   └── 📁 4_feature_selection/
│   │       ├── 📁 tracks/
│   │       │   ├── 📁 baseline_all_feats/
│   │       │   ├── 📁 featurewiz_corr_xgb/
│   │       │   └── 📁 mljar_internal/
│   │       └── 📁 optuna_subset_optim/          # (optional)
│   │
│   ├── 📁 models/                          # ➌ 5  Model Training & Tuning
│   │   ├── 📁 baseline_models/                 # default hyper-params
│   │   ├── 📁 tuned_gridsearch/                # GridSearchCV artefacts
│   │   ├── 📁 tuned_optuna/                    # (optional)
│   │   ├── 📁 ensembles/                       # stacking / blending
│   │   └── 📁 incremental_stream/              # (optional, river)
│   │
│   ├── 📁 reports/                         # optional: metrics, plots, notebooks
│   │
│   └── 📁 drive_mirror/ (optional)         # Google Drive mount
│       └── [replicates the same sub-tree]
│
├── 📁 station_52008/
│   └── [identical structure as station_52009]
├── 📁 station_52007/
│   └── [identical structure as station_52009]
```
### 📈 Pipeline Flowchart
                                ▶ PARTIAL DISCHARGE (PD) CLASSIFICATION PIPELINE ◀
                     ────────────────────────────────────────────────────────────────────────────
      1️⃣  DATA INGESTION & PRE-PROCESSING
      ────────────────────────────────────────────────────────────────────────────────────────────
      • Load raw time-series PD signals (npy)
      • Standard denoising, normalisation
      • ⚙️  Advanced denoising (VMD / EWT) ..................... (optional)
      • ⚙️  Synthetic augmentation (tsaug / jitter / PRPD sims)  (optional)
      • ⚙️  Outlier / novelty detection (PyOD) ................. (optional)
                                      │
                                      ▼
      2️⃣  FEATURE ENGINEERING
      ────────────────────────────────────────────────────────────────────────────────────────────
      • Classic stats (RMS, kurtosis, skew, crest-factor)
      • Time–frequency summaries (FFT bands, STFT)               │
      • ⚙️  Wavelet-based features (CWT, energy) ............... (optional)
      • ⚙️  Multiscale entropy & fractal dimension ............. (optional)
      • ⚙️  Featuretools Deep Feature Synthesis ................ (optional)
      • ⚙️  Image representations → tiny 2-D CNN ............... (optional, CPU)
                                      │
                                      ▼
      3️⃣  FEATURE COMBINATION & EXPANSION
      ────────────────────────────────────────────────────────────────────────────────────────────
      • Pairwise math ops  (+, −, ×, ÷, |Δ|)                     │
      • PolynomialFeatures (interaction-only)                    │
      • feature-engine MathematicalCombination                   │
      • ⚙️  AutoFeat nonlinear search .......................... (optional)
      • ⚙️  Symbolic regression (PySR / gplearn) ............... (optional)
                                      │
                                      ▼
      4️⃣  FEATURE SELECTION TRACKS
      ────────────────────────────────────────────────────────────────────────────────────────────
      ┌───────────────┬─────────────────────┬─────────────────────┐
      │ 4A Baseline   │ 4B Featurewiz       │ 4C MLJAR-supervised │
      │ (all feats)   │ (corr-prune+XGB)    │ internal selector   │
      └───────────────┴─────────────────────┴─────────────────────┘
      • ⚙️  Optuna Feature-Subset Optimiser ..................... (optional)
                                      │
                                      ▼
      5️⃣  MODEL TRAINING & HYPER-TUNING
      ────────────────────────────────────────────────────────────────────────────────────────────
      • Ridge ▪ LogReg ▪ LDA/QDA ▪ SVM-linear ▪ KNN
      • DT ▪ RF ▪ ExtraTrees ▪ GBDT ▪ Ada ▪ XGB ▪ LGBM ▪ CatBoost
      • GridSearchCV + Stratified K-fold
      • ⚙️  Optuna / Hyperband Bayesian search ................. (optional)
      • ⚙️  Ensemble stacking / blending ........................ (optional)
      • ⚙️  Incremental (river) for streaming .................. (optional)
                                      │
                                      ▼
      6️⃣  EVALUATION & EXPLAINABILITY
      ────────────────────────────────────────────────────────────────────────────────────────────
      • Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
      • Confusion matrix, feature importance charts
      • ⚙️  Class-imbalance metrics (balanced acc, MCC) ........ (optional)
      • ⚙️  SHAP / LIME global & local explanations ............ (optional)
                                      │
                                      ▼
      7️⃣  REPORTING & EXPERIMENT MANAGEMENT
      ────────────────────────────────────────────────────────────────────────────────────────────
      • CSV & Markdown summaries per experiment
      • Models saved with joblib
      • ⚙️  MLflow / W&B run tracking .......................... (optional)
      • ⚙️  DVC / Git-LFS data & model versioning .............. (optional)
                                      │
                                      ▼
      8️⃣  DEPLOYMENT & MONITORING
      ────────────────────────────────────────────────────────────────────────────────────────────
      • Export best model → ONNX ➜ CPU inference
      • ⚙️  Drift monitoring (Evidently) ....................... (optional)
      • ⚙️  Online retraining triggers ........................ (optional)
                     ────────────────────────────────────────────────────────────────────────────


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

### **Key CLI Flags at a Glance**

| Flag                                                               | Purpose                                                  |
| ------------------------------------------------------------------ | -------------------------------------------------------- |
| `--stage {full-run \| preprocess \| extract \| analyze \| report}` | Selects which phase(s) to run.                           |
| `--force`                                                          | Overwrite existing master feature file during Phase 1.   |
| `--jobs N`                                                         | Parallel workers for heavy loops (feature calc, CV).     |
| `--advanced-denoise`                                               | Enables VMD/EWT denoising during Phase 0.                |
| `--augment`                                                        | Adds synthetic time-series augmentation in Phase 0.      |
| `--wavelet-feats`                                                  | Adds wavelet & entropy features in Phase 1.              |
| `--optuna`                                                         | Replaces GridSearch with Optuna Bayesian HPO in Phase 2. |


### Quick Start

Execute the preprocessing stage with:

```bash
python main.py --stage preprocess --advanced-denoise --augment
```

The script reads datasets and parameters from `config.yaml` and dispatches
`preprocess/run_preprocess.py`. Cleaned windows are saved to
`data/processed/<dataset>/<sensor>/`.



---

## 🧪 **Experiment Pipelines (Standardized 4-Step Structure)**

---

### 🅀 Baseline (Exp 0) — *All features, no combination/selection*

#### 1️⃣ DATA INGESTION & PRE-PROCESSING

* Load raw time-series PD signals (`.npy`)
* Standard denoising & normalization

#### 2️⃣ FEATURE ENGINEERING

* Classic stats (RMS, kurtosis, skew, crest-factor)
* Time–frequency (FFT bands, STFT)
* Wavelet-based (CWT, wavelet energy)
* Multiscale entropy, fractal dimension
* Featuretools Deep Feature Synthesis
* Image representations → scalograms

#### 3️⃣ FEATURE COMBINATION & EXPANSION

* *None* (raw features only)

#### 4️⃣ FEATURE SELECTION TRACKS

* **4A Baseline** (all features passed to classifier directly)

---

### 🅁 Featurewiz Track (Exp 1)

#### 1️⃣ DATA INGESTION & PRE-PROCESSING

* Load `.npy`, standard denoising & normalization

#### 2️⃣ FEATURE ENGINEERING

* All standard + optional features (see baseline)

#### 3️⃣ FEATURE COMBINATION & EXPANSION

* Pairwise math ops (+, −, ×, ÷, |Δ|)
* `PolynomialFeatures` (interaction-only)
* `feature-engine` MathematicalCombination

#### 4️⃣ FEATURE SELECTION TRACKS

* **4B Featurewiz** (correlation pruning + XGBoost selector)

---

### 🅂 MLJAR-Supervised (Exp 2)

#### 1️⃣ DATA INGESTION & PRE-PROCESSING

* Load `.npy`, standard denoising & normalization

#### 2️⃣ FEATURE ENGINEERING

* All features enabled

#### 3️⃣ FEATURE COMBINATION & EXPANSION

* Pairwise ops + PolynomialFeatures + feature-engine MC

#### 4️⃣ FEATURE SELECTION TRACKS

* **4C MLJAR-supervised** (internal selector + AutoML leaderboard)

---

### 🅃 Advanced Denoising + BorutaShap (Exp 3)

#### 1️⃣ DATA INGESTION & PRE-PROCESSING

* Load `.npy`, **VMD denoising**

#### 2️⃣ FEATURE ENGINEERING

* All features enabled

#### 3️⃣ FEATURE COMBINATION & EXPANSION

* Pairwise ops + PolynomialFeatures + `AutoFeat` nonlinear features

#### 4️⃣ FEATURE SELECTION TRACKS

* BorutaShap selector → XGBoost

---

### 🅄 Data Augmentation + CatBoost (Exp 4)

#### 1️⃣ DATA INGESTION & PRE-PROCESSING

* Load `.npy`, standard denoising
* Synthetic augmentation: `tsaug`, jitter, PRPD simulations

#### 2️⃣ FEATURE ENGINEERING

* All features enabled

#### 3️⃣ FEATURE COMBINATION & EXPANSION

* Pairwise ops + PolynomialFeatures + feature-engine MC

#### 4️⃣ FEATURE SELECTION TRACKS

* **4B Featurewiz** → CatBoost (class-balanced loss)

---

### 🅆 Wavelet Image CNN (Exp 5)

#### 1️⃣ DATA INGESTION & PRE-PROCESSING

* Load `.npy`, standard denoising

#### 2️⃣ FEATURE ENGINEERING

* Image representations (scalograms, STFT slices)
* Optional: all others computed for auxiliary use

#### 3️⃣ FEATURE COMBINATION & EXPANSION

* *Not applicable* (CNN model consumes image input)

#### 4️⃣ FEATURE SELECTION TRACKS

* CNN training (end-to-end) → Softmax prediction

---

### 🅇 EWT + Symbolic Regression (Exp 6)

#### 1️⃣ DATA INGESTION & PRE-PROCESSING

* Load `.npy`, **EWT denoising**

#### 2️⃣ FEATURE ENGINEERING

* All features enabled (EWT band energy emphasis)

#### 3️⃣ FEATURE COMBINATION & EXPANSION

* Pairwise ops + PolynomialFeatures + `feature-engine`
* **Symbolic regression** via PySR (closed-form synthesis)

#### 4️⃣ FEATURE SELECTION TRACKS

* PySR-selected equations → LightGBM

---

### 🅈 UMAP + Instance-Based Learner (Exp 7)

#### 1️⃣ DATA INGESTION & PRE-PROCESSING

* Load `.npy`, standard denoising

#### 2️⃣ FEATURE ENGINEERING

* All features enabled

#### 3️⃣ FEATURE COMBINATION & EXPANSION

* Pairwise ops + PolynomialFeatures + MC
* Dimensionality reduction: UMAP (3D manifold)

#### 4️⃣ FEATURE SELECTION TRACKS

* Cluster-based learner (k-NN / DBSCAN) → Majority vote


# Partial Discharge Classification Pipeline for High-Voltage Cable Diagnostics

**Introduction:** Partial discharges (PD) are tiny electrical sparks that occur in weakened insulation of high-voltage cables and equipment. They are often the **earliest warning** of insulation failure, so detecting and classifying PD activity is critical for preventing cable breakdown. A PD classification pipeline typically involves several stages, from capturing noisy sensor data to producing a deployable model. Each stage of the pipeline is designed to **extract meaningful information from PD signals and ensure reliable early fault detection**. Below, we explain each stage in detail, focusing on high-voltage cable monitoring and why each step is important for diagnostics.

## 1. Data Ingestion and Pre-processing

&#x20;*Figure 1: A partial discharge measurement schematic (IEC 60270 method) where a 1000 pF coupling capacitor and an LDM-5 measuring impedance detect PD pulses from the high-voltage “Test object,” with a 15 mH inductor filtering out high-frequency noise. A UHF sensor (antenna) coupled via coaxial cable to a signal conditioning system and oscilloscope captures radiated PD signals for analysis.*

**PD Data Collection:** In high-voltage cable monitoring, PD data typically come from **electrical sensors** that capture fast current or voltage pulses caused by discharges. For example, a coupling capacitor or high-frequency current transformer (HFCT) clamped on the cable can pick up PD pulses per the IEC 60270 standard. Some systems also use **UHF or acoustic sensors** to detect electromagnetic or sound emissions from PD events (especially in enclosed equipment). These raw signals are essentially time-series pulses often buried in noise.

**Denoising:** Raw PD signals are usually very weak and accompanied by substantial electrical interference and background noise. Pre-processing applies filters or signal processing (e.g. band-pass filtering, wavelet denoising) to improve the signal-to-noise ratio. **Noise reduction is crucial** for accurate and reliable PD detection – a cleaner signal ensures that true PD pulses aren’t masked and false alarms (noise spikes) are minimized. In high-voltage environments (substations, industrial sites), multiple PD sources or external RF interference can overlap, so denoising and maybe pulse separation are necessary to isolate meaningful PD events. Early-stage denoising helps ensure that subsequent analysis focuses on real insulation discharge activity, which is vital for early fault diagnosis.

**Normalization:** After cleaning, the PD signal is typically normalized to a consistent scale. For instance, one can divide the entire waveform by its maximum amplitude so that the signal values range between –1 and 1. This prevents large amplitude differences from dominating the learning process and allows **comparison across sensors or tests**. Normalizing is important in PD data because different cables or sensors might record different absolute magnitudes of discharge; scaling them puts all data on equal footing for feature extraction and model training.

**Segmentation and Splitting:** The continuous data stream is then segmented into samples (for example, one sample could be a fixed time window capturing one or multiple PD pulses, or a phase window in AC cycles). These samples are labeled (e.g. discharge type or “no PD” vs “PD”) if known. The dataset is **split into training and test sets** (often 80%/20% or similar) to allow unbiased evaluation. Sometimes cross-validation (k-fold) is used if the dataset is small. Splitting ensures the model is evaluated on unseen data, which is critical for verifying that the PD classifier will generalize to new cables and conditions. In a safety-critical domain like cable monitoring, we must be confident the model will detect PD patterns it wasn’t explicitly trained on.

> *Why this stage matters:* Careful ingestion and pre-processing of PD data lays the groundwork for the entire pipeline. High-voltage cable PD signals can be extremely subtle and noisy; without denoising and normalization, a classifier might learn noise instead of real PD characteristics. Likewise, proper data splitting means we can trust the performance estimates – we don’t want a model that only works on historical data but fails to alert on a new developing cable fault. In short, this stage ensures we **capture true PD indicators** and not artifacts, making early detection feasible.

## 2. Feature Engineering (Statistical, Temporal, Spectral Features)

Once we have clean, normalized PD waveforms, the next step is to extract informative **features**. Rather than feeding raw waveforms into a classifier (which can be high-dimensional and hard to interpret), we compute descriptive metrics that capture the essence of the PD events. Traditionally, PD diagnostics rely on features derived from **time-domain and frequency-domain analysis** of the signals. Below are key categories of features and their role in PD analysis:

* **Statistical Features:** Basic stats describe the distribution of PD measurements. Examples include mean and standard deviation of pulse amplitude, pulse count over a period, pulse repetition rate, or higher-order moments like skewness and kurtosis of the amplitude distribution. These features summarize the overall PD activity level and variability. For instance, a high variance or a heavy-tailed amplitude distribution might indicate sporadic but strong discharges. Statistical features are simple but provide a baseline fingerprint of the discharge behavior.

* **Time-Domain Features:** These capture characteristics of the PD pulses in the time signal. Important time-domain features include the **peak value** of the PD pulse, **rise time** (pulse steepness), **fall time** or pulse width, pulse **energy** (e.g. area under the pulse waveform), and **root mean square (RMS)** value of the signal. Also, features like the **average amplitude** or **pulse count per cycle** are used. For example, a particular defect type might produce many small pulses vs. a few large pulses. Time-domain features relate directly to physical discharge properties (like how quickly the discharge extinguishes) and are thus very useful for distinguishing PD types. In high-voltage cables, internal void discharges might show different pulse shapes than surface discharges; time-domain parameters can reflect these differences.

* **Frequency-Domain Features:** PD pulses contain high-frequency components, so transforming the signal (e.g. via Fast Fourier Transform or wavelet analysis) can reveal frequency content. Features here might include the **dominant frequency** or frequency band where most energy resides, the **bandwidth** or spectrum spread of the discharge, or spectral entropy. Often, **discrete wavelet transform (DWT)** is used to decompose PD signals into frequency bands. For example, corona discharges may emit more high-frequency energy than internal discharges. By capturing spectral features, we utilize the fact that different PD mechanisms have characteristic frequency signatures. In practice, one might compute the power in specific bands (e.g.  HF, VHF ranges) or identify spectral peaks that correlate with certain defects.

* **Entropy-Based Features:** Entropy measures the complexity or unpredictability of the signal. **Shannon entropy** or more specialized metrics like *permutation entropy* and *dispersion entropy* have been applied to PD signals. These features quantify how “random” or regular the discharge patterns are. A periodic or regular PD pattern (low entropy) might indicate a stable repetitive defect, whereas a highly irregular pattern (high entropy) could suggest sporadic discharges or heavy noise. Entropy features are especially useful to differentiate true PD from random noise – true partial discharges often follow physical patterns (e.g. phase-related firing) whereas pure noise might be more random. Indeed, researchers have found that combining multiple entropy measures can improve PD classification robustness.

* **Domain-Specific Features (Phase-Resolved PD):** High-voltage AC systems often use **phase-resolved partial discharge (PRPD) patterns** – essentially mapping each discharge pulse to the phase angle of the AC cycle when it occurred. Domain knowledge tells us that different insulation defects fire at characteristic phase angles and with different polarity preferences. For example, **internal void PD** might occur in both half-cycles, whereas **surface discharges** often occur only in one polarity (due to one electrode being grounded), and **corona** (air discharge) might have a distinctive phase clustering. We can engineer features like the **average phase angle of discharges**, the **phase spread**, or counts of pulses in specific phase windows. Each PD source yields a distinct PRPD pattern, which can be recognized by appropriate features. Other domain-driven features include the apparent charge (in picocoulombs, from calibration), or the time *interval* between successive discharges (which might relate to how quickly the local charge re-accumulates). By incorporating these domain logic features, we infuse the model with expert knowledge about PD behavior in cables – for instance, a model might learn that discharges occurring only near the peak voltage of each cycle (phase \~90°/270°) are indicative of a certain type of defect.

> **Example:** In practice, a comprehensive feature set for a cable PD might include: max amplitude of each PD pulse, rise/fall times, number of pulses per AC cycle, mean phase of occurrence, standard deviation of phase, total energy in 0.1–1 MHz band (to detect ultrasonic EMI), Shannon entropy of the signal segment, etc. Researchers have studied many such features – statistical, texture (for PRPD images), fractal dimension of patterns, etc. – all with the goal of distinguishing PD sources.

Feature engineering is often guided by the physics of PD and the need to reduce data size. Instead of raw waveforms with thousands of sample points, we end up with a vector of, say, tens of features per sample. This makes the classification problem tractable and often more interpretable. It’s worth noting that extracting too many features can backfire (the “curse of dimensionality”), which is addressed in later steps (feature selection). But at this stage, we aim to be **inclusive** – capture anything that might help separate classes, so that no important information is left behind.

> *Why this stage matters:* Good features act as **symptom descriptors** of insulation problems. High-voltage engineers can often infer a lot from PD waveform traits (like “a slow rise-time, low magnitude pulse likely indicates a surface discharge”). By quantifying these traits, we allow a machine learning model to make similar inferences. For early failure detection, it’s crucial that features correlate with physical degradation – e.g. increasing PD counts or energy might precede breakdown. Thus, feature engineering translates raw sensor data into **insightful indicators** (pulse counts, frequencies, etc.) that can be tracked and classified. In summary, this stage distills the complex PD signal into **meaningful parameters** that reflect the health of the cable insulation.

## 3. Feature Combination and Expansion

After computing an initial set of features, we may **expand the feature space** by creating new features from combinations or transformations of the original ones. The motivation is that sometimes relationships between features carry important information. For example, the *ratio* of two features might separate classes better than either feature alone. In PD analysis, one might consider combining features like *pulse count* and *average amplitude* (e.g. a combined feature = count × amplitude could represent total activity). In fact, applying mathematical functions to existing features can generate a large number of new candidate features. Each PD pattern can potentially yield **thousands of features** once you include polynomial combinations and interactions.

**Why combine features?** Some classification boundaries are not linear in the original feature space. By adding nonlinear combinations (like products, ratios, squares), we allow linear models to fit more complex patterns. For instance, if two features individually are not distinctive, their combination might be. In high-voltage cable PD, maybe neither “pulse repetition rate” nor “pulse magnitude” alone classifies a defect, but the *energy* (rate × magnitude) might do so. Therefore, we systematically expand features to give the model the best chance at capturing such interactions.

**Automated Tools for Feature Expansion:** Manually deciding which features to combine can be tedious, so data scientists often use libraries and tools to automate this process:

* **Feature-engine (Python library):** This tool can generate interaction features (e.g. multiplying or adding features pairwise) and perform transformations on groups of features. It provides a pipeline to easily apply transformations like **binarizing, polynomial expansion, or group statistics** on features without manual coding. For example, feature-engine could add squares or cross-terms (f1 \* f2, f1^2, etc.) for every numeric feature.

* **Autofeat:** Autofeat is an automated feature engineering library that **creates non-linear features** from the original data and then prunes them. During its fit, Autofeat tries various mathematical transformations (polynomial terms, exponentials, logarithms of features, interactions between features) and evaluates which ones improve a simple model’s performance. Essentially, it can generate a huge pool of candidate features by applying formulas to existing ones, then use an internal selection (often L1-regularized regression) to keep only the useful ones. This is very helpful if, say, the product of two PD features or the square of a feature has predictive power – Autofeat will find it for you. By automating this, we ensure we don’t miss subtle relations (like maybe “(peak amplitude)^2” correlates with a particular defect severity).

* **Symbolic Regression & Genetic Programming:** These are techniques where algorithms try to **evolve mathematical expressions** that fit the target. In essence, they treat feature construction as a search problem: combining basic operations (+, –, ×, /, sin, etc.) on original features to form new ones that correlate with the class label. Symbolic regression can yield human-understandable formulas like `FeatureX = log(f1 * f2) + 3*f3`. In the PD context, this might uncover a formula like “if (pulse\_count \* sqrt(energy)) is above a threshold, classify as internal PD” – something not obvious a priori. This approach, while computationally heavy, can directly incorporate domain knowledge operations (like maybe absolute value for polarity). Some modern libraries use genetic programming to automate this feature discovery.

Using these tools, we **expand** the feature set dramatically. However, it’s important to note that while we generate many features in this stage, we will **not keep all of them** – the next stage will reduce the set. The idea is to cast a wide net now (even up to thousands of features) so that potentially valuable signals are present somewhere in the features. Redundant or useless ones will be thrown away later. Indeed, research has shown that applying “different mathematical functions” to generate combination features yields lots of features, but using all of them would make the model inefficient; hence subsequent **feature selection** is needed to maintain recognition efficiency.

> *Why this stage matters:* Feature combination increases the expressive power of our feature space. In high-voltage monitoring, the relationships between raw features might encode physical laws or empirical patterns (like a defect might cause *proportionally* more PD at higher voltages – a multiplicative effect). By including combined features, we enable simpler models (like linear or low-depth trees) to capture complex patterns. For early fault detection, this could mean capturing subtle precursors: maybe only when *several* simple indicators rise together does it signify a serious issue. Automated expansion ensures these joint indicators are available for the model to consider. Of course, it also increases the risk of overfitting and adds noise features, which is why the **next step (feature selection)** is critical.

## 4. Feature Selection (Reducing and Choosing the Best Features)

After engineering potentially dozens or hundreds of features, we need to **prune** the feature set to retain only the most informative, non-redundant ones. Feeding too many features into a classifier can cause overfitting (especially if some features are noisy or irrelevant) and can slow down the training. In PD classification, many features might be correlated with each other (for example, peak value and energy are related, or count and phase distribution might overlap in information). Feature selection addresses this by **identifying a subset of features that gives the best classification performance**.

**Baseline (Manual) Feature Selection:** A baseline approach is to use domain knowledge or simple statistical tests to drop useless features. For instance, we can remove features that have near-zero variance (constant in all samples) or features that are highly correlated with others (keeping only one of a correlated group). One might also manually pick features that historically mattered (e.g. known PD indicators like phase angle or magnitude) and ignore exotic ones that seem less reliable. This approach is straightforward but might miss combinations that a human wouldn’t realize are important.

**Automated Feature Selection Techniques:** Several modern tools can help automate this process, often outperforming manual selection:

* **Featurewiz:** Featurewiz is a library designed for **advanced feature selection** on large datasets. It uses a two-step strategy: first, it applies the SULOV algorithm (“Selecting Uncorrelated Features”) to remove highly correlated features, then it uses a **recursive XGBoost** approach to find the most important features. In practice, featurewiz will train XGBoost models iteratively, each time eliminating features with low importance, until it finds a stable subset. This yields a smaller feature list that still has strong predictive power. Featurewiz’s automated approach can handle dozens or hundreds of features and whittle them down to a critical few, which is extremely useful when we generated many polynomial features in the previous step. By removing redundant features, it not only prevents overfitting but also makes the model faster and more interpretable.

* **MLJAR-Supervised (AutoML):** The *mljar-supervised* AutoML framework includes built-in feature selection steps. Notably, it employs a clever **“random feature” test**: it adds a fake random noise feature to the data and checks feature importances against this noise. If a real feature is consistently less important than the random feature across many models, it’s considered not useful and is dropped. This is done in a cross-validated way to be robust. After dropping weak features, MLJAR retrains models on the selected subset. This two-step process (drop unimportant, then retrain) yields a lean feature set without sacrificing accuracy. The advantage is it’s automatic and model-agnostic (it uses permutation importance so it can work with any model type). In our context, mljar-supervised could systematically eliminate features that do not contribute to PD classification (perhaps some weird polynomial features we generated that turned out to be pure noise).

* **Other methods:** There are many algorithms like ReliefF, mRMR (Max Relevance, Min Redundancy), genetic algorithms, L1-regularization, etc., which can rank and select features. For instance, **mRMR** selects features that individually correlate with the class but are uncorrelated with each other – a good principle for PD features which ensures we cover different aspects of the PD phenomenon. In PD research, using mRMR combined with Random Forest was shown to drastically improve classification accuracy by focusing on the optimum features. Similarly, wrappers like recursive feature elimination (RFE) with a classifier can be used (the MDPI study used RFE with logistic regression to pick top features).

**Eliminating Redundancy:** A key goal is not just picking the top performers but ensuring they aren’t redundant. Often, an automated selector will be followed by a **correlation analysis** on the chosen features to verify they’re not too inter-correlated. For example, in one PD source separation study, after using RFE and mutual information to select features, the authors computed the correlation matrix and further eliminated features until the final set had minimal inter-correlation. This ensured the selected features each added unique information. In their case, they ended up with three features that were complementary and sufficient to separate the PD sources. By limiting to a small number (like 3) they also made the results easier to visualize and explain.

**Result of Feature Selection:** We end up with a “clean” feature set, maybe on the order of just a few to tens of features, down from potentially hundreds. These remaining features should: (a) have high predictive power for the PD classification task, and (b) not be too redundant with each other. Empirical evidence shows that using an optimal subset of features can **improve classification accuracy and efficiency**, while reducing model complexity. It also shortens training time and can reduce the risk of overfitting, because each feature included is there for a good reason.

> *Why this stage matters:* In high-voltage cable diagnostics, we want a **robust and parsimonious model** – one that relies on clear indicators of insulation problems, not a brittle combination of dozens of signals. Feature selection delivers just that: it strips away the noise and redundancy, leaving features that are strongly tied to PD behavior. This not only boosts accuracy (since unhelpful features aren’t distracting the model) but also improves **interpretability**. For example, if the final model uses only, say, “phase skewness” and “pulse energy” as features, an engineer can focus on those factors when assessing a cable. Moreover, a smaller feature set means fewer sensors or computations in a real deployment (perhaps we realize we only need the HFCT sensor and not the UHF antenna, if none of the UHF-derived features were selected). Ultimately, this stage ensures the model’s focus is on the **truly telling signs of insulation distress**, which is exactly what we need for early and reliable PD detection.

## 5. Model Training and Hyperparameter Optimization

With a curated feature set in hand, we proceed to train one or more **classification models** to actually distinguish between classes of interest (e.g. “normal vs PD” or different types of PD defects). In our PD pipeline, we typically try a variety of classification algorithms to see which works best for the data. Each algorithm may have hyperparameters that need tuning for optimal performance.

**Classifiers Used:**

* **Ridge Classifier (Logistic Regression with L2 regularization):** This is a linear model that finds a weighted combination of features to separate classes, with an L2 penalty to prevent overfitting. Ridge is good as a baseline because it’s simple and tends to handle collinear features gracefully by shrinking coefficients. If the PD features have a roughly linear relationship to the class (for instance, a weighted sum of certain feature thresholds indicates a defect), ridge can capture that. It’s also very fast to train and not prone to overfit if regularization is tuned.

* **LDA (Linear Discriminant Analysis):** LDA is a probabilistic linear model that assumes each class’s features have a Gaussian distribution. It projects data into a space that maximizes class separability. LDA can perform well if those assumptions hold (or approximately hold). In PD classification, LDA might be effective when the classes form distinguishable clusters in feature space (e.g. one type of PD yields consistently higher mean amplitude and lower count than another). It also provides insight by showing which linear combination of features is the discriminant. However, LDA can struggle if feature distributions are very non-normal or if classes have very different variances.

* **Support Vector Machines (SVM):** SVMs are powerful for smaller datasets and can use kernel functions to handle non-linear boundaries. An SVM will find the optimal hyperplane (or higher-dimensional separation surface) that maximizes the margin between classes. For PD data, if a linear separation isn’t good, a kernel SVM (like RBF kernel) can model complex boundaries. SVMs often performed well in PD pattern recognition tasks historically. One consideration is that SVM requires careful tuning of the kernel parameters (like the kernel width and regularization C), and scaling of features (which we have done via normalization). SVM is robust to high-dimensional input, which suits a scenario where we perhaps kept many features.

* **Decision Trees & Random Forests:** Decision trees split the feature space into regions by simple rules (thresholds on features). They are very interpretable (one can see the rule path for a classification) which is nice for explainability. However, single trees can overfit. **Random Forests** (RF) combine many trees (each trained on random subsets of data and features) and average their predictions, improving generalization. In PD classification research, RFs have been popular and often yield high accuracy. They handle nonlinear relationships and feature interactions implicitly. For example, a tree might learn a rule like “IF (phase\_skewness > X) AND (pulse\_energy > Y) THEN class = internal\_PD”. Ensembles of such trees (RF) are usually quite effective, and they also provide feature importance metrics. Tree-based models can handle categorical features (if any) and don’t require feature scaling. The downside is they have hyperparameters like number of trees, tree depth, etc., and very large forests can be slow.

* **Boosting Methods:** Boosting algorithms (like **Gradient Boosting Machines, XGBoost, LightGBM, CatBoost**) build an ensemble of weak learners (usually shallow trees) in a stage-wise manner, each new tree correcting errors of the previous ensemble. Boosting often achieves state-of-the-art accuracy on structured data. For PD classification, a boosting model can capture subtle patterns by combining many rules. XGBoost, for instance, was cited as effectively identifying significant PD features in some studies. Boosted trees are a bit more prone to overfitting if not tuned, but with proper regularization, they offer great performance. They have a number of hyperparameters (tree depth, learning rate, number of trees, regularization terms) that we need to optimize.

* **Others:** We may also consider **k-Nearest Neighbors (kNN)** (which was used in some PD studies as well) if the feature scaling is meaningful, or even simple neural networks or Naive Bayes. However, the question specifically lists the above models. Notably, in recent times some have tried deep learning on raw PD data, but in our pipeline we are focusing on classic ML with feature engineering.

Often, we will train multiple models and compare their performance (perhaps via cross-validation or on a validation set). Each model type has strengths: e.g. SVM might shine with a clear margin in feature space, whereas Random Forest might excel if a few decision rules capture most cases. In a published comparison, for example, RF, SVM, and kNN were all tried on a PD dataset; RF with proper feature selection gave the best accuracy (\~99.9%).

**Hyperparameter Tuning:** To get the best out of each model, we must tune hyperparameters:

* **Grid Search:** This is the brute-force method where we define a grid of possible values for each hyperparameter and train/evaluate the model for every combination. For instance, for an SVM we might grid-search C (regularization strength) over {0.1, 1, 10} and gamma (kernel width) over {0.01, 0.1, 1}, etc. Grid search is exhaustive and guarantees we check all combos, but it **becomes expensive** as the grid size grows. With many hyperparameters or continuous ranges, grid search can be impractical. Still, for a small model and feature set, we often start with grid search on a coarse grid to get a sense of good regions.

* **Optuna (Bayesian Optimization):** Optuna is a modern hyperparameter optimization framework that uses smarter search strategies than brute force. It treats the tuning process as an optimization problem: evaluate some hyperparameter sets, then use an algorithm (like Tree-structured Parzen Estimator, a Bayesian method) to propose new sets focusing on promising areas of the search space. In practice, Optuna can find a near-optimal combination in far fewer trials than grid search by **prioritizing promising regions and pruning unpromising trials early**. For example, if it finds that a small learning rate is consistently better, it will focus around that and vary other parameters next. Optuna often outperforms grid search in efficiency, especially for complex models with many hyperparameters. We would use Optuna (or a similar approach like Random Search or Bayesian optimization) when the parameter space is large – e.g. tuning a gradient boosting model with 5+ parameters. The result is usually a set of hyperparameters that maximizes our chosen score (accuracy, F1, etc.) on validation data.

* **Comparison:** Grid search is easy to implement and parallelize, and it’s **interpretable** (we see exactly how each combination did). It ensures no combination is missed, which is fine for low-dimensional searches. Optuna (or Random search) is more **adaptive and efficient** – it may stumble on a very good configuration that a limited grid wouldn’t include, and it avoids wasting time on obviously bad combos. In many cases, Optuna or similar can achieve equal or better results with far fewer model training runs than grid search. This is valuable in PD classification if training each model is time-consuming (though with typically smaller datasets, training time is not huge – but if we do cross-validation for robust estimates, it multiplies the cost).

During tuning, we also must be mindful of **class imbalance** (if present) – for example, ensure that cross-validation folds maintain roughly the same class ratios or use stratified sampling, and consider using scoring metrics that account for imbalance (more on that in evaluation).

Ultimately, we select the model (and hyperparameters) that gives the best performance on validation data. We should be cautious to then test it on a truly held-out test set to confirm the performance, ensuring we didn’t overfit during model selection.

**Models in PD context:** In literature, **ensemble methods and SVMs** are quite popular for PD classification. Simpler linear models might underfit if the problem is complex (except in cases where feature selection distills it well). We may end up choosing, say, a Random Forest with 100 trees of depth 5, or an RBF SVM with C=10 and gamma=0.1, etc., depending on what the data demands.

> *Why this stage matters:* This is the heart of the pipeline – we are training the system that will **make decisions about cable health**. Choosing the right model and tuning it properly can be the difference between catching a developing insulation fault or missing it. A well-tuned model will accurately classify PD patterns it has seen *and* generalize to new patterns. In an early warning scenario, we prefer a model that perhaps is slightly conservative (few false negatives – i.e., it doesn’t miss real PD) while not panicking on noise. By exploring different algorithms, we ensure we’re using the method best suited to our feature characteristics. For example, if the relationship between features and PD type is linear, a simpler model is sufficient (and more interpretable); if it’s highly nonlinear, a tree ensemble or SVM might be needed. Hyperparameter optimization further squeezes maximum performance – a model like XGBoost can fail if not tuned (overfit or underfit), but when tuned, it could discover very subtle PD signatures. In sum, this stage produces the **predictive engine** of the pipeline, and careful training gives us a reliable classifier we can trust for monitoring high-voltage cables.

## 6. Evaluation and Explainability

Once we have a trained model (or models), we need to **evaluate** its performance rigorously and also be able to **explain** its decisions to stakeholders (engineers, safety managers, etc.). Evaluation tells us how well the model is doing in identifying PD conditions (and not raising false alarms), while explainability tools help us trust the model by illuminating which features are driving its predictions.

**Performance Metrics:** For a classification model, common metrics include accuracy, precision, recall, F1-score, etc.:

* **Accuracy:** the fraction of correct predictions. While accuracy is easy to understand, it can be **misleading in imbalanced scenarios**. In PD detection, for example, if 95% of time windows have no PD and 5% have PD, a model that always predicts “no PD” is 95% accurate but utterly fails its purpose of detecting discharges. So accuracy alone is not sufficient.

* **Precision and Recall:** Precision is the proportion of predicted PD events that were actually PD (how many false positives we have), and Recall (sensitivity) is the proportion of actual PD events that the model caught (how many false negatives we have). For early failure detection, *recall* is extremely important – missing a real PD (false negative) means a potential insulation defect goes unnoticed, which could be dangerous. *Precision* matters too because if the model raises too many false alarms, maintenance might start ignoring the warnings. There’s a trade-off: you can tune the model’s threshold to favor recall or precision depending on what’s critical (usually, missing a PD is worse than a false alarm, up to a point).

* **F1-Score:** the harmonic mean of precision and recall. This is a single metric that balances both, useful for imbalanced datasets. If our task is, say, multi-class classification of defect type, we might use a macro-average F1 across classes to ensure we’re performing well on each type, even if some are rarer. The F1 score is generally more informative than accuracy in imbalanced classification because it only gets high if both precision and recall are high. In a PD context, a high F1 means the model catches most PD instances and doesn’t cry wolf too often – a desirable scenario.

* **AUC (Area Under ROC) / Balanced Accuracy:** Sometimes we use these for imbalance as well. Balanced accuracy is the average of true positive rate and true negative rate, which is like accuracy adjusted for imbalance. AUC measures performance across all thresholds. If we had continuous outputs, we might look at AUC to evaluate how well the model ranks PD vs non-PD cases.

For evaluation, we would test the model on a **held-out test set** or through cross-validation results that were not seen in training or hyperparameter tuning. In PD research, it’s common to report accuracy, precision, recall for each class, etc. (like in the MDPI study, they reported average accuracy \~92% and per-class precision/recall). We aim for high recall for PD detection and decent precision. If class distribution was skewed, we make sure to highlight recall/F1 since a high accuracy could be meaningless.

**Confusion Matrix:** A confusion matrix is a table that shows how the model’s predictions are distributed against true labels. For example, in a binary case, it shows True Positives, True Negatives, False Positives, False Negatives. For multi-class, it’s an N×N matrix. We examine this to see **which mistakes the model is making**. In a PD classification scenario, a confusion matrix might reveal, for instance, that the model sometimes confuses corona vs surface discharges, but never confuses internal PD with no PD. This kind of insight is valuable: it tells us if certain classes are systematically problematic. We can then possibly address that by adding features or adjusting the model. Moreover, analyzing the False Positives (FP) vs False Negatives (FN) in the confusion matrix helps us judge error criticality. For example, are most errors false alarms or missed detections? In critical monitoring, we often prefer FP over FN (false alarm is better than missed event), but we still want to minimize both. The confusion matrix analysis might guide threshold tuning or cost-sensitive adjustments (if missing a PD is very costly, we’d adjust the decision threshold to reduce FN even if FP increase). It also helps validate that the model isn’t biased: e.g., if for a certain type of known defect (say “void discharge”) the recall is much lower, we might need more data or features for that scenario. In short, the confusion matrix provides a **detailed breakdown of performance**, beyond a single number, and highlights where the model excels or struggles.

**Explainability – SHAP values:** Engineers will want to know *why* the model says a particular cable has a PD or what drove a classification. **SHAP (SHapley Additive exPlanations)** is a popular tool that assigns each feature an importance value for a given prediction, based on Shapley values from cooperative game theory. In essence, SHAP tells us how each feature is contributing to the model’s output for an instance – whether it’s pushing the prediction towards a certain class or away from it, and by how much. SHAP values provide a consistent and theoretically sound way to explain model predictions.

For our PD classifier, using SHAP can yield insights like: *“For this sample classified as ‘internal PD’, the features ‘pulse count’ and ‘energy’ had the largest positive SHAP values pushing it towards that class, while ‘phase skewness’ pushed slightly against.”* This aligns with physical expectations perhaps (e.g., internal PD might indeed produce more pulses of higher energy). On a global level, we can use SHAP to see overall feature importance – which features have the biggest impact on the model’s decisions across the dataset. This might show, for example, that *phase-related features are dominating the decisions*, or that *entropy turned out to be highly informative*, etc. Such information is extremely useful for trust: if the model were focusing on some odd feature that doesn’t make sense physically, we’d be wary. If it focuses on known important features, we gain confidence.

Other explainability methods include **LIME (Local Interpretable Model-agnostic Explanations)** and simply looking at feature importance from tree models or coefficients from linear models. In a Random Forest or XGBoost, for instance, we can rank features by how much they reduce impurity or by permutation importance. Often, these simpler measures align with SHAP’s global importance. The MDPI study’s model, for example, identified features like “square root amplitude” and “energy” as key after selection, which one could double-check with importance scores.

**Insights from Explainability:** Using these tools on PD classification can unearth patterns: maybe it reveals that when the model predicts “dangerous PD”, it’s mostly because of a high repetition rate and high entropy – which an engineer would recognize as a chaotic severe discharge condition. If explainability showed something counter-intuitive, it might signal a data issue or an overfitting. For instance, if SHAP showed that a feature like “temperature reading” (if we had that as a feature) is driving predictions more than actual PD pulse features, we’d investigate if temperature is confounding the data (maybe all serious PD happened in summer in our data). We could then adjust the model or data to avoid such spurious correlations.

**Class Imbalance Handling:** It’s worth noting that if the classes were imbalanced, we might also have applied techniques like resampling (oversampling PD cases or undersampling normal) or class weighting during model training. The evaluation should reflect that – e.g., using F1 or balanced accuracy as mentioned. We must ensure our test metrics truly reflect real-field performance. For example, if in reality PD events are rare, we might focus on **precision-recall curves** rather than ROC, because ROC can be overly optimistic in highly imbalanced settings.

Finally, we document the evaluation: e.g., *“On a test set of 1000 samples, the model achieved 95% accuracy, with 90% recall and 85% precision for PD detection (F1 ≈ 0.87). The confusion matrix showed 5 missed PD events and 20 false alarms. Feature importance analysis indicated the top features were pulse count and phase skewness, aligning with known diagnostics.”* This kind of summary gives stakeholders a clear picture of performance and reassurance that the model is making sense.

> *Why this stage matters:* In high-voltage cable monitoring, **verification is critical** – we’re potentially making decisions about maintenance or shutdown based on this model. Evaluation metrics tell us if the model is ready for deployment or which aspects need improvement (if recall is low, we need to boost sensitivity, etc.). Explainability builds **trust**: engineers will not accept a black-box blindly when safety is at stake. But if we can show them, for instance, a SHAP plot that highlights high discharge counts and magnitudes leading to an alarm – things they intuitively agree are dangerous – they’ll be far more confident in using the model’s output. Additionally, analyzing errors (with confusion matrix and example case studies) can lead to further refinements in the pipeline (maybe back to feature engineering or data collection if needed). In summary, this stage ensures the model’s performance is **quantified and transparent**, which is indispensable for critical systems like PD monitoring.

## 7. Reporting and Experiment Management

Throughout the development of the PD classification pipeline, it’s important to **keep track of experiments, results, and models** in a structured way. This stage is about documentation, saving outputs, and managing the workflow so that we (and others) can reproduce and learn from the work. In an industrial or academic project context, dozens of experiments might be run (trying different features, algorithms, etc.), so systematic reporting and management prevent confusion and loss of information.

**Structured Outputs:** For each experiment or final model, we generate structured results – for example, CSV files or Excel sheets containing the performance metrics for each class, or even the prediction results on the test set. Key outcomes like confusion matrices, feature importance rankings, and hyperparameters used are saved in an organized manner (not just looked at on-screen and forgotten). Often, the pipeline code will output a summary in Markdown or PDF format that includes all important information (this could be an auto-generated report). By using markdown tables or CSV logs, we ensure results are easy to parse and compare.

For instance, we might have a CSV where each row is an experiment (with columns: features used, model type, hyperparams, accuracy, F1, etc.). This makes it easy to sort and find which experiment was best. If writing a thesis or report, we can copy from these structured outputs directly, ensuring consistency.

**Model Saving:** The trained model (or models) should be saved to disk in a suitable format (pickle/joblib for Python scikit-learn models, or as an ONNX file as we’ll discuss in deployment). We also often save the entire **pipeline**: i.e., the preprocessing steps + feature engineering + model all packaged together. This way, the exact model used in evaluation can be reloaded later for deployment or further testing. It’s common to version these saved models (like `model_v1.0.pkl`, `model_after_feature_selection.onnx`, etc.). In case we discover an issue or want to compare a new approach, we have the old model to benchmark against.

Additionally, saving the model allows us to apply it to new data easily (e.g., simulate it on data from another cable or from a later date). It’s crucial because retraining from scratch might not be feasible if we hand over the model to a client or another team.

**Experiment Tracking with MLflow/W\&B:** Tools like **MLflow** and **Weights & Biases (W\&B)** greatly facilitate experiment management. MLflow, for example, provides an API and UI for logging parameters, metrics, artifacts, and even the model itself for each run. Every time we train a model, we log the run in MLflow with tags like “ModelType=RandomForest”, hyperparameter values, the training data identifier, and resulting metrics. We can then compare runs in a dashboard, see trends, and even retrieve any model by date or parameters later. W\&B similarly allows logging and visualization of experiments on a dashboard, and it’s very convenient for generating plots of training curves, comparing confusion matrices, etc. Using such tools ensures that nothing gets lost – every experiment’s details are recorded.

For example, using MLflow we might query: *“show me all experiments where feature\_set = ‘entropy\_included’ and get the best F1”*. This kind of query can pinpoint which combination was best. Moreover, MLflow can version models in a **model registry** – so once we decide “this is the final model for deployment”, we register it, and any future improvements would be registered as new versions. This provides traceability (we know exactly which code and data produced model v1, which produced v2, and so on).

**Reporting Results:** In a final-year project context, it’s common to produce a detailed report (perhaps just like this one) describing each stage, the decisions made, and the outcomes. We include plots such as the confusion matrix, perhaps a PRPD pattern plot with model classifications highlighted, or feature importance charts. We may also report specific examples: e.g., *“Figure X shows a case where the model correctly identified internal PD in a cable termination, with SHAP values indicating phase distribution was the key feature.”* This makes the results tangible.

We likely also maintain **markdown logs or Jupyter notebooks** that narrate the experiments, so that later on, one can read through and see the progression of the project. Since readability and clarity are important (especially for others reviewing the work), we format tables and figures neatly, use headings (as per this guideline), and ensure each result is explained in text.

**Collaboration and Reproducibility:** If multiple people are working, or if this project is handed over to a company, proper experiment management means they can reproduce our environment. We’d keep track of versions of any libraries, random seeds used (for reproducible results), and data splitting details. That way, if the model is retrained later with new data, we can tell if improvements are due to new data or just randomness.

To sum up, we treat the machine learning pipeline not as ad-hoc scripts, but as a **well-documented process**. Each run yields artifacts (models, logs) that are saved. Each important finding (like “entropy features improved recall by 10%”) is recorded in reports or notebooks with evidence. This level of organization is what turns a good experiment into a robust, auditable system. In industries like power, such record-keeping is often necessary for compliance and safety reviews, because one might need to justify why the model is making decisions – having the training and testing records provides that justification.

> *Why this stage matters:* Good reporting and management ensure that the insights and models we’ve developed can be **trusted, verified, and iterated upon**. In high-voltage diagnostics, imagine a scenario where our model raises an alarm on a cable – the maintenance team might ask, *“How was this model made? How do we know it works?”* Being able to produce a detailed report of its validation performance and even the experiment history builds confidence. Furthermore, if down the line, new types of PD are observed, we’ll likely update the model. Experiment tracking will allow us to integrate new data and **measure improvement** against old performance, ensuring that any changes really are beneficial. It also prevents regressions (we won’t accidentally deploy a worse model because we have clear records of what the best was). In essence, this stage is about **rigor and continuity** – it turns the one-time analysis into a sustainable monitoring solution that can be maintained and improved systematically.

## 8. Deployment and Monitoring in the Field

With a validated model and thorough documentation, the final stage is to deploy the PD classification system for real-time use in cable monitoring, and to set up monitoring for its continued performance. Deployment involves making the model available in the operations environment (which could be on a server, embedded device, or cloud service that the sensors feed into). Monitoring means we keep an eye on the model’s predictions and the incoming data to detect any issues or drifts over time.

**Model Deployment (ONNX and Real-Time Inference):** In many cases, the model will be deployed on systems that cannot easily run the full development stack (like Python with scikit-learn) especially if it needs to run continuously and efficiently on perhaps an embedded computer at a substation. This is where **ONNX (Open Neural Network Exchange)** comes in. ONNX is an open format to represent machine learning models that is supported by many frameworks. We can take our trained model (e.g. a scikit-learn RandomForest or an XGBoost model) and convert it to an ONNX format. The **ONNX Runtime** is a high-performance engine that can load this model and run inference on a variety of platforms (Windows, Linux, even on ARM devices, etc.) very efficiently, typically with hardware optimizations. ONNX is considered a standard for deploying ML models and provides cross-platform, high-speed inference, especially on CPU-only environments.

Using ONNX, our PD classifier becomes a self-contained artifact that can be integrated into, say, a C++ monitoring software or a .NET application that already collects sensor data. The runtime handles the computations in optimized C/C++ code, which is much faster than running through a Python interpreter. This is crucial for real-time monitoring – if PD events are streaming in rapidly (for example, thousands of pulses per second), we need the model to classify them on the fly without lag. The ONNX Runtime can also leverage accelerators if available (like AVX instructions on CPUs, or even GPUs if we had a neural network model), but in our case CPU is typically sufficient.

So the deployment steps might be: export model to ONNX, write a small program that reads new sensor data (after the same kind of preprocessing pipeline we defined) and feeds features into the ONNX model, then triggers an alert or logs the classification result. We also include any necessary pre-processing as part of deployment – e.g., the normalization factor should be derived from training data and applied to new data in the same way. We might incorporate that into the pipeline before the model or handle it in code.

**Integration:** We ensure the deployed system is robust – for instance, what if the sensor data has a glitch (NaNs or spikes)? We might include sanity checks or fallbacks. Also, the system should be set up to produce notifications or visualizations for operators: e.g. “Cable #5 PD activity classified as Internal Discharge – Severity High” which might be shown on a dashboard or SCADA system.

**Model and Data Drift Monitoring:** Deployment is not the end – models can degrade over time due to **drift**. **Data drift** means the statistical properties of the input data change from what the model saw during training. In our case, perhaps the background noise level increases because a new piece of equipment was installed, or the cable ages and starts exhibiting a different PD pattern that wasn’t in the training set. **Concept drift** means the relationship between features and the target changes – for instance, maybe a new type of defect can produce high pulse counts without leading to failure (so previously that pattern meant danger, but now not necessarily). To handle this, we implement monitoring:

* We log the model’s predictions and the features of new data over time. We can use statistical tests or monitoring libraries (such as *Evidently AI* or built-in drift detection in some MLOps platforms) to compare the distribution of new input data to the training distribution. For example, track the mean and variance of each feature, or the frequency of predicted classes per day.

* If we detect a significant shift – say the model’s output distribution changes (it suddenly starts predicting a new class far more often) or input feature distributions shift beyond natural variation – this flags that **model performance may be deteriorating**. In practice, since we might not have immediate ground truth for whether an alarm was a true PD or not (someone has to inspect the cable to confirm, which might happen later), monitoring data drift is a proxy to catch issues early. For instance, if the PD classifier starts labeling many events as “unknown type” or showing low confidence (if such output exists), that’s something to investigate.

* We also continue to evaluate the model on any new labeled data that comes in. If a cable was inspected and a defect confirmed, we use that as a new test point – did our model predict it correctly? Over a year, we might accumulate more examples, and we can periodically retrain or fine-tune the model with this new data to improve it (this becomes a cycle of MLOps: continuous training).

**Alerts for Drift:** If our monitoring detects drift or unusual behavior, we might set up an alert for the engineering team. For example, *“Drift Warning: Feature distribution for ‘pulse\_energy’ has shifted by X amount compared to training baseline in the last month”*. This could suggest the environment changed or the sensor calibration shifted. In response, we might recalibrate sensors or retrain the model including recent data.

It’s worth noting that **unlike traditional software, ML models can “age” as data evolves**. Thus, deploying a model is not one-and-done; it requires a maintenance plan. High-voltage equipment can have a lifespan of decades, and the PD characteristics might evolve with aging insulation, or as partial repairs are done. Having drift monitoring ensures the model remains reliable through these changes. It’s much like how one would schedule periodic maintenance for physical equipment – here we schedule checks on our digital model’s health.

**Logging and Traceability:** In deployment, every prediction made by the model might be logged with a timestamp and key features, so that if an event occurs (e.g., a cable fails despite no alarm from the model, or vice versa), we can go back and analyze what the model saw and why it did or did not trigger. This ties back to explainability – we might even deploy a simplified SHAP analysis in the system to log “explanation” of each alarm. That could be overkill, but in critical systems, such transparency can be valuable.

Finally, we ensure the entire system is **user-friendly** for the operators. Maybe a dashboard visualizes PD activity trends and the model’s classifications, so human experts can cross-verify. The model should complement human monitoring, not completely replace it. Early in deployment, we might run the model in “shadow mode” (monitoring but not alerting) to build trust, comparing its outputs to traditional PD monitoring methods before fully relying on it.

> *Why this stage matters:* This is where the work delivers real value – the model runs on live data to give early warnings of insulation faults. By using a standardized deployment format like ONNX, we ensure the model can run **efficiently on the actual hardware** available (often just a CPU in a substation computer) and be integrated into existing systems. Real-time inference means we can catch PD events as they happen and possibly correlate with operating conditions. The drift monitoring is the safety net: it ensures our model remains as vigilant as day one. Without it, the model’s performance might silently degrade – which in a safety context is dangerous because you’d have **false confidence** in an outdated model. Instead, with drift detection, we can proactively update the model or retrain, keeping the monitoring system accurate over the years. In summary, deployment is about turning the trained model into a **practical tool** in the field, and setting up a process to **maintain its reliability** as conditions change. Early detection of PD in live operation can prompt maintenance at just the right time, preventing catastrophic cable failures and unplanned outages – which is the ultimate goal of the entire pipeline.

---

**Conclusion:** From ingesting raw, noisy signals to deploying an optimized model on the frontlines of the power grid, each step of the PD classification pipeline plays a vital role. By carefully cleaning and normalizing data, we start on a solid foundation. Through thoughtful feature extraction (guided by both data and physics) and expansion, we give our models the descriptive power needed to tell apart subtle fault signatures. Feature selection then hones this down to the essence, avoiding distraction by noise. Robust model training with hyperparameter tuning yields a system that can accurately flag dangerous PD activity while limiting false alarms. We validate and explain the model’s behavior to ensure it aligns with real-world expectations and to build trust. Finally, we wrap everything into a deployable, monitorable system that will keep watch over high-voltage cables in real time. Each stage is like a link in a chain – if any were weak or ignored, the whole outcome could fail (e.g., bad data in, bad predictions out). But when executed diligently, this pipeline provides a **powerful diagnostic tool**: one that can mean the difference between catching an insulation problem early or dealing with a costly, unexpected power failure. By understanding and justifying each step, a final-year industrial physics student (and future engineer) can appreciate not just how to implement such a pipeline, but why each component is essential for **keeping the lights on safely and reliably**.

**Sources:**

1. Hussein, R. *et al.* (2018). Denoising different types of acoustic partial discharge signals using power spectral subtraction. *High Volt.*, **3**, 44–50&#x20;

2. Nature Scientific Reports (2024). *Optimum feature selection for classification of PD signals produced by multiple insulation defects in electric motors*

3. MDPI Sensors (2024). *Identification of Partial Discharge Sources by Feature Extraction from a Signal Conditioning System*

4. Abdalrahman Shahrour (2023). *Optuna vs GridSearch – A Comparison of Hyperparameter Optimization Techniques*

5. Daksh Rathi (2024). *Handling Imbalanced Data: Key Techniques for Better Machine Learning*

6. NumberAnalytics Blog (2025). *Analyzing Confusion Matrix for Insights*

7. DataCamp. *Introduction to SHAP Values for Model Interpretability*

8. MLflow Documentation. *Experiment Tracking and Model Management*

9. Fiddler Labs (2020). *How to Detect Model Drift in ML Monitoring*

10. Reddit – r/LocalLLaMA (2023). *ONNX as a Standard for Deploying Models*
