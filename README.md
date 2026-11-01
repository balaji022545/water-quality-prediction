# Water Quality Prediction Using Machine Learning

An end-to-end Machine Learning project designed to screen water potability (suitability for human consumption) based on physical and chemical water quality metrics.

---

## 📌 1. Problem Definition

Access to clean, safe drinking water is vital for human survival and disease prevention. Traditional water testing involves extensive laboratory chemical spectrometry and microbiological culture assays, which are resource-intensive and time-consuming.

This project implements a binary classification Machine Learning pipeline to screen water samples rapidly. By analyzing 9 physicochemical parameters, the model predicts whether a water sample is labeled **Potable** (`1`) or **Not Potable** (`0`).

> [!WARNING]
> **Responsible-Use & Regulatory Warning**:
> This machine learning model is strictly an **auxiliary statistical screening and research tool**. It **DOES NOT** provide certified drinking-water safety clearance or EPA/WHO regulatory compliance certification. Acute waterborne diseases are primarily caused by pathogenic microorganisms (*E. coli*, *Cryptosporidium*, viruses) which are **not recorded** in this chemical dataset.

---

## 🎯 2. Target Variable Explanation

The dataset target is **`Potability`**:
- **`0` (Not Potable / Unsafe)**: Water sample fails acceptable quality standards; it may be excessively mineralized, contaminated, or unsafe for direct human consumption.
- **`1` (Potable / Safe)**: Water sample satisfies acceptable safety thresholds based on the dataset annotations.

---

## 🧪 3. Input Features (9 Physicochemical Parameters)

| # | Feature | Health & Chemical Significance | Reference / Desirable Standard |
|---|---------|--------------------------------|--------------------------------|
| 1 | **`ph`** | Acidic vs. alkaline balance. Extremes corrode plumbing and dissolve toxic metals. | WHO standard: **6.5 – 8.5** |
| 2 | **`Hardness`** | Dissolved calcium and magnesium ions ($Ca^{2+}, Mg^{2+}$). Causes mineral scaling. | Desirable: **< 200 mg/L** |
| 3 | **`Solids`** | Total Dissolved Solids (TDS) in ppm. High mineral load impairs taste and palatability. | Safe ceiling: **< 1,000 mg/L** |
| 4 | **`Chloramines`** | Chlorine-ammonia disinfectant residuals controlling microbial contamination. | Maximum limit: **<= 4 ppm** |
| 5 | **`Sulfate`** | Dissolved sulfate minerals. Elevated concentrations induce laxative and bitter effects. | Desirable: **< 250 mg/L** |
| 6 | **`Conductivity`** | Electrical conductivity ($\mu S/cm$), reflecting total dissolved ionic concentration. | Freshwater standard: **< 400 $\mu S/cm$** |
| 7 | **`Organic_carbon`**| Total Organic Carbon (TOC) measuring decayed organic matter; precursors to disinfection byproducts. | Safe threshold: **< 4 ppm** |
| 8 | **`Trihalomethanes`**| Chemical byproducts produced when chlorine reacts with organic matter ($\mu g/L$). | Regulated limit: **<= 80 $\mu g/L$** |
| 9 | **`Turbidity`** | Water cloudiness caused by suspended colloids (clay, silt, microbes), hindering disinfection. | WHO guideline: **< 5 NTU**, ideal **< 1 NTU** |

---

## 🏆 4. Champion Model Architecture

- **Model**: **Tuned Random Forest Classifier**
- **Pipeline Architecture (Self-Contained & Leakage-Free)**:
  $$\text{Raw Features} \longrightarrow \mathbf{SimpleImputer}(\text{strategy}=\text{'median'}) \longrightarrow \mathbf{StandardScaler}() \longrightarrow \mathbf{RandomForestClassifier}()$$
- **Best Hyperparameters**:
  - `n_estimators`: `150`
  - `max_depth`: `15`
  - `min_samples_split`: `2`
  - `min_samples_leaf`: `1`
  - `max_features`: `'sqrt'`
  - `random_state`: `42`
- **Saved Pipeline Artifact**: [models/best_model.joblib](file:///c:/Users/balaj/OneDrive/Desktop/project_1/models/best_model.joblib)

---

## 📊 5. Evaluation & Benchmark Results

### A. Previously Recorded Final Test Evaluation ($N=656$, Unseen Test Set)
Evaluated once on the unseen 20% stratified holdout partition (no retraining, no leakage):

| Metric | Score | Public Health Interpretation |
|---|:---:|---|
| **Test Accuracy** | **0.6692** | 66.92% overall correct classifications (beats 60.99% majority baseline). |
| **Test ROC-AUC** | **0.6603** | Threshold-independent ranking capability. |
| **Class 0 Recall** *(Not Potable)* | **0.9250** | **92.50% of unsafe samples correctly flagged**. Highly protective against false safe alarms. |
| **Class 0 Precision** | **0.6643** | 66.43% of samples flagged as unsafe are truly unsafe. |
| **Class 0 F1-Score** | **0.7732** | Strong harmonic performance on the unsafe class. |
| **Class 1 Recall** *(Potable)* | **0.2695** | **Limitation**: At default 0.50 threshold, the model is conservative, capturing only 26.95% of potable samples. |
| **Class 1 Precision** | **0.6970** | When predicted potable, water is potable 69.70% of the time. |
| **Class 1 F1-Score** | **0.3887** | Reflects lower positive recall at default probability cutoff. |

#### Recorded Test Confusion Matrix ($N=656$):
```text
                       Predicted Unsafe (0)    Predicted Potable (1)
Actual Unsafe (0) [400]        370 (TN)                 30 (FP)
Actual Potable (1)[256]        187 (FN)                 69 (TP)
```

---

### B. Training-Data Cross-Validation Diagnostics (5-Fold Stratified OOF, $N=2,620$)
Generated strictly on the training partition via out-of-fold cross-validation with fold-level imputation and scaling:

- **OOF Accuracy**: `0.6683` (66.83%)
- **OOF ROC-AUC**: `0.6849` (Cross-validation fold mean: `0.6860`)
- **Class 0 (Not Potable) Diagnostics**:
  - Recall: **0.9061** (1,448 of 1,598 unsafe samples caught)
  - Precision: **0.6682**
  - F1-Score: **0.7692**
- **Class 1 (Potable) Diagnostics**:
  - Recall: **0.2965** (303 of 1,022 potable samples caught)
  - Precision: **0.6689**
  - F1-Score: **0.4108**
- **OOF Confusion Matrix**:
  ```text
  [[1448 (TN),  150 (FP)],
   [ 719 (FN),  303 (TP)]]
  ```

---

## 🔍 6. SHAP Explainability & Feature Importance Findings

Feature attributions were generated using `shap.TreeExplainer` on the Random Forest champion model using a representative training sample ($N=300$):

### Global Feature Importance Ranking (Mean Absolute SHAP Value):
1. **`Sulfate`** (`0.0596`): Primary driver of model predictions.
2. **`ph`** (`0.0424`): Second most influential factor.
3. **`Hardness`** (`0.0329`): Moderate influence on decision boundaries.
4. **`Solids`** (`0.0261`): Dissolved solids influence.
5. **`Chloramines`** (`0.0253`): Residual disinfectant impact.
6. **`Conductivity`** (`0.0205`): Total ion contribution.
7. **`Trihalomethanes`** (`0.0165`): Byproduct chemical level.
8. **`Turbidity`** (`0.0145`): Suspended particle contribution.
9. **`Organic_carbon`** (`0.0130`): Organic decay matter.

### Key SHAP Observations:
- **Non-Linear Threshold Effects**:
  The beeswarm summary plot shows that **Sulfate** and **pH** exhibit pronounced non-linear spreads. Extreme acidic (pH < 6.5) and extreme basic (pH > 8.5) values push SHAP attributions downward (toward Not Potable).
- **Attribution vs. Causation**:
  SHAP values describe **how the Random Forest combines features**, not biological or physical causation. High sulfate does not cause water to be safe; rather, observations with balanced sulfate within typical ranges are statistically associated with potability labels in this dataset.

---

## 📁 7. Saved Diagnostic & Visual Artifacts

All figures and reports are stored under [artifacts/](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts):
- **Diagnostic Visualizations**:
  - [test_confusion_matrix_recorded.png](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts/test_confusion_matrix_recorded.png): Recorded final test confusion matrix (one-time benchmark).
  - [cv_confusion_matrix.png](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts/cv_confusion_matrix.png): 5-fold training out-of-fold confusion matrix.
  - [cv_roc_curve.png](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts/cv_roc_curve.png): Out-of-fold ROC curve ($AUC = 0.6849$).
  - [cv_precision_recall_curve.png](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts/cv_precision_recall_curve.png): Out-of-fold PR curve.
- **SHAP Interpretability Artifacts**:
  - [shap_summary_beeswarm.png](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts/shap_summary_beeswarm.png): Beeswarm plot displaying feature magnitude and directionality.
  - [shap_feature_importance_bar.png](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts/shap_feature_importance_bar.png): Global feature importance bar chart.
  - [shap_local_sample_potable.png](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts/shap_local_sample_potable.png): Local waterfall explanation for a high-probability potable sample.
  - [shap_local_sample_not_potable.png](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts/shap_local_sample_not_potable.png): Local waterfall explanation for a low-probability (unsafe) sample.
- **Machine-Readable Reports**:
  - `test_evaluation_report.json`
  - `cv_diagnostic_metrics.json`
  - `shap_feature_importance.json`

---

## 📁 8. Complete Project Architecture

```text
project_1/
│
├── data/
│   ├── raw/
│   │   └── water_potability.csv        # Immutable raw dataset (3,276 rows × 10 cols)
│   └── processed/
│       ├── X_train.csv / X_test.csv    # Raw partitioned features (missingness preserved)
│       ├── y_train.csv / y_test.csv    # Target vectors (Potability)
│       ├── train.csv / test.csv        # Merged feature + target tables
│       ├── X_train_imputed.csv         # Reference median-imputed features
│       └── X_train_scaled.csv          # Reference standardized features
│
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py               # (Phase 2) Loading & preliminary schema audit
│   ├── data_preprocessing.py           # (Phase 3) Stratified split & preprocessing pipeline
│   ├── model_training.py               # (Phase 4) Fold-isolated training, SMOTE & tuning
│   ├── model_evaluation.py             # (Phase 5) Evaluation report & OOF diagnostics
│   ├── model_explainability.py         # (Phase 5) SHAP TreeExplainer & local interpretations
│   ├── threshold_tuning.py             # Classification threshold evaluation & selection
│   ├── class_imbalance.py              # (Phase 7) Class imbalance experiments & comparison
│   └── utils.py                        # Helper functions for threshold loading & validation
│
├── models/
│   ├── best_model.joblib               # Self-contained champion pipeline (RF Tuned)
│   ├── random_forest_tuned.joblib      # Tuned Random Forest pipeline
│   ├── xgboost_tuned.joblib            # Tuned XGBoost pipeline
│   ├── preprocessor.joblib             # ColumnTransformer (Imputer + Scaler)
│   ├── imputer.joblib                  # Standalone SimpleImputer
│   └── experiments/
│       ├── best_imbalance_model.joblib # Phase 7 experimental model (Random Undersampling)
│       └── experiment_info.json        # Experimental model metadata
│
├── artifacts/
│   ├── class_distribution.png          # Training set class imbalance visualization
│   ├── imbalance_comparison.csv        # 6 imbalance experiments comparison table
│   ├── imbalance_accuracy_comparison.png # Accuracy & Balanced Accuracy chart
│   ├── imbalance_recall_comparison.png # Class 0 vs Class 1 recall chart
│   ├── imbalance_f1_comparison.png     # F1-score comparison chart
│   ├── imbalance_confusion_matrices.png# 2x3 confusion matrix grid across experiments
│   ├── best_imbalance_threshold_comparison.csv # Threshold analysis for winning candidate
│   ├── phase_7_report.md               # Detailed technical report for Phase 7
│   ├── threshold_comparison.csv        # Metrics across 11 candidate thresholds (0.20 - 0.70)
│   ├── threshold_metrics.png           # Performance curve vs threshold
│   ├── threshold_confusion_matrix.png  # Confusion matrix at optimal threshold (0.40)
│   ├── threshold_config.json           # Synchronized active threshold metadata
│   └── ...                             # Prior diagnostic plots & reports
│
├── app.py                              # (Phase 6) Streamlit Web Application (Threshold-Tuned)
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation & execution guide
```

---

## 🚀 9. Streamlit Web Application: Setup & Execution Guide

An interactive web interface is provided in [app.py](file:///c:/Users/balaj/OneDrive/Desktop/project_1/app.py) powered by **Streamlit**.

### A. Environment Installation
Ensure Python (version 3.10+ recommended) is available, then install dependencies:
```bash
pip install -r requirements.txt
```

### B. Launching the Web Application
Run the Streamlit application from the project root directory:
```bash
streamlit run app.py
```
This will launch a local server and display a browser link:
```text
Local URL: http://localhost:8501
Network URL: http://<your-network-ip>:8501
```

### C. Required Input Features & Physical Units
The application accepts 9 physicochemical parameters:

| Parameter | UI Field Label | Physical Units | Typical Range | Description & Notes |
|---|---|---|---|---|
| `ph` | **pH Level** | pH scale (0–14) | 0.0 – 14.0 | Acid-base balance. Optimal neutral baseline is ~7.0. |
| `Hardness` | **Hardness** | mg/L | 0.0 – 400.0 | Dissolved calcium ($Ca^{2+}$) and magnesium ($Mg^{2+}$) mineral content. |
| `Solids` | **Solids (TDS)** | ppm | 0.0 – 70,000.0 | Total Dissolved Solids indicating dissolved minerals and salts. |
| `Chloramines` | **Chloramines** | ppm | 0.0 – 20.0 | Disinfection compound residue controlling microbial agents. |
| `Sulfate` | **Sulfate** | mg/L | 0.0 – 600.0 | Naturally occurring dissolved sulfate mineral salts. |
| `Conductivity` | **Conductivity** | µS/cm | 0.0 – 1,000.0 | Ability of water to conduct electrical current via dissolved ions. |
| `Organic_carbon`| **Organic Carbon** | ppm | 0.0 – 35.0 | Total organic carbon (TOC) from decayed matter. |
| `Trihalomethanes`| **Trihalomethanes** | µg/L | 0.0 – 150.0 | Chlorination byproducts formed during water treatment. |
| `Turbidity` | **Turbidity** | NTU | 0.0 – 10.0 | Water cloudiness/haziness caused by suspended colloids. |

### D. Missing Value Handling
- **Optional Missing Values**: If a measurement is unknown or unmeasured, check the **"Unmeasured (NaN)"** checkbox next to that feature.
- **Dynamic Median Imputation**: Unmeasured features are passed as `np.nan` directly to `models/best_model.joblib`. The pipeline's embedded `SimpleImputer` replaces them using the median values computed from the training dataset.

### E. How to Interpret Model Predictions
- **Classification Decision Rule**:
  $$\hat{y} = \begin{cases} 1 \ (\text{Potable}), & \text{if } P(\text{Potable} \mid X) \ge \tau^* \\ 0 \ (\text{Not Potable}), & \text{otherwise} \end{cases}$$
  where $\tau^* = 0.40$ is the tuned decision cutoff loaded dynamically from [artifacts/threshold_config.json](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts/threshold_config.json).
- **Class 0 (Not Potable / Unsafe)**:
  - Model estimated probability for potable water is $< 0.40$.
  - Indicates that the chemical profile exhibits characteristics typical of non-potable samples in the training dataset.
- **Class 1 (Potable / Safe)**:
  - Model estimated probability for potable water is $\ge 0.40$.
  - Indicates that the chemical profile satisfies dataset-based potability patterns under the tuned decision boundary.
- **Estimated Class Probabilities**:
  - The application displays the estimated probability distribution (e.g., `Class 0: 67.5%`, `Class 1: 32.5%`).
  - **Important**: The model probability is a statistical measure of model confidence based on dataset patterns, **NOT** the physical probability that the water is safe to drink.

---

## ⚖️ 10. Classification Threshold Tuning & Decision Optimization

### A. What is a Classification Threshold?
In probabilistic binary classification, machine learning algorithms output continuous posterior class probabilities:
$$P(Y = 1 \mid X) \in [0, 1]$$
To assign a discrete categorical label, a **decision threshold ($\tau$)** is applied:
$$\hat{Y} = \mathbb{I}(P(Y=1 \mid X) \ge \tau)$$
By default, standard Scikit-learn classifiers apply an arbitrary cutoff of $\tau = 0.50$. However, in real-world scenarios with moderate class imbalance, asymmetric risk, or non-uniform probability distributions, the default 0.50 threshold is rarely optimal.

### B. Why Threshold Tuning was Introduced
In the original model benchmarks, the champion Random Forest model achieved strong performance on the majority class but suffered from poor sensitivity on the minority class:
- **Class 0 Recall**: **90.61%** (CV OOF) / **92.50%** (Test)
- **Class 1 Recall**: **29.65%** (CV OOF) / **26.95%** (Test)

Because the dataset contains fewer potable instances (39%) and Random Forest tree ensembles average hundreds of trees (which naturally compresses extreme probabilities toward the prior mean of ~0.389), the model required an excessively high standard of evidence to output $P \ge 0.50$. As a result, **over 70% of safe drinking water samples were misclassified as unsafe**.

Threshold tuning decouples probability estimation from the categorical decision rule, allowing us to find a decision cutoff that provides a much better balance between safe and unsafe classifications without retraining the model.

### C. Zero-Leakage Validation Strategy
To prevent data leakage, the optimal threshold was chosen **exclusively using training data**:
1. **Partition Used**: `data/processed/X_train.csv` ($N=2,620$) and `data/processed/y_train.csv`.
2. **Cross-Validation**: Stratified 5-Fold Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
3. **Fold Isolation**: Within each fold, `SimpleImputer` and `StandardScaler` were fitted only on the 4 training folds, and predictions were evaluated on the held-out 5th validation fold.
4. **Out-of-Fold Probabilities**: Every sample's predicted probability was produced by an ensemble that never saw that sample during fitting.
5. **Holdout Test Isolation**: The unseen test partition (`X_test.csv`, `y_test.csv`) was **strictly preserved** and not used to evaluate or tune candidate thresholds.

### D. Threshold Grid Evaluation (11 Candidate Thresholds)
Across the 11 candidate thresholds evaluated on the out-of-fold training predictions ($N=2,620$):

| Threshold ($\tau$) | Accuracy | Balanced Accuracy | Class 1 Precision | Class 1 Recall | Class 1 F1 | Class 0 Precision | Class 0 Recall | Class 0 F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.20 | 0.4252 | 0.5253 | 0.4027 | **0.9804** | 0.5709 | 0.8485 | 0.0701 | 0.1295 |
| 0.25 | 0.4634 | 0.5488 | 0.4165 | 0.9374 | 0.5768 | 0.8000 | 0.1602 | 0.2669 |
| 0.30 | 0.5275 | 0.5887 | 0.4457 | 0.8669 | 0.5887 | 0.7848 | 0.3104 | 0.4448 |
| 0.35 | 0.5935 | 0.6177 | 0.4860 | 0.7280 | 0.5828 | 0.7447 | 0.5075 | 0.6036 |
| **0.40 (Optimal)** | **0.6458** | **0.6336** | **0.5432** | **0.5783** | **0.5602** | **0.7187** | **0.6890** | **0.7035** |
| 0.45 | 0.6683 | 0.6237 | 0.6082 | 0.4207 | 0.4974 | 0.6905 | 0.8267 | 0.7525 |
| **0.50 (Default)** | **0.6683** | **0.6013** | **0.6689** | **0.2965** | **0.4108** | **0.6682** | **0.9061** | **0.7692** |
| 0.55 | 0.6607 | 0.5786 | 0.7317 | 0.2055 | 0.3209 | 0.6520 | 0.9518 | 0.7738 |
| 0.60 | 0.6508 | 0.5590 | 0.7923 | 0.1419 | 0.2407 | 0.6401 | 0.9762 | 0.7732 |
| 0.65 | 0.6351 | 0.5360 | 0.8056 | 0.0851 | 0.1540 | 0.6278 | 0.9869 | 0.7674 |
| 0.70 | 0.6290 | 0.5266 | 0.8378 | 0.0607 | 0.1131 | 0.6229 | 0.9925 | 0.7654 |

### E. Selected Threshold & Rational Trade-Off Analysis
**Selected Optimal Threshold**: **`τ = 0.40`**

**Selection Justification**:
1. **Global Maximum in Balanced Accuracy**:
   Balanced accuracy reaches its global peak across all thresholds at **0.6336** (up from 0.6013 at 0.50).
2. **Nearly Doubled Class 1 Sensitivity**:
   Class 1 Recall increases from **29.65% to 57.83%** (+28.18 percentage points), detecting 591 safe samples compared to only 303 at 0.50.
3. **Significant Gain in Class 1 F1-Score**:
   F1-score for Potable water improves from **0.4108 to 0.5602** (+14.94 percentage points).
4. **Strong Protection of Screening Power**:
   Class 0 Recall remains at a robust **68.90%** (1,101 of 1,598 unsafe samples caught), and Class 0 Precision is **71.87%**.
5. **Comparison with Lower Cutoffs (e.g., 0.35)**:
   While $\tau = 0.35$ achieves 72.80% Class 1 recall, its Class 0 recall drops to 50.75% (missing nearly half of all unsafe water samples), which poses unacceptable safety risks. Thus, **$\tau = 0.40$ represents the optimal Pareto frontier**.

### F. Before-and-After Comparison

#### 1. Training Out-of-Fold Cross-Validation Metrics ($N=2,620$):
| Metric | Default ($\tau = 0.50$) | Tuned ($\tau = 0.40$) | Absolute Difference |
|---|:---:|:---:|:---:|
| **Balanced Accuracy** | 0.6013 | **0.6336** | **+3.23%** |
| **Class 1 Recall (Potable)** | 0.2965 (303/1022) | **0.5783 (591/1022)** | **+28.18% (Almost Doubled)** |
| **Class 1 Precision** | 0.6689 | 0.5432 | -12.57% (Trade-off) |
| **Class 1 F1-Score** | 0.4108 | **0.5602** | **+14.94%** |
| **Class 0 Recall (Unsafe)** | 0.9061 (1448/1598) | **0.6890 (1101/1598)** | -21.71% (Screening adjusted) |
| **Class 0 Precision** | 0.6682 | **0.7187** | **+5.05%** |
| **Overall Accuracy** | 0.6683 | 0.6458 | -2.25% |

#### 2. One-Time Unseen Test Set Verification ($N=656$ Holdout):
| Metric | Test ($\tau = 0.50$) | Test Tuned ($\tau = 0.40$) | Impact on Generalization |
|---|:---:|:---:|:---:|
| **Balanced Accuracy** | 0.5973 | **0.6070** | Improved generalization balance |
| **Class 1 Recall (Potable)** | 0.2695 (69/256) | **0.5039 (129/256)** | **+23.44% (Caught 60 additional safe samples)** |
| **Class 1 Precision** | 0.6970 | 0.5265 | Expected precision-recall trade-off |
| **Class 1 F1-Score** | 0.3887 | **0.5150** | **+12.63% improvement** |
| **Class 0 Recall (Unsafe)** | 0.9250 (370/400) | **0.7100 (284/400)** | Maintains solid 71% screening |
| **Class 0 Precision** | 0.6643 | **0.6910** | **+2.67% improvement** |
| **Overall Accuracy** | 0.6692 | 0.6296 | Retains 63% overall correctness |

---

## ⚠️ 11. Important Limitations & Public Health Safety Warning

> [!WARNING]
> **CRITICAL RESPONSIBLE-USE NOTICE**:
> 1. **Auxiliary Decision-Support Tool Only**: This machine learning model and web interface are intended strictly for academic, educational, and preliminary screening research. It **DOES NOT** provide certified drinking-water safety clearance or EPA/WHO regulatory approval.
> 2. **Professional Laboratory Testing is Mandatory**: Genuine water potability certification requires comprehensive multi-parameter chemical spectrometry and microbiological culture assays performed by certified public health laboratories.
> 3. **Unmeasured Biological Pathogens**: This dataset records 9 inorganic/organic physicochemical measurements. Acute, deadly waterborne diseases are caused by **microbiological pathogens (such as *Escherichia coli*, *Vibrio cholerae*, *Salmonella*, *Giardia lamblia*, and enteroviruses)**, which are completely absent from this dataset. Chemically balanced water may still contain lethal bacterial or viral pathogens.
> 4. **Statistical Model vs. Physical Reality**: Threshold tuning shifts the model's decision sensitivity to detect more safe water samples based on training correlations. It cannot compensate for unmeasured physical contaminants or replace physical water treatment processes (filtration, boiling, chlorination, UV irradiation).

---

## ⚖️ 12. Phase 7: Class Imbalance Handling & Optimization Summary

An empirical investigation was conducted across 6 class imbalance handling techniques using strict Stratified 5-Fold Cross-Validation on the training data ($N=2,620$, zero test leakage):

### A. Training Class Distribution
- **Class 0 (Not Potable / Unsafe)**: **1,598 samples (60.99%)**
- **Class 1 (Potable / Safe)**: **1,022 samples (39.01%)**
- **Imbalance Ratio**: **1.564 : 1** (Moderate Imbalance)
- **Artifact**: [artifacts/class_distribution.png](artifacts/class_distribution.png)

### B. Experimental Results at Threshold 0.50 (5-Fold CV Out-of-Fold, $N=2,620$):

| Experiment | Technique | Accuracy | Balanced Accuracy | Recall (C1) | Precision (C1) | F1-Score (C1) | Recall (C0) | ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp A** | No Resampling (Baseline) | 0.6683 | 0.6013 | 0.2965 | **0.6689** | 0.4108 | **0.9061** | 0.6849 |
| **Exp B** | Random Oversampling | 0.6672 | 0.6242 | 0.4286 | 0.6033 | 0.5011 | 0.8198 | 0.6862 |
| **Exp C** | **Random Undersampling (Winner)** | 0.6401 | **0.6318** | **0.5939** | 0.5348 | **0.5628** | 0.6696 | 0.6796 |
| **Exp D** | SMOTE Synthetic Oversampling | 0.6542 | 0.6301 | 0.5205 | 0.5612 | 0.5401 | 0.7397 | 0.6850 |
| **Exp E** | Class Weight (`balanced`) | **0.6698** | 0.6117 | 0.3474 | 0.6420 | 0.4508 | 0.8761 | **0.6921** |
| **Exp F** | Balanced Subsample | 0.6630 | 0.6031 | 0.3307 | 0.6294 | 0.4336 | 0.8755 | 0.6867 |

### C. Key Findings & Trade-Offs
1. **Best Imbalance Technique**: **Random Undersampling (Exp C)** achieved the highest Balanced Accuracy (**0.6318**) and doubled Class 1 Recall to **59.39%** (capturing 607 potable samples vs. 303 in baseline).
2. **Overall Accuracy Trade-Off**: Overall accuracy decreased slightly from 66.83% to 64.01% (-2.82 percentage points) due to the inherent trade-off of reducing majority-class false reassurance.
3. **Imbalance Handling vs. Threshold Tuning**: Both Random Undersampling at $\tau = 0.50$ (Balanced Accuracy: 0.6318, Class 1 Recall: 59.39%) and Threshold Tuning at $\tau = 0.40$ on the baseline model (Balanced Accuracy: 0.6336, Class 1 Recall: 57.83%) achieve virtually identical operating points. However, Threshold Tuning preserves all training data without discarding samples.
4. **Model Preservation**: The production champion model (`models/best_model.joblib`) and active threshold configuration (`artifacts/threshold_config.json`, $\tau = 0.40$) remain active and protected. The winning experimental model is archived separately under [models/experiments/best_imbalance_model.joblib](models/experiments/best_imbalance_model.joblib).
5. **Comprehensive Technical Report**: For detailed methodology and diagnostic plots, see [artifacts/phase_7_report.md](artifacts/phase_7_report.md).



