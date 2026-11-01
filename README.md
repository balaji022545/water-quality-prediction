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
│   └── utils.py
│
├── models/
│   ├── best_model.joblib               # Self-contained champion pipeline (RF Tuned)
│   ├── random_forest_tuned.joblib      # Tuned Random Forest pipeline
│   ├── xgboost_tuned.joblib            # Tuned XGBoost pipeline
│   ├── preprocessor.joblib             # ColumnTransformer (Imputer + Scaler)
│   └── imputer.joblib                  # Standalone SimpleImputer
│
├── artifacts/                          # Visual diagnostic plots, reports & SHAP charts
│
├── app.py                              # (Phase 6) Streamlit Web Application
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
- **Class 0 (Not Potable / Unsafe)**:
  - The model's estimated probability for potable water is $< 0.50$.
  - Indicates that the chemical profile matches patterns commonly observed in non-potable samples within the Kaggle training dataset.
- **Class 1 (Potable / Safe)**:
  - The model's estimated probability for potable water is $\ge 0.50$.
  - Indicates that the chemical profile matches patterns observed in potable samples within the Kaggle training dataset.
- **Estimated Class Probabilities**:
  - The application displays the estimated probability distribution (e.g., `Class 0: 72.1%`, `Class 1: 27.9%`).
  - **Note**: The estimated probability is a statistical measure of model confidence based on dataset patterns, **NOT** the physical probability that the water is safe to drink.

### F. Critical Responsible-Use Limitations & Public Health Warning
1. **Auxiliary Screening Only**: This application is strictly an educational and decision-support screening tool; it **does NOT constitute official laboratory drinking-water certification**.
2. **Mandatory Laboratory Testing**: Accurate water potability clearance requires certified laboratory chemical spectrometry and bacteriological culture testing in compliance with WHO or EPA guidelines.
3. **Unmeasured Biological Pathogens**: The model evaluates only 9 inorganic/organic parameters. Fatal biological pathogens (e.g., *Escherichia coli*, *Vibrio cholerae*, *Salmonella*, *Giardia lamblia*, and enteroviruses) are **completely unmeasured** in this dataset. Chemically balanced water may still harbor lethal biological contaminants.
4. **Conservative Screening Bias**: The champion model exhibits high Class 0 recall (92.5%) and lower Class 1 recall (27.0%) at the standard 0.50 threshold, intentionally prioritizing caution over false reassurance.

