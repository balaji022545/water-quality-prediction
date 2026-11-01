# Phase 7 Report: Class Imbalance Handling and Model Optimization

**Project**: Water Quality Prediction Using Machine Learning  
**Champion Model**: Tuned Random Forest (`n_estimators=150, max_depth=15, random_state=42, n_jobs=1`)  
**Target Variable**: `Potability` (`0 = Not Potable / Unsafe`, `1 = Potable / Safe`)  
**Scope**: Systematic empirical investigation of six class imbalance handling strategies using strict 5-fold cross-validation out-of-fold (OOF) evaluation on training data ($N=2,620$).

---

## 1. What Class Imbalance Means

In binary classification, **class imbalance** occurs when the categorical outcomes in the target variable are not equally represented. One class (the **majority class**) contains substantially more samples than the other (the **minority class**).

When standard machine learning algorithms minimize global empirical loss (such as Gini impurity or binary cross-entropy), they are heavily incentivized to predict the majority class. Even if the classifier performs poorly on the minority class, it can achieve deceptively high accuracy simply by predicting the majority class for most samples. In health and environmental applications, the minority class (e.g., safe drinking water or rare toxic contamination) often represents the primary positive interest.

---

## 2. Actual Training Data Class Distribution

From `data/processed/X_train.csv` and `data/processed/y_train.csv` ($N=2,620$):

| Class | Semantic Meaning | Count ($N$) | Percentage (%) | Representation |
|:---:|---|:---:|:---:|:---:|
| **Class 0** | Not Potable (Unsafe) | **1,598** | **60.99%** | Majority Class |
| **Class 1** | Potable (Safe) | **1,022** | **39.01%** | Minority Class |
| **Total** | Full Training Partition | **2,620** | **100.00%** | — |

- **Majority-to-Minority Imbalance Ratio**:
  $$\text{Ratio} = \frac{1,598}{1,022} = \mathbf{1.564 : 1}$$
- **Degree of Imbalance**:
  This distribution is characterized as a **moderate (or mild-to-moderate) imbalance**. Unlike extreme anomalies (such as credit fraud at 100:1 or rare disease screening at 1000:1), a 61:39 split retains substantial minority support (over 1,000 real samples). However, when coupled with tree-ensemble probability averaging and standard 0.50 thresholding, it caused the baseline model to miss over 70% of potable water samples.
- **Visual Artifact**: [artifacts/class_distribution.png](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts/class_distribution.png)

---

## 3. Why Imbalance Handling is Relevant to this Dataset

In earlier project phases, the baseline Random Forest classifier at the default 0.50 decision threshold produced:
- **Class 0 Recall**: **90.61%** (caught 1,448 of 1,598 unsafe samples)
- **Class 1 Recall**: **29.65%** (caught only 303 of 1,022 safe samples)
- **Class 1 F1-Score**: **0.4108**

The model was excessively biased toward classifying samples as non-potable. Investigating whether sampling methods (oversampling, undersampling, SMOTE) or algorithmic weighting (`class_weight`) can rebalance the decision boundary during training—without degrading overall discriminative ability—is crucial to understanding model optimization options.

---

## 4. Explanation of the Six Techniques Evaluated

1. **Experiment A: No Resampling (Baseline)**:
   - Uses the original unaugmented training partition.
   - Preserves the empirical 61:39 data distribution. Serves as the benchmark for all comparisons.
2. **Experiment B: Random Oversampling (`RandomOverSampler`)**:
   - Randomly duplicates existing minority-class (Class 1) instances with replacement until the training partition has equal numbers of Class 0 and Class 1 ($N = 1,598$ each).
   - *Risk*: Duplicating identical samples can cause decision trees to form overly specific, localized leaf rules, potentially leading to overfitting.
3. **Experiment C: Random Undersampling (`RandomUnderSampler`)**:
   - Randomly removes majority-class (Class 0) instances without replacement until both classes match the minority count ($N = 1,022$ each).
   - *Advantage*: Balances class priors directly without synthetic artifacts.
   - *Risk*: Discards ~36% of majority-class information ($576$ samples per fold), which may slightly reduce classifier precision.
4. **Experiment D: Synthetic Minority Over-sampling Technique (`SMOTE`)**:
   - Rather than duplicating rows, SMOTE synthesizes new minority-class examples along the feature-space line segments connecting $k=5$ nearest minority neighbours.
   - *Advantage*: Broadens the minority decision region without exact duplicates.
   - *Risk*: In noisy feature spaces with high overlap (like water potability), synthetic samples can bridge class clusters and generate noisy borderline points.
5. **Experiment E: Algorithmic Class Weighting (`class_weight='balanced'`)**:
   - Leaves training data instances unchanged. Instead, modifies the loss function during tree node splitting by inversely weighting sample frequencies:
     $$w_j = \frac{N}{2 \cdot N_j}$$
     $$w_0 = \frac{2620}{2 \times 1598} \approx 0.82, \quad w_1 = \frac{2620}{2 \times 1022} \approx 1.28$$
   - Penalizes misclassifications on minority samples $1.56\times$ more heavily.
6. **Experiment F: Bootstrap Balanced Subsampling (`class_weight='balanced_subsample'`)**:
   - Similar to `class_weight='balanced'`, but recomputes class weights dynamically for each individual bootstrap sample drawn when building each tree in the Random Forest ensemble.

---

## 5. Implementation Details & Architecture

All experiments were executed with strict architectural consistency:
- **Base Classifier**: `RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=1)`
- **Preprocessing Pipeline**: `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`
- **Pipeline Harness**: `imblearn.pipeline.Pipeline`
- **Execution Script**: [src/class_imbalance.py](file:///c:/Users/balaj/OneDrive/Desktop/project_1/src/class_imbalance.py)

---

## 6. Cross-Validation Methodology & Out-of-Fold (OOF) Prediction

- **Validation Splitter**: Stratified 5-Fold Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
- **Out-of-Fold Estimation**: Evaluated via `cross_val_predict(..., method='predict_proba', n_jobs=1)[:, 1]`.
- Every sample in the training set ($N=2,620$) received a predicted probability generated exclusively by a model fold that did not observe that sample during fitting.
- Identical fold splits were applied across all six experiments, ensuring fair comparisons.

---

## 7. Leakage Prevention Protocol

Data leakage was systematically prevented:
1. **Pipeline Embedding**: Samplers (`RandomOverSampler`, `RandomUnderSampler`, `SMOTE`) were placed **after** `SimpleImputer` and `StandardScaler` inside `imblearn.pipeline.Pipeline`.
2. **Fold-Isolated Fitting**: Imputation medians, scaling means/variances, and resampled training sets were computed **exclusively on the 4 training folds** ($N \approx 2,096$).
3. **Untouched Validation Folds**: Validation folds ($N \approx 524$) were transformed using training parameters and **never resampled**.
4. **Complete Test Set Isolation**: Neither `X_test.csv` nor `y_test.csv` was loaded or evaluated during model comparison, ranking, or threshold exploration.

---

## 8. Complete Comparison Table (Threshold 0.50, $N=2,620$ OOF)

| Experiment | Accuracy | Balanced Accuracy | Precision (C0) | Recall (C0) | F1 (C0) | Precision (C1) | Recall (C1) | F1 (C1) | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp A: No Resampling (Baseline)** | 0.6683 | 0.6013 | 0.6682 | **0.9061** | **0.7692** | **0.6689** | 0.2965 | 0.4108 | 0.6849 | 0.5987 | 1448 | 150 | 719 | 303 |
| **Exp B: Random Oversampling** | 0.6672 | 0.6242 | 0.6917 | 0.8198 | 0.7503 | 0.6033 | 0.4286 | 0.5011 | 0.6862 | 0.5973 | 1310 | 288 | 584 | 438 |
| **Exp C: Random Undersampling** | 0.6401 | **0.6318** | **0.7205** | 0.6696 | 0.6941 | 0.5348 | **0.5939** | **0.5628** | 0.6796 | 0.5923 | 1070 | 528 | 415 | 607 |
| **Exp D: SMOTE** | 0.6542 | 0.6301 | 0.7069 | 0.7397 | 0.7229 | 0.5612 | 0.5205 | 0.5401 | 0.6850 | 0.5837 | 1182 | 416 | 490 | 532 |
| **Exp E: Class Weight (balanced)** | **0.6698** | 0.6117 | 0.6773 | 0.8761 | 0.7640 | 0.6420 | 0.3474 | 0.4508 | **0.6921** | **0.6016** | 1400 | 198 | 667 | 355 |
| **Exp F: Balanced Subsample** | 0.6630 | 0.6031 | 0.6716 | 0.8755 | 0.7601 | 0.6294 | 0.3307 | 0.4336 | 0.6867 | 0.5936 | 1399 | 199 | 684 | 338 |

- **Comparison CSV**: [artifacts/imbalance_comparison.csv](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts/imbalance_comparison.csv)
- **Visual Plots**:
  - [artifacts/imbalance_accuracy_comparison.png](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts/imbalance_accuracy_comparison.png)
  - [artifacts/imbalance_recall_comparison.png](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts/imbalance_recall_comparison.png)
  - [artifacts/imbalance_f1_comparison.png](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts/imbalance_f1_comparison.png)
  - [artifacts/imbalance_confusion_matrices.png](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts/imbalance_confusion_matrices.png)

---

## 9. Confusion Matrix Interpretation

Examining the confusion matrices across techniques:
- **Baseline (Exp A)**: High TN ($1,448$), but high FN ($719$). It severely underpredicts Potable water (only $303$ TP).
- **Random Undersampling (Exp C)**: Captures **$607$ True Positives** (more than double the baseline), reducing False Negatives from $719$ down to $415$. However, False Positives increase from $150$ to $528$.
- **SMOTE (Exp D)**: Captures **$532$ True Positives**, reducing FN to $490$, with $416$ False Positives.
- **Class Weighting (Exp E & F)**: Provides a slight shift toward Class 1 ($355$ and $338$ TP), but the shift is modest compared to resampling.

---

## 10. Best-Performing Approach and Selection Rationale

### Winner: **Experiment C: Random Undersampling**

**Why Random Undersampling Won**:
1. **Highest Balanced Accuracy**:
   Reaches **0.6318** (vs. Baseline 0.6013, SMOTE 0.6301, Oversampling 0.6242).
2. **Highest Minority Class Recall (Sensitivity)**:
   Class 1 Recall reaches **59.39%** (vs. Baseline 29.65%), successfully identifying **607 out of 1,022 potable water samples**.
3. **Highest Class 1 F1-Score**:
   Class 1 F1-score peaks at **0.5628** (vs. Baseline 0.4108, a +15.20 percentage point improvement).
4. **Simplicity Over Synthetic Generation**:
   Random Undersampling is simpler, faster, and introduces no artificial interpolation artifacts unlike SMOTE, while achieving slightly better balanced accuracy ($0.6318$ vs $0.6301$) and higher Class 1 recall ($59.39\%$ vs $52.05\%$).

---

## 11. Additional Threshold Analysis for the Best Candidate (Exp C)

Evaluating training out-of-fold probabilities for **Random Undersampling** across candidate thresholds:

| Threshold ($\tau$) | Accuracy | Balanced Accuracy | Precision (C1) | Recall (C1) | F1 (C1) | Recall (C0) | Precision (C0) | F1 (C0) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.30 | 0.4309 | 0.5298 | 0.4051 | 0.9795 | 0.5731 | 0.0801 | 0.8591 | 0.1465 |
| 0.35 | 0.4595 | 0.5451 | 0.4145 | 0.9344 | 0.5743 | 0.1558 | 0.7880 | 0.2602 |
| 0.40 | 0.5164 | 0.5808 | 0.4397 | 0.8738 | 0.5850 | 0.2879 | 0.7810 | 0.4207 |
| 0.45 | 0.5767 | 0.6084 | 0.4732 | 0.7524 | 0.5810 | 0.4643 | 0.7457 | 0.5723 |
| **0.50 (Natural Peak)** | **0.6401** | **0.6318** | **0.5348** | **0.5939** | **0.5628** | **0.6696** | **0.7205** | **0.6941** |
| 0.55 | 0.6676 | 0.6255 | 0.6024 | 0.4344 | 0.5048 | 0.8166 | 0.6930 | 0.7498 |
| 0.60 | 0.6664 | 0.5985 | 0.6667 | 0.2896 | 0.4038 | 0.9074 | 0.6664 | 0.7684 |

- **Artifact**: [artifacts/best_imbalance_threshold_comparison.csv](file:///c:/Users/balaj/OneDrive/Desktop/project_1/artifacts/best_imbalance_threshold_comparison.csv)

### Key Methodological Finding:
When a model is trained using balanced undersampling, the training class prior is equalized to 50:50. Consequently, the model's output probabilities are already centered around 0.50. **Unlike the unresampled baseline (which required lowering the threshold to 0.40), the undersampled model naturally achieves its peak Balanced Accuracy (0.6318) right at the standard 0.50 decision threshold.** Lowering the threshold further for an undersampled model severely degrades Class 0 recall (dropping to 28.8% at $\tau = 0.40$).

---

## 12. Accuracy and Recall Trade-offs: Imbalance Handling vs. Threshold Tuning

Comparing the two distinct optimization paradigms:

| Performance Attribute | Phase 4 Baseline ($\tau = 0.50$) | Phase 6 Threshold Tuning ($\tau = 0.40$) | Phase 7 Random Undersampling ($\tau = 0.50$) |
|---|:---:|:---:|:---:|
| **Balanced Accuracy** | 0.6013 | **0.6336** | 0.6318 |
| **Class 1 Recall (Potable)** | 29.65% | 57.83% | **59.39%** |
| **Class 1 Precision** | **66.89%** | 54.32% | 53.48% |
| **Class 1 F1-Score** | 0.4108 | 0.5602 | **0.5628** |
| **Class 0 Recall (Unsafe)** | **90.61%** | 68.90% | 66.96% |
| **Class 0 Precision** | 66.82% | 71.87% | **72.05%** |
| **Overall Accuracy** | **66.83%** | 64.58% | 64.01% |
| **ROC-AUC** | 0.6849 | 0.6849 | 0.6796 |
| **Implementation Complexity** | Simple | Decoupled Post-Processing | Data Resampling Pipeline |

### Honest Technical Assessment:
- **Did overall accuracy improve?** **No.** Overall accuracy declined slightly from 66.83% to 64.01% (-2.82 percentage points). This is expected because catching more minority cases inevitably increases False Positives on the majority class.
- **Did Class 1 recall improve?** **Yes.** Class 1 recall doubled from 29.65% to 59.39% (+29.74 percentage points).
- **Did Balanced Accuracy improve?** **Yes.** Balanced accuracy rose from 60.13% to 63.18% (+3.05 percentage points).
- **Comparison to Threshold Tuning**: Threshold tuning on the unresampled model ($\tau=0.40$) achieved virtually the same operating point (BalAcc 0.6336, C1 Rec 57.83%, F1 0.5602, Acc 64.58%) without discarding 576 majority-class samples.

---

## 13. Limitations & Recommendation for the Next Phase

### Limitations
1. **Precision Reduction**: Catching ~60% of potable samples reduces Class 1 Precision from 66.9% to 53.5%. Approximately 46% of predicted potable samples may actually be non-potable in dataset terms.
2. **Data Discarding in Undersampling**: Random Undersampling discards 576 training samples per fold, slightly decreasing ROC-AUC from 0.6849 to 0.6796.
3. **No Biological Pathogen Data**: As documented throughout, the dataset measures chemical properties only; microbiological pathogens (*E. coli*, cholera, enteroviruses) remain unmeasured.

### Final Recommendation
- **Preserve the current champion model in production**: [models/best_model.joblib](file:///c:/Users/balaj/OneDrive/Desktop/project_1/models/best_model.joblib) with threshold $\tau = 0.40$ remains the cleaner, non-destructive approach because it retains all training data, maintains a higher ROC-AUC (0.6849 vs 0.6796), and achieves slightly higher balanced accuracy (0.6336 vs 0.6318).
- **Experimental Model Archived**: The fitted Random Undersampling pipeline is safely archived at [models/experiments/best_imbalance_model.joblib](file:///c:/Users/balaj/OneDrive/Desktop/project_1/models/experiments/best_imbalance_model.joblib) for research comparison.
- **Streamlit Application Protected**: The production interface remains untouched pending user review.

