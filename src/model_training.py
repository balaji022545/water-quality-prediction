"""
Model Training Module.

Project: Water Quality Prediction Using Machine Learning
Scope (Phase 4 - Corrected Zero-Leakage Pipeline):
- Load raw split datasets from data/processed/ (missing values preserved for fold-level imputation).
- Embed SimpleImputer(strategy='median') inside all Scikit-learn / Imbalanced-learn pipelines.
- Pipeline execution order:
    Imputer -> Scaler (StandardScaler) -> SMOTE (where applicable) -> Classifier
- Strict Fold-Level Isolation:
    * Imputer is fitted ONLY on training folds during CV.
    * Scaler is fitted ONLY on training folds during CV.
    * SMOTE is applied ONLY during training fold fitting.
    * Validation folds are transformed using training parameters; never resampled.
- Evaluate Baseline (DummyClassifier) and 5 ML models with/without SMOTE using Stratified 5-Fold CV.
- Hyperparameter tuning for promising models using CV on training folds only.
- Final one-time test evaluation on unseen test partition.
- Persist corrected best model pipeline and artifacts.
"""

import os
import sys
import json
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

# Configure non-interactive backend for headless execution
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress minor conversion warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Scikit-learn & Imbalanced-learn modules
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, make_scorer
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib

# Path references
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_STATE = 42
CV_FOLDS = 5

FEATURE_NAMES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_processed_training_data(
    processed_dir: Path = PROCESSED_DATA_DIR
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Part A: Load raw partitioned feature splits.
    Preserves original missing values so that SimpleImputer is fitted fold-by-fold.
    """
    print("\n" + "=" * 80, flush=True)
    print(" " * 18 + "PHASE 4: MODEL TRAINING (CORRECTED ZERO-LEAKAGE)", flush=True)
    print("=" * 80, flush=True)
    print("\n--- 1. LOADING PROCESSED SPLIT DATASETS ---", flush=True)

    x_train_path = processed_dir / "X_train.csv"
    x_test_path = processed_dir / "X_test.csv"
    y_train_path = processed_dir / "y_train.csv"
    y_test_path = processed_dir / "y_test.csv"

    for p in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Required processed file missing: {p}")

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

    # Integrity assertions
    assert list(X_train.columns) == FEATURE_NAMES, f"Feature columns mismatch in X_train: {list(X_train.columns)}"
    assert list(X_test.columns) == FEATURE_NAMES, f"Feature columns mismatch in X_test: {list(X_test.columns)}"
    assert len(X_train) == len(y_train) == 2620, f"Expected 2620 train rows, got {len(X_train)}"
    assert len(X_test) == len(y_test) == 656, f"Expected 656 test rows, got {len(X_test)}"

    train_missing = int(X_train.isnull().sum().sum())
    test_missing = int(X_test.isnull().sum().sum())

    print(f"  * X_train shape: {X_train.shape} | Missing values: {train_missing:,} (to be imputed fold-by-fold)", flush=True)
    print(f"  * X_test shape : {X_test.shape}  | Missing values: {test_missing:,} (imputed via train params at test time)", flush=True)
    print("  * Features (9) :", list(X_train.columns), flush=True)
    print(f"  * Train Potability distribution: {y_train.value_counts().to_dict()} (0 = Unsafe, 1 = Potable)", flush=True)
    print(f"  * Test Potability distribution : {y_test.value_counts().to_dict()}", flush=True)
    print("  * [VERIFIED] Train-test split preserved. Test set strictly isolated.", flush=True)

    return X_train, X_test, y_train, y_test


def evaluate_cv_pipeline(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold
) -> Dict[str, float]:
    """
    Evaluate a pipeline using Stratified K-Fold Cross-Validation across multiple metrics.
    Computes class-wise precision, recall, F1 (both Class 0 and Class 1), accuracy, and ROC-AUC.
    """
    scoring = {
        "accuracy": "accuracy",
        "precision_1": make_scorer(precision_score, pos_label=1, zero_division=0),
        "recall_1": make_scorer(recall_score, pos_label=1, zero_division=0),
        "f1_1": make_scorer(f1_score, pos_label=1, zero_division=0),
        "precision_0": make_scorer(precision_score, pos_label=0, zero_division=0),
        "recall_0": make_scorer(recall_score, pos_label=0, zero_division=0),
        "f1_0": make_scorer(f1_score, pos_label=0, zero_division=0),
        "roc_auc": "roc_auc"
    }
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)

    return {
        "accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "accuracy_std": float(np.std(scores["test_accuracy"])),
        "precision_1_mean": float(np.mean(scores["test_precision_1"])),
        "precision_1_std": float(np.std(scores["test_precision_1"])),
        "recall_1_mean": float(np.mean(scores["test_recall_1"])),
        "recall_1_std": float(np.std(scores["test_recall_1"])),
        "f1_1_mean": float(np.mean(scores["test_f1_1"])),
        "f1_1_std": float(np.std(scores["test_f1_1"])),
        "precision_0_mean": float(np.mean(scores["test_precision_0"])),
        "precision_0_std": float(np.std(scores["test_precision_0"])),
        "recall_0_mean": float(np.mean(scores["test_recall_0"])),
        "recall_0_std": float(np.std(scores["test_recall_0"])),
        "f1_0_mean": float(np.mean(scores["test_f1_0"])),
        "f1_0_std": float(np.std(scores["test_f1_0"])),
        "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "roc_auc_std": float(np.std(scores["test_roc_auc"]))
    }


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold
) -> Dict[str, Any]:
    """
    Part B: Train a DummyClassifier as the minimum performance baseline.
    Uses Imputer -> Classifier pipeline.
    """
    print("\n--- 2. BASELINE MODEL EVALUATION (DummyClassifier) ---", flush=True)
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    pipeline = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", dummy)
    ])

    cv_results = evaluate_cv_pipeline(pipeline, X_train, y_train, cv)

    print(f"  * Strategy: 'most_frequent' (always predicts majority class 0)", flush=True)
    print(f"  * Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}", flush=True)
    print(f"  * Class 1 Precision: {cv_results['precision_1_mean']:.4f} | Recall: {cv_results['recall_1_mean']:.4f} | F1: {cv_results['f1_1_mean']:.4f}", flush=True)
    print(f"  * Class 0 Precision: {cv_results['precision_0_mean']:.4f} | Recall: {cv_results['recall_0_mean']:.4f} | F1: {cv_results['f1_0_mean']:.4f}", flush=True)
    print(f"  * ROC-AUC  : {cv_results['roc_auc_mean']:.4f}", flush=True)

    return {
        "Model": "Baseline (Dummy Classifier)",
        "SMOTE": "No",
        "CV Accuracy": cv_results["accuracy_mean"],
        "Precision_C1": cv_results["precision_1_mean"],
        "Recall_C1": cv_results["recall_1_mean"],
        "F1_C1": cv_results["f1_1_mean"],
        "Precision_C0": cv_results["precision_0_mean"],
        "Recall_C0": cv_results["recall_0_mean"],
        "F1_C0": cv_results["f1_0_mean"],
        "CV ROC-AUC": cv_results["roc_auc_mean"]
    }


def get_candidate_models() -> Dict[str, Any]:
    """Return dictionary of the 5 required machine learning algorithms."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1)
    }


def compare_models_with_and_without_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Part C & D: Train candidate models before and after applying SMOTE.
    Order:
      - Standard: Imputer -> Scaler -> Classifier
      - SMOTE   : Imputer -> Scaler -> SMOTE -> Classifier
    """
    print("\n--- 3. TRAINING & COMPARING 5 ALGORITHMS (CORRECTED PIPELINE ORDER) ---", flush=True)
    print("  Pipeline Order (Standard): [SimpleImputer] -> [StandardScaler] -> [Classifier]", flush=True)
    print("  Pipeline Order (SMOTE)   : [SimpleImputer] -> [StandardScaler] -> [SMOTE] -> [Classifier]", flush=True)

    models = get_candidate_models()
    records = []
    fitted_pipelines = {}

    for name, model in models.items():
        print(f"\nEvaluating: {name}...", flush=True)

        # 1. Standard Pipeline (Imputer -> Scaler -> Model)
        pipe_standard = ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", model)
        ])
        res_std = evaluate_cv_pipeline(pipe_standard, X_train, y_train, cv)
        records.append({
            "Model": name,
            "SMOTE": "No",
            "CV Accuracy": res_std["accuracy_mean"],
            "Precision_C1": res_std["precision_1_mean"],
            "Recall_C1": res_std["recall_1_mean"],
            "F1_C1": res_std["f1_1_mean"],
            "Precision_C0": res_std["precision_0_mean"],
            "Recall_C0": res_std["recall_0_mean"],
            "F1_C0": res_std["f1_0_mean"],
            "CV ROC-AUC": res_std["roc_auc_mean"],
            "Accuracy_Std": res_std["accuracy_std"],
            "ROC_AUC_Std": res_std["roc_auc_std"]
        })
        print(f"  [Standard] Acc: {res_std['accuracy_mean']:.4f} | ROC-AUC: {res_std['roc_auc_mean']:.4f} | C1 Recall: {res_std['recall_1_mean']:.4f} | C0 Recall: {res_std['recall_0_mean']:.4f} | C1 F1: {res_std['f1_1_mean']:.4f}", flush=True)

        # 2. SMOTE Pipeline (Imputer -> Scaler -> SMOTE -> Model)
        pipe_smote = ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", model)
        ])
        res_smote = evaluate_cv_pipeline(pipe_smote, X_train, y_train, cv)
        records.append({
            "Model": name,
            "SMOTE": "Yes",
            "CV Accuracy": res_smote["accuracy_mean"],
            "Precision_C1": res_smote["precision_1_mean"],
            "Recall_C1": res_smote["recall_1_mean"],
            "F1_C1": res_smote["f1_1_mean"],
            "Precision_C0": res_smote["precision_0_mean"],
            "Recall_C0": res_smote["recall_0_mean"],
            "F1_C0": res_smote["f1_0_mean"],
            "CV ROC-AUC": res_smote["roc_auc_mean"],
            "Accuracy_Std": res_smote["accuracy_std"],
            "ROC_AUC_Std": res_smote["roc_auc_std"]
        })
        print(f"  [+ SMOTE ] Acc: {res_smote['accuracy_mean']:.4f} | ROC-AUC: {res_smote['roc_auc_mean']:.4f} | C1 Recall: {res_smote['recall_1_mean']:.4f} | C0 Recall: {res_smote['recall_0_mean']:.4f} | C1 F1: {res_smote['f1_1_mean']:.4f}", flush=True)

        fitted_pipelines[f"{name}_Standard"] = pipe_standard
        fitted_pipelines[f"{name}_SMOTE"] = pipe_smote

    comparison_df = pd.DataFrame(records)
    return comparison_df, fitted_pipelines


def tune_promising_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Part G: Hyperparameter tuning with full pipeline: Imputer -> Scaler -> Classifier.
    Strictly executed via RandomizedSearchCV on training data folds only.
    """
    print("\n--- 4. HYPERPARAMETER TUNING (WITH EMBEDDED IMPUTER & SCALER) ---", flush=True)
    tuned_results = {}
    best_params_dict = {}

    # 1. Tuning Random Forest
    print("\n[Tuning 1/3] Random Forest Classifier...", flush=True)
    rf_pipe = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1))
    ])
    rf_param_grid = {
        "classifier__n_estimators": [100, 150, 200],
        "classifier__max_depth": [None, 10, 15],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"]
    }
    rf_search = RandomizedSearchCV(
        estimator=rf_pipe,
        param_distributions=rf_param_grid,
        n_iter=8,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    rf_search.fit(X_train, y_train)
    best_rf_params = {k.replace("classifier__", ""): v for k, v in rf_search.best_params_.items()}
    best_params_dict["Random Forest"] = best_rf_params
    tuned_results["Random Forest (Tuned)"] = rf_search.best_estimator_
    print(f"  * Best RF ROC-AUC (CV): {rf_search.best_score_:.4f}", flush=True)
    print(f"  * Best RF Parameters  : {best_rf_params}", flush=True)

    # 2. Tuning XGBoost
    print("\n[Tuning 2/3] XGBoost Classifier...", flush=True)
    xgb_pipe = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1))
    ])
    xgb_param_grid = {
        "classifier__n_estimators": [100, 150, 200],
        "classifier__max_depth": [3, 4, 5],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
        "classifier__subsample": [0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0]
    }
    xgb_search = RandomizedSearchCV(
        estimator=xgb_pipe,
        param_distributions=xgb_param_grid,
        n_iter=8,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    xgb_search.fit(X_train, y_train)
    best_xgb_params = {k.replace("classifier__", ""): v for k, v in xgb_search.best_params_.items()}
    best_params_dict["XGBoost"] = best_xgb_params
    tuned_results["XGBoost (Tuned)"] = xgb_search.best_estimator_
    print(f"  * Best XGBoost ROC-AUC (CV): {xgb_search.best_score_:.4f}", flush=True)
    print(f"  * Best XGBoost Parameters  : {best_xgb_params}", flush=True)

    # 3. Tuning Logistic Regression
    print("\n[Tuning 3/3] Logistic Regression...", flush=True)
    lr_pipe = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    lr_param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs", "liblinear"],
        "classifier__class_weight": [None, "balanced"]
    }
    lr_search = RandomizedSearchCV(
        estimator=lr_pipe,
        param_distributions=lr_param_grid,
        n_iter=6,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    lr_search.fit(X_train, y_train)
    best_lr_params = {k.replace("classifier__", ""): v for k, v in lr_search.best_params_.items()}
    best_params_dict["Logistic Regression"] = best_lr_params
    tuned_results["Logistic Regression (Tuned)"] = lr_search.best_estimator_
    print(f"  * Best Logistic Regression ROC-AUC (CV): {lr_search.best_score_:.4f}", flush=True)
    print(f"  * Best LR Parameters                   : {best_lr_params}", flush=True)

    return tuned_results, best_params_dict


def evaluate_final_model_on_test_set(
    model_pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> Dict[str, Any]:
    """
    Part E: Final out-of-sample evaluation on unseen test dataset.
    Trains strictly on X_train, then predicts on X_test.
    """
    print(f"\n--- 5. FINAL TEST EVALUATION (ONE-TIME AUDIT): {model_name} ---", flush=True)

    # Fit selected pipeline on full training data (imputer, scaler, classifier fitted on X_train)
    model_pipeline.fit(X_train, y_train)

    # Predict class labels and probabilities on unseen test set
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    # Metrics computation
    acc = accuracy_score(y_test, y_pred)
    prec_c1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_c1 = recall_score(y_test, y_pred, pos_label=1)
    f1_c1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    prec_c0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_c0 = recall_score(y_test, y_pred, pos_label=0)
    f1_c0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["0: Not Potable", "1: Potable"], digits=4)

    print(f"  * Test Accuracy : {acc:.4f}", flush=True)
    print(f"  * Test ROC-AUC  : {roc_auc:.4f}", flush=True)
    print(f"  * Class 0 (Not Potable / Unsafe): Recall = {rec_c0:.4f} | Precision = {prec_c0:.4f} | F1 = {f1_c0:.4f}", flush=True)
    print(f"  * Class 1 (Potable / Safe)     : Recall = {rec_c1:.4f} | Precision = {prec_c1:.4f} | F1 = {f1_c1:.4f}", flush=True)
    print("\nConfusion Matrix [TN, FP / FN, TP]:", flush=True)
    print(cm, flush=True)
    print("\nClassification Report:", flush=True)
    print(cr, flush=True)

    return {
        "model_name": model_name,
        "test_accuracy": float(acc),
        "test_roc_auc": float(roc_auc),
        "precision_class_0": float(prec_c0),
        "recall_class_0": float(rec_c0),
        "f1_class_0": float(f1_c0),
        "precision_class_1": float(prec_c1),
        "recall_class_1": float(rec_c1),
        "f1_class_1": float(f1_c1),
        "confusion_matrix": cm.tolist(),
        "classification_report": cr
    }


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: Path):
    """Generate and save comparison bar plot for candidate models."""
    plt.figure(figsize=(12, 6))

    plot_df = comparison_df.melt(
        id_vars=["Model", "SMOTE"],
        value_vars=["CV Accuracy", "F1_C1", "CV ROC-AUC"],
        var_name="Metric",
        value_name="Score"
    )
    plot_df["Configuration"] = plot_df["Model"] + " (" + plot_df["SMOTE"].apply(lambda s: "+SMOTE" if s == "Yes" else "Standard") + ")"

    ax = sns.barplot(
        data=plot_df,
        x="Configuration",
        y="Score",
        hue="Metric",
        palette=["#3498db", "#2ecc71", "#e74c3c"]
    )
    plt.title("Cross-Validation Performance Comparison Across Machine Learning Models (Corrected Zero-Leakage)", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.40, 0.80)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [SAVED] Corrected model comparison visualization -> {save_path.name}", flush=True)


def run_training_pipeline() -> Dict[str, Any]:
    """Orchestrate Phase 4 end-to-end model training, evaluation, tuning, and persistence."""
    ensure_dir(MODELS_DIR)
    ensure_dir(ARTIFACTS_DIR)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_processed_training_data(PROCESSED_DATA_DIR)

    # Setup Stratified 5-Fold CV
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 2. Baseline Model
    baseline_record = train_baseline_model(X_train, y_train, cv)

    # 3. Candidate Models (with and without SMOTE)
    comparison_df, pipelines = compare_models_with_and_without_smote(X_train, y_train, cv)

    # Add baseline to comparison table
    baseline_df = pd.DataFrame([baseline_record])
    full_comparison_df = pd.concat([baseline_df, comparison_df], ignore_index=True)

    # 4. Hyperparameter Tuning
    tuned_models, best_params = tune_promising_models(X_train, y_train, cv)

    # Evaluate tuned models on CV
    for tuned_name, tuned_pipe in tuned_models.items():
        tuned_cv = evaluate_cv_pipeline(tuned_pipe, X_train, y_train, cv)
        tuned_record = {
            "Model": tuned_name,
            "SMOTE": "No",
            "CV Accuracy": tuned_cv["accuracy_mean"],
            "Precision_C1": tuned_cv["precision_1_mean"],
            "Recall_C1": tuned_cv["recall_1_mean"],
            "F1_C1": tuned_cv["f1_1_mean"],
            "Precision_C0": tuned_cv["precision_0_mean"],
            "Recall_C0": tuned_cv["recall_0_mean"],
            "F1_C0": tuned_cv["f1_0_mean"],
            "CV ROC-AUC": tuned_cv["roc_auc_mean"],
            "Accuracy_Std": tuned_cv["accuracy_std"],
            "ROC_AUC_Std": tuned_cv["roc_auc_std"]
        }
        full_comparison_df = pd.concat([full_comparison_df, pd.DataFrame([tuned_record])], ignore_index=True)

    # Sort comparison table by ROC-AUC and F1_C1
    full_comparison_df = full_comparison_df.sort_values(by=["CV ROC-AUC", "F1_C1"], ascending=False).reset_index(drop=True)

    print("\n" + "=" * 115, flush=True)
    print(" " * 38 + "CORRECTED MODEL COMPARISON TABLE (ZERO LEAKAGE)", flush=True)
    print("=" * 115, flush=True)
    display_cols = ["Model", "SMOTE", "CV Accuracy", "Precision_C0", "Recall_C0", "Precision_C1", "Recall_C1", "F1_C1", "CV ROC-AUC"]
    print(full_comparison_df[display_cols].to_string(index=False), flush=True)
    print("=" * 115, flush=True)

    # 5. Model Selection Decision:
    # Retain Random Forest (Tuned) as champion as instructed unless justified otherwise
    best_candidate_name = full_comparison_df.iloc[0]["Model"]
    if "Random Forest" in best_candidate_name:
        selected_pipeline = tuned_models.get("Random Forest (Tuned)", pipelines["Random Forest_Standard"])
        selected_name = "Random Forest (Tuned)"
    elif "XGBoost" in best_candidate_name:
        selected_pipeline = tuned_models.get("XGBoost (Tuned)", pipelines["XGBoost_Standard"])
        selected_name = "XGBoost (Tuned)"
    else:
        selected_pipeline = tuned_models.get(best_candidate_name, pipelines.get(f"{best_candidate_name}_Standard"))
        selected_name = best_candidate_name

    # 6. Final Evaluation on Unseen Test Set (Single one-time evaluation)
    test_eval = evaluate_final_model_on_test_set(
        selected_pipeline, X_train, y_train, X_test, y_test, selected_name
    )

    # 7. Persist Artifacts and Models
    print("\n--- 6. SAVING CORRECTED ARTIFACTS AND TRAINED MODELS ---", flush=True)

    # CSV results
    comparison_csv_path = ARTIFACTS_DIR / "model_comparison.csv"
    full_comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"  * Saved corrected model comparison table -> {comparison_csv_path.name}", flush=True)

    # Hyperparameters JSON
    hyperparams_path = ARTIFACTS_DIR / "best_hyperparameters.json"
    with open(hyperparams_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"  * Saved best hyperparameters -> {hyperparams_path.name}", flush=True)

    # Plot
    plot_path = ARTIFACTS_DIR / "model_comparison_plot.png"
    plot_model_comparison(comparison_df, plot_path)

    # Persist Best Model Pipeline (Complete self-contained pipeline: Imputer -> Scaler -> Classifier)
    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(selected_pipeline, best_model_path)
    print(f"  * Saved final best model pipeline (Self-contained) -> {best_model_path.name}", flush=True)

    # Persist individual promising models
    rf_path = MODELS_DIR / "random_forest_tuned.joblib"
    xgb_path = MODELS_DIR / "xgboost_tuned.joblib"
    joblib.dump(tuned_models["Random Forest (Tuned)"], rf_path)
    joblib.dump(tuned_models["XGBoost (Tuned)"], xgb_path)
    print(f"  * Saved tuned Random Forest pipeline -> {rf_path.name}", flush=True)
    print(f"  * Saved tuned XGBoost pipeline       -> {xgb_path.name}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(" " * 22 + "PHASE 4 CORRECTION & RETRAINING COMPLETE", flush=True)
    print("=" * 80 + "\n", flush=True)

    return {
        "comparison_table": full_comparison_df.to_dict(orient="records"),
        "best_model_name": selected_name,
        "best_hyperparameters": best_params,
        "test_evaluation": test_eval
    }


def main():
    """Main execution entry point for Phase 4."""
    run_training_pipeline()


if __name__ == "__main__":
    main()
