"""
Phase 8: Advanced Model Optimization Module.

Project: Water Quality Prediction Using Machine Learning
Objective:
- Systematically evaluate alternative high-capacity algorithms:
    1. Extra Trees Classifier (Extremely Randomized Trees)
    2. HistGradientBoosting Classifier (Histogram-based Gradient Boosting)
    3. XGBoost Classifier (Extreme Gradient Boosting)
    4. Current Production Baseline: Tuned Random Forest (at threshold 0.40)
- Perform hyperparameter optimization using Stratified 5-Fold Cross-Validation strictly on training data (X_train, y_train).
- Pipeline architecture: SimpleImputer(median) -> StandardScaler -> Classifier.
- Evaluate Out-Of-Fold (OOF) predictions across 10 evaluation metrics:
    Accuracy, Balanced Accuracy, Precision C0/C1, Recall C0/C1, F1 C0/C1, ROC-AUC, PR-AUC.
- Perform threshold sensitivity analysis on OOF probabilities across [0.30 - 0.60].
- Save artifacts:
    * artifacts/phase_8_model_comparison.csv
    * artifacts/phase_8_best_parameters.json
    * artifacts/phase_8_model_comparison.png
    * artifacts/phase_8_report.md
    * models/experiments/ (experimental models persisted separately)
- Strict Zero-Leakage & Model Protection:
    * models/best_model.joblib is strictly preserved and not overwritten.
    * artifacts/threshold_config.json is strictly preserved and not overwritten.
    * app.py is untouched.
    * X_test.csv and y_test.csv are strictly isolated and never accessed.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd
import numpy as np

# Configure non-interactive Matplotlib backend for headless execution
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)
import joblib

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_STATE = 42
CV_FOLDS = 5

CANDIDATE_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_training_data(processed_dir: Path = PROCESSED_DATA_DIR) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load strictly training data partition (X_train.csv, y_train.csv).
    Test partition is strictly isolated.
    """
    x_train_path = processed_dir / "X_train.csv"
    y_train_path = processed_dir / "y_train.csv"

    if not x_train_path.is_file() or not y_train_path.is_file():
        raise FileNotFoundError(f"Processed training data not found at {processed_dir}")

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()

    print(f"Loaded training data: {X_train.shape} samples (Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()})", flush=True)
    return X_train, y_train


def define_search_spaces() -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    """
    Define base pipelines and manageable hyperparameter search spaces for candidate models.
    """
    spaces = {
        "Extra Trees": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1))
            ]),
            {
                "classifier__n_estimators": [100, 150, 200],
                "classifier__max_depth": [10, 15, 20, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", "log2"]
            }
        ),
        "HistGradientBoosting": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", HistGradientBoostingClassifier(random_state=RANDOM_STATE))
            ]),
            {
                "classifier__max_iter": [100, 150, 200],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__max_depth": [4, 6, 8, None],
                "classifier__min_samples_leaf": [10, 20, 30],
                "classifier__l2_regularization": [0.0, 0.1, 1.0]
            }
        ),
        "XGBoost": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1))
            ]),
            {
                "classifier__n_estimators": [100, 150, 200],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__max_depth": [3, 5, 7],
                "classifier__subsample": [0.8, 0.9, 1.0],
                "classifier__colsample_bytree": [0.7, 0.8, 1.0],
                "classifier__reg_alpha": [0.0, 0.1, 1.0]
            }
        )
    }
    return spaces


def tune_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
    """
    Perform hyperparameter optimization using RandomizedSearchCV on training data folds only.
    """
    print("\n" + "=" * 80, flush=True)
    print(" " * 18 + "STEP 1: HYPERPARAMETER OPTIMIZATION (5-FOLD CV)", flush=True)
    print("=" * 80, flush=True)

    search_spaces = define_search_spaces()
    tuned_pipelines = {}
    best_params_dict = {}

    for name, (pipe, param_grid) in search_spaces.items():
        print(f"\n[Optimizing] {name}...", flush=True)
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            n_iter=8,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1
        )
        search.fit(X_train, y_train)

        clean_params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
        best_params_dict[name] = clean_params
        tuned_pipelines[name] = search.best_estimator_

        print(f"  * Best CV ROC-AUC : {search.best_score_:.4f}", flush=True)
        print(f"  * Best Parameters : {clean_params}", flush=True)

    # Add existing Production Champion Random Forest Baseline
    # Existing champion hyperparameters: n_estimators=150, max_depth=15, min_samples_split=2, min_samples_leaf=1, max_features='sqrt'
    rf_baseline_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=1
        ))
    ])
    tuned_pipelines["Random Forest (Production Baseline)"] = rf_baseline_pipe
    best_params_dict["Random Forest (Production Baseline)"] = {
        "n_estimators": 150,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    }

    # Save best parameters to artifacts/phase_8_best_parameters.json
    params_path = ARTIFACTS_DIR / "phase_8_best_parameters.json"
    ensure_dir(params_path.parent)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(best_params_dict, f, indent=4)
    print(f"\n  [SAVED] Best hyperparameter configurations -> {params_path.name}", flush=True)

    return tuned_pipelines, best_params_dict


def compute_metrics(y_true: pd.Series, probas: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    """Calculate full set of 10 evaluation metrics at specified threshold."""
    preds = (probas >= threshold).astype(int)

    acc = accuracy_score(y_true, preds)
    bacc = balanced_accuracy_score(y_true, preds)
    prec0 = precision_score(y_true, preds, pos_label=0, zero_division=0)
    rec0 = recall_score(y_true, preds, pos_label=0, zero_division=0)
    f1_0 = f1_score(y_true, preds, pos_label=0, zero_division=0)
    prec1 = precision_score(y_true, preds, pos_label=1, zero_division=0)
    rec1 = recall_score(y_true, preds, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_true, probas)
    pr_auc = average_precision_score(y_true, probas)

    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": round(float(threshold), 2),
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bacc), 4),
        "precision_c0": round(float(prec0), 4),
        "recall_c0": round(float(rec0), 4),
        "f1_c0": round(float(f1_0), 4),
        "precision_c1": round(float(prec1), 4),
        "recall_c1": round(float(rec1), 4),
        "f1_c1": round(float(f1_1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def evaluate_models_and_thresholds(
    tuned_pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
    """
    Step 2: Generate Out-Of-Fold probabilities for all models and evaluate:
    - Default threshold 0.50
    - Threshold sensitivity across [0.30 - 0.60]
    """
    print("\n" + "=" * 80, flush=True)
    print(" " * 15 + "STEP 2: OUT-OF-FOLD EVALUATION & THRESHOLD OPTIMIZATION", flush=True)
    print("=" * 80, flush=True)

    oof_probabilities = {}
    comparison_records = []
    threshold_tables = {}

    for name, pipe in tuned_pipelines.items():
        print(f"\nComputing 5-Fold OOF probabilities for: {name}...", flush=True)
        probas = cross_val_predict(
            pipe,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1
        )[:, 1]
        oof_probabilities[name] = probas

        # 1. Performance at default threshold 0.50
        m_def = compute_metrics(y_train, probas, threshold=0.50)

        # 2. Threshold sensitivity analysis across [0.30 - 0.60]
        thresh_records = []
        for t in CANDIDATE_THRESHOLDS:
            m_t = compute_metrics(y_train, probas, threshold=t)
            thresh_records.append(m_t)
        thresh_df = pd.DataFrame(thresh_records)
        threshold_tables[name] = thresh_df

        # Find optimal threshold for this model (maximizing balanced accuracy)
        best_t_row = thresh_df.loc[thresh_df["balanced_accuracy"].idxmax()]
        opt_thresh = float(best_t_row["threshold"])

        print(f"  * Default 0.50 : Acc={m_def['accuracy']:.4f} | BalAcc={m_def['balanced_accuracy']:.4f} | C1 Rec={m_def['recall_c1']:.4f} | C1 Prec={m_def['precision_c1']:.4f} | C1 F1={m_def['f1_c1']:.4f} | ROC-AUC={m_def['roc_auc']:.4f}", flush=True)
        print(f"  * Optimal tau={opt_thresh:.2f}: Acc={best_t_row['accuracy']:.4f} | BalAcc={best_t_row['balanced_accuracy']:.4f} | C1 Rec={best_t_row['recall_c1']:.4f} | C1 Prec={best_t_row['precision_c1']:.4f} | C1 F1={best_t_row['f1_c1']:.4f}", flush=True)

        comparison_records.append({
            "Model": name,
            "Default_Threshold": 0.50,
            "Accuracy_at_0.50": m_def["accuracy"],
            "Balanced_Accuracy_at_0.50": m_def["balanced_accuracy"],
            "Recall_C1_at_0.50": m_def["recall_c1"],
            "Precision_C1_at_0.50": m_def["precision_c1"],
            "F1_C1_at_0.50": m_def["f1_c1"],
            "Recall_C0_at_0.50": m_def["recall_c0"],
            "Precision_C0_at_0.50": m_def["precision_c0"],
            "ROC_AUC": m_def["roc_auc"],
            "PR_AUC": m_def["pr_auc"],
            "Optimal_Threshold": opt_thresh,
            "Accuracy_at_Optimal": best_t_row["accuracy"],
            "Balanced_Accuracy_at_Optimal": best_t_row["balanced_accuracy"],
            "Recall_C1_at_Optimal": best_t_row["recall_c1"],
            "Precision_C1_at_Optimal": best_t_row["precision_c1"],
            "F1_C1_at_Optimal": best_t_row["f1_c1"],
            "Recall_C0_at_Optimal": best_t_row["recall_c0"]
        })

    comp_df = pd.DataFrame(comparison_records)

    # Save comparison CSV
    csv_path = ARTIFACTS_DIR / "phase_8_model_comparison.csv"
    ensure_dir(csv_path.parent)
    comp_df.to_csv(csv_path, index=False)
    print(f"\n  [SAVED] Phase 8 model comparison table -> {csv_path.name}", flush=True)

    return comp_df, oof_probabilities, threshold_tables


def generate_comparison_plots(
    comp_df: pd.DataFrame,
    artifacts_dir: Path = ARTIFACTS_DIR
):
    """Generate comparative visualization across models for default and optimal thresholds."""
    ensure_dir(artifacts_dir)
    save_path = artifacts_dir / "phase_8_model_comparison.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    models_clean = [
        "Extra Trees",
        "HistGradientBoosting",
        "XGBoost",
        "Random Forest (Baseline)"
    ]

    # Panel 1: ROC-AUC and PR-AUC (Threshold-Independent)
    ax1 = axes[0, 0]
    x = np.arange(len(models_clean))
    width = 0.35
    ax1.bar(x - width/2, comp_df["ROC_AUC"], width, label="ROC-AUC", color="#3498db", edgecolor="#1b4f72")
    ax1.bar(x + width/2, comp_df["PR_AUC"], width, label="PR-AUC", color="#9b59b6", edgecolor="#4a235a")
    for i in x:
        ax1.text(i - width/2, comp_df["ROC_AUC"].iloc[i] + 0.01, f"{comp_df['ROC_AUC'].iloc[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax1.text(i + width/2, comp_df["PR_AUC"].iloc[i] + 0.01, f"{comp_df['PR_AUC'].iloc[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_title("A. Threshold-Independent Ranking (ROC-AUC & PR-AUC)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_clean, fontsize=9.5)
    ax1.set_ylim(0.40, 0.78)
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Balanced Accuracy (Default 0.50 vs Optimal Threshold)
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, comp_df["Balanced_Accuracy_at_0.50"], width, label="At Default 0.50", color="#95a5a6", edgecolor="#566573")
    ax2.bar(x + width/2, comp_df["Balanced_Accuracy_at_Optimal"], width, label="At Optimal Threshold", color="#27ae60", edgecolor="#145a32")
    for i in x:
        ax2.text(i - width/2, comp_df["Balanced_Accuracy_at_0.50"].iloc[i] + 0.01, f"{comp_df['Balanced_Accuracy_at_0.50'].iloc[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax2.text(i + width/2, comp_df["Balanced_Accuracy_at_Optimal"].iloc[i] + 0.01, f"{comp_df['Balanced_Accuracy_at_Optimal'].iloc[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_title("B. Balanced Accuracy: Default (0.50) vs. Optimal Threshold", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_clean, fontsize=9.5)
    ax2.set_ylim(0.45, 0.72)
    ax2.legend(loc="upper left")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 3: Class 1 (Potable) Recall: Default vs Optimal
    ax3 = axes[1, 0]
    ax3.bar(x - width/2, comp_df["Recall_C1_at_0.50"], width, label="Recall C1 (Default 0.50)", color="#e74c3c", edgecolor="#78281f")
    ax3.bar(x + width/2, comp_df["Recall_C1_at_Optimal"], width, label="Recall C1 (Optimal Threshold)", color="#2ecc71", edgecolor="#145a32")
    for i in x:
        ax3.text(i - width/2, comp_df["Recall_C1_at_0.50"].iloc[i] + 0.015, f"{comp_df['Recall_C1_at_0.50'].iloc[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax3.text(i + width/2, comp_df["Recall_C1_at_Optimal"].iloc[i] + 0.015, f"{comp_df['Recall_C1_at_Optimal'].iloc[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax3.set_title("C. Minority Class 1 (Potable) Recall Improvement", fontsize=11, fontweight="bold", pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models_clean, fontsize=9.5)
    ax3.set_ylim(0.15, 0.75)
    ax3.legend(loc="upper left")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 4: Class 1 F1-Score: Default vs Optimal
    ax4 = axes[1, 1]
    ax4.bar(x - width/2, comp_df["F1_C1_at_0.50"], width, label="F1 C1 (Default 0.50)", color="#f39c12", edgecolor="#7e5109")
    ax4.bar(x + width/2, comp_df["F1_C1_at_Optimal"], width, label="F1 C1 (Optimal Threshold)", color="#16a085", edgecolor="#0e6251")
    for i in x:
        ax4.text(i - width/2, comp_df["F1_C1_at_0.50"].iloc[i] + 0.012, f"{comp_df['F1_C1_at_0.50'].iloc[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax4.text(i + width/2, comp_df["F1_C1_at_Optimal"].iloc[i] + 0.012, f"{comp_df['F1_C1_at_Optimal'].iloc[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax4.set_title("D. Minority Class 1 (Potable) F1-Score Improvement", fontsize=11, fontweight="bold", pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(models_clean, fontsize=9.5)
    ax4.set_ylim(0.25, 0.65)
    ax4.legend(loc="upper left")
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Phase 8: Advanced Model Optimization Benchmark (5-Fold CV Out-Of-Fold, N=2,620)", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [SAVED] Phase 8 comparison plot -> {save_path.name}", flush=True)


def persist_experimental_models(
    tuned_pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    experiments_dir: Path = EXPERIMENTS_DIR
):
    """
    Fit each candidate pipeline on the full training set and persist under models/experiments/.
    models/best_model.joblib is strictly preserved and untouched.
    """
    ensure_dir(experiments_dir)
    print("\n" + "=" * 80, flush=True)
    print(f"STEP 3: PERSISTING EXPERIMENTAL MODELS TO {experiments_dir}", flush=True)
    print("=" * 80, flush=True)

    filenames = {
        "Extra Trees": "extra_trees_tuned.joblib",
        "HistGradientBoosting": "hist_gradient_boosting_tuned.joblib",
        "XGBoost": "xgboost_phase8_tuned.joblib"
    }

    for name, filename in filenames.items():
        pipe = tuned_pipelines[name]
        print(f"Fitting and saving {name} -> {filename}...", flush=True)
        pipe.fit(X_train, y_train)
        target_path = experiments_dir / filename
        joblib.dump(pipe, target_path)
        print(f"  * Persisted: {target_path.name}", flush=True)


def run_phase_8_optimization() -> Dict[str, Any]:
    """Orchestrate Phase 8 execution."""
    ensure_dir(ARTIFACTS_DIR)
    ensure_dir(EXPERIMENTS_DIR)

    # 1. Load data
    X_train, y_train = load_training_data()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 2. Tune candidate models
    tuned_pipelines, best_params = tune_candidate_models(X_train, y_train, cv)

    # 3. Evaluate models across OOF probabilities and thresholds
    comp_df, oof_probabilities, threshold_tables = evaluate_models_and_thresholds(
        tuned_pipelines, X_train, y_train, cv
    )

    # 4. Generate comparison plots
    generate_comparison_plots(comp_df)

    # 5. Persist experimental models
    persist_experimental_models(tuned_pipelines, X_train, y_train)

    print("\n" + "=" * 80, flush=True)
    print(" " * 22 + "PHASE 8 OPTIMIZATION COMPLETE", flush=True)
    print("=" * 80 + "\n", flush=True)

    return {
        "comparison_table": comp_df.to_dict(orient="records"),
        "best_parameters": best_params,
        "threshold_tables": {k: v.to_dict(orient="records") for k, v in threshold_tables.items()}
    }


def main():
    run_phase_8_optimization()


if __name__ == "__main__":
    main()
