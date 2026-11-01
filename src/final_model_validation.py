"""
Phase 9: Final Model Validation and Selection Module.

Project: Water Quality Prediction Using Machine Learning
Objective:
- Perform rigorous nested cross-validation to compare:
    1. Existing Production Champion: Tuned Random Forest (Fixed Configuration)
    2. Phase 8 Challenger: Tuned Extra Trees (Nested CV with Inner CV Tuning)
    3. Phase 8 Challenger Reference: Tuned Extra Trees (Fixed Phase 8 Configuration)
- Data Governance:
    * Evaluated strictly on the training partition (X_train.csv, y_train.csv, N=2,620).
    * Zero access to holdout test partition (X_test.csv, y_test.csv).
- Cross-Validation Scheme:
    * Outer CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    * Inner CV: StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    * Preprocessing: SimpleImputer(median) -> StandardScaler -> Classifier fitted inside training folds.
- Decision Cutoff:
    * Fixed threshold of 0.40 for all models (no threshold optimization on outer folds).
- Deliverables:
    * src/final_model_validation.py
    * artifacts/phase_9_nested_cv_results.csv
    * artifacts/phase_9_model_comparison.png
    * artifacts/phase_9_confusion_matrices.png
    * artifacts/phase_9_report.md
    * Experimental model in models/experiments/phase_9_extra_trees_final.joblib
- Production Protection:
    * models/best_model.joblib is preserved untouched.
    * artifacts/threshold_config.json is preserved untouched.
    * app.py is preserved untouched.
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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from scipy import stats
import joblib

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 3
FIXED_THRESHOLD = 0.40


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_training_data(processed_dir: Path = PROCESSED_DATA_DIR) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load strictly the training data partition (X_train.csv, y_train.csv).
    Test partition is strictly isolated.
    """
    x_train_path = processed_dir / "X_train.csv"
    y_train_path = processed_dir / "y_train.csv"

    if not x_train_path.is_file() or not y_train_path.is_file():
        raise FileNotFoundError(f"Processed training data not found at {processed_dir}")

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()

    n_c0 = int((y_train == 0).sum())
    n_c1 = int((y_train == 1).sum())
    print(f"Loaded training data: {X_train.shape} samples (Class 0: {n_c0}, Class 1: {n_c1})", flush=True)
    return X_train, y_train


def calculate_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = FIXED_THRESHOLD) -> Dict[str, float]:
    """Calculate 10 evaluation metrics given true labels and Class 1 probabilities."""
    y_pred = (y_proba >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    prec_c1 = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_c1 = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_c1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    prec_c0 = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_c0 = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_c0 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except Exception:
        pr_auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "Accuracy": acc,
        "Balanced_Accuracy": bal_acc,
        "Precision_C1": prec_c1,
        "Recall_C1": rec_c1,
        "F1_C1": f1_c1,
        "Precision_C0": prec_c0,
        "Recall_C0": rec_c0,
        "F1_C0": f1_c0,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


def run_nested_cross_validation(X: pd.DataFrame, y: pd.Series):
    """
    Execute 5-Fold Outer CV with 3-Fold Inner CV for Extra Trees.
    Compare against Production Random Forest (fixed configuration)
    and Phase 8 Fixed Extra Trees configuration.
    """
    print("\n" + "=" * 80, flush=True)
    print(" " * 18 + "PHASE 9: NESTED CROSS-VALIDATION VALIDATION", flush=True)
    print("=" * 80, flush=True)
    print(f"Outer CV: 5-Fold StratifiedKFold (shuffle=True, random_state={RANDOM_STATE})", flush=True)
    print(f"Inner CV: 3-Fold StratifiedKFold (shuffle=True, random_state={RANDOM_STATE})", flush=True)
    print(f"Fixed Evaluation Threshold: {FIXED_THRESHOLD:.2f}", flush=True)

    outer_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Search space for Extra Trees in inner CV
    et_param_grid = {
        "classifier__n_estimators": [100, 150],
        "classifier__max_depth": [15, None],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
        "classifier__max_features": ["sqrt", "log2"]
    }

    # Tracking storage
    models = ["Random Forest (Production)", "Extra Trees (Nested CV)", "Extra Trees (Phase 8 Fixed)"]
    fold_metrics = {m: [] for m in models}
    oof_probabilities = {m: np.zeros(len(y)) for m in models}
    best_params_per_fold = []

    for fold_num, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        print(f"\n--- Outer Fold {fold_num}/{OUTER_SPLITS} ---", flush=True)
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        # ---------------------------------------------------------------------
        # 1. Random Forest (Production Configuration)
        # ---------------------------------------------------------------------
        rf_pipe = Pipeline([
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
        rf_pipe.fit(X_train_fold, y_train_fold)
        rf_proba = rf_pipe.predict_proba(X_val_fold)[:, 1]
        oof_probabilities["Random Forest (Production)"][val_idx] = rf_proba
        rf_res = calculate_metrics(y_val_fold.values, rf_proba, threshold=FIXED_THRESHOLD)
        rf_res["Fold"] = fold_num
        fold_metrics["Random Forest (Production)"].append(rf_res)
        print(f"  [RF Prod]        ROC-AUC={rf_res['ROC_AUC']:.4f} | BalAcc={rf_res['Balanced_Accuracy']:.4f} | C1-Rec={rf_res['Recall_C1']:.4f} | C1-F1={rf_res['F1_C1']:.4f}", flush=True)

        # ---------------------------------------------------------------------
        # 2. Extra Trees (Nested CV with Inner CV Hyperparameter Tuning)
        # ---------------------------------------------------------------------
        et_base_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1))
        ])
        grid_search = GridSearchCV(
            estimator=et_base_pipe,
            param_grid=et_param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=1
        )
        grid_search.fit(X_train_fold, y_train_fold)
        best_et_estimator = grid_search.best_estimator_
        clean_inner_params = {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()}
        best_params_per_fold.append({"Fold": fold_num, "Best_Params": clean_inner_params, "Inner_CV_AUC": float(grid_search.best_score_)})

        et_nested_proba = best_et_estimator.predict_proba(X_val_fold)[:, 1]
        oof_probabilities["Extra Trees (Nested CV)"][val_idx] = et_nested_proba
        et_nested_res = calculate_metrics(y_val_fold.values, et_nested_proba, threshold=FIXED_THRESHOLD)
        et_nested_res["Fold"] = fold_num
        fold_metrics["Extra Trees (Nested CV)"].append(et_nested_res)
        print(f"  [ET Nested]      ROC-AUC={et_nested_res['ROC_AUC']:.4f} | BalAcc={et_nested_res['Balanced_Accuracy']:.4f} | C1-Rec={et_nested_res['Recall_C1']:.4f} | C1-F1={et_nested_res['F1_C1']:.4f} (Inner AUC={grid_search.best_score_:.4f})", flush=True)

        # ---------------------------------------------------------------------
        # 3. Extra Trees (Phase 8 Fixed Configuration)
        # ---------------------------------------------------------------------
        et_fixed_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", ExtraTreesClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="log2",
                random_state=RANDOM_STATE,
                n_jobs=1
            ))
        ])
        et_fixed_pipe.fit(X_train_fold, y_train_fold)
        et_fixed_proba = et_fixed_pipe.predict_proba(X_val_fold)[:, 1]
        oof_probabilities["Extra Trees (Phase 8 Fixed)"][val_idx] = et_fixed_proba
        et_fixed_res = calculate_metrics(y_val_fold.values, et_fixed_proba, threshold=FIXED_THRESHOLD)
        et_fixed_res["Fold"] = fold_num
        fold_metrics["Extra Trees (Phase 8 Fixed)"].append(et_fixed_res)
        print(f"  [ET Fixed Ph8]   ROC-AUC={et_fixed_res['ROC_AUC']:.4f} | BalAcc={et_fixed_res['Balanced_Accuracy']:.4f} | C1-Rec={et_fixed_res['Recall_C1']:.4f} | C1-F1={et_fixed_res['F1_C1']:.4f}", flush=True)

    return fold_metrics, oof_probabilities, best_params_per_fold, y.values


def summarize_results(fold_metrics: Dict[str, List[Dict[str, Any]]], oof_probas: Dict[str, np.ndarray], y_true: np.ndarray):
    """Compile fold-level and pooled metrics into detailed summary DataFrames."""
    metric_cols = [
        "Accuracy", "Balanced_Accuracy",
        "Recall_C1", "Precision_C1", "F1_C1",
        "Recall_C0", "Precision_C0", "F1_C0",
        "ROC_AUC", "PR_AUC"
    ]

    records = []

    for model_name, folds in fold_metrics.items():
        df_folds = pd.DataFrame(folds)

        # Fold rows
        for _, row in df_folds.iterrows():
            rec = {"Model": model_name, "Evaluation_Type": f"Fold_{int(row['Fold'])}"}
            for col in metric_cols:
                rec[col] = row[col]
            rec["TN"] = row["TN"]
            rec["FP"] = row["FP"]
            rec["FN"] = row["FN"]
            rec["TP"] = row["TP"]
            records.append(rec)

        # Mean row
        rec_mean = {"Model": model_name, "Evaluation_Type": "Outer_CV_Mean"}
        rec_std = {"Model": model_name, "Evaluation_Type": "Outer_CV_Std"}
        for col in metric_cols:
            rec_mean[col] = float(df_folds[col].mean())
            rec_std[col] = float(df_folds[col].std())
        records.append(rec_mean)
        records.append(rec_std)

        # Pooled metrics
        pooled_res = calculate_metrics(y_true, oof_probas[model_name], threshold=FIXED_THRESHOLD)
        rec_pooled = {"Model": model_name, "Evaluation_Type": "Outer_CV_Pooled"}
        for col in metric_cols:
            rec_pooled[col] = pooled_res[col]
        rec_pooled["TN"] = pooled_res["TN"]
        rec_pooled["FP"] = pooled_res["FP"]
        rec_pooled["FN"] = pooled_res["FN"]
        rec_pooled["TP"] = pooled_res["TP"]
        records.append(rec_pooled)

    summary_df = pd.DataFrame(records)
    return summary_df


def plot_model_comparison(summary_df: pd.DataFrame, oof_probas: Dict[str, np.ndarray], y_true: np.ndarray, save_path: Path):
    """Render comprehensive 4-panel diagnostic plot comparing outer fold distributions and curves."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    models = ["Random Forest (Production)", "Extra Trees (Nested CV)", "Extra Trees (Phase 8 Fixed)"]
    palette = {"Random Forest (Production)": "#1f77b4", "Extra Trees (Nested CV)": "#2ca02c", "Extra Trees (Phase 8 Fixed)": "#ff7f0e"}

    # 1. Bar Chart: Mean ROC-AUC & Balanced Accuracy with error bars
    ax1 = axes[0, 0]
    fold_data = summary_df[summary_df["Evaluation_Type"].str.startswith("Fold_")].copy()
    
    metrics_to_plot = ["ROC_AUC", "Balanced_Accuracy"]
    plot_rows = []
    for _, r in fold_data.iterrows():
        for m in metrics_to_plot:
            plot_rows.append({"Model": r["Model"], "Metric": m.replace("_", " "), "Value": r[m]})
    df_bar = pd.DataFrame(plot_rows)

    sns.barplot(
        data=df_bar,
        x="Metric",
        y="Value",
        hue="Model",
        palette=palette,
        errorbar="sd",
        capsize=0.08,
        ax=ax1
    )
    ax1.set_title("Outer 5-Fold Mean & Std (ROC-AUC & Balanced Accuracy)", fontsize=13, weight="bold")
    ax1.set_ylim(0.55, 0.75)
    ax1.set_ylabel("Score")
    ax1.legend(loc="lower right", framealpha=0.9)

    # 2. Bar Chart: Precision vs Recall (Class 1) with error bars
    ax2 = axes[0, 1]
    c1_metrics = ["Recall_C1", "Precision_C1", "F1_C1"]
    c1_rows = []
    for _, r in fold_data.iterrows():
        for m in c1_metrics:
            c1_rows.append({"Model": r["Model"], "Metric": m.replace("_", " "), "Value": r[m]})
    df_c1 = pd.DataFrame(c1_rows)

    sns.barplot(
        data=df_c1,
        x="Metric",
        y="Value",
        hue="Model",
        palette=palette,
        errorbar="sd",
        capsize=0.08,
        ax=ax2
    )
    ax2.set_title(f"Class 1 (Potable) Performance at Threshold {FIXED_THRESHOLD:.2f}", fontsize=13, weight="bold")
    ax2.set_ylim(0.40, 0.70)
    ax2.set_ylabel("Score")
    ax2.legend(loc="lower right", framealpha=0.9)

    # 3. ROC Curves (Pooled OOF)
    ax3 = axes[1, 0]
    for model_name in models:
        fpr, tpr, _ = roc_curve(y_true, oof_probas[model_name])
        auc_val = roc_auc_score(y_true, oof_probas[model_name])
        ax3.plot(fpr, tpr, label=f"{model_name} (Pooled AUC = {auc_val:.4f})", color=palette[model_name], lw=2)
    ax3.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.50)")
    ax3.set_title("Pooled Out-of-Fold ROC Curves (N=2,620)", fontsize=13, weight="bold")
    ax3.set_xlabel("False Positive Rate (1 - Specificity)")
    ax3.set_ylabel("True Positive Rate (Sensitivity)")
    ax3.legend(loc="lower right", framealpha=0.9)

    # 4. Precision-Recall Curves (Pooled OOF)
    ax4 = axes[1, 1]
    baseline_c1 = (y_true == 1).mean()
    for model_name in models:
        prec, rec, _ = precision_recall_curve(y_true, oof_probas[model_name])
        pr_auc = average_precision_score(y_true, oof_probas[model_name])
        ax4.plot(rec, prec, label=f"{model_name} (Pooled PR-AUC = {pr_auc:.4f})", color=palette[model_name], lw=2)
    ax4.axhline(baseline_c1, color="k", linestyle="--", alpha=0.5, label=f"No-Skill Baseline ({baseline_c1:.3f})")
    ax4.set_title("Pooled Out-of-Fold Precision-Recall Curves", fontsize=13, weight="bold")
    ax4.set_xlabel("Class 1 Recall")
    ax4.set_ylabel("Class 1 Precision")
    ax4.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Phase 9 model comparison plot -> {save_path.name}", flush=True)


def plot_confusion_matrices(summary_df: pd.DataFrame, save_path: Path):
    """Render side-by-side pooled confusion matrices."""
    pooled_rows = summary_df[summary_df["Evaluation_Type"] == "Outer_CV_Pooled"].set_index("Model")
    models = ["Random Forest (Production)", "Extra Trees (Nested CV)", "Extra Trees (Phase 8 Fixed)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        row = pooled_rows.loc[model_name]
        cm = np.array([
            [int(row["TN"]), int(row["FP"])],
            [int(row["FN"]), int(row["TP"])]
        ])

        labels = [
            [f"TN: {cm[0, 0]}\n(Spec: {cm[0, 0]/cm[0].sum():.1%})", f"FP: {cm[0, 1]}\n(FPR: {cm[0, 1]/cm[0].sum():.1%})"],
            [f"FN: {cm[1, 0]}\n(FNR: {cm[1, 0]/cm[1].sum():.1%})", f"TP: {cm[1, 1]}\n(Rec: {cm[1, 1]/cm[1].sum():.1%})"]
        ]

        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_title(f"{model_name}\n(Threshold = {FIXED_THRESHOLD:.2f})", fontsize=12, weight="bold")
        ax.set_xlabel("Predicted Label (0: Non-Potable, 1: Potable)")
        ax.set_ylabel("True Label")
        ax.set_xticklabels(["Non-Potable (0)", "Potable (1)"])
        ax.set_yticklabels(["Non-Potable (0)", "Potable (1)"])

    plt.suptitle(f"Phase 9: Pooled Out-of-Fold Confusion Matrices (N=2,620, Threshold={FIXED_THRESHOLD:.2f})", fontsize=14, weight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Phase 9 confusion matrices -> {save_path.name}", flush=True)


def fit_and_save_final_experimental_model(X: pd.DataFrame, y: pd.Series, best_params_per_fold: List[Dict[str, Any]]):
    """
    Fit final Extra Trees pipeline on all training records using most frequent inner parameters.
    Save strictly under models/experiments/.
    """
    ensure_dir(EXPERIMENTS_DIR)

    # Find modal parameter set across inner CV folds
    param_counts = {}
    for item in best_params_per_fold:
        key = json.dumps(item["Best_Params"], sort_keys=True)
        param_counts[key] = param_counts.get(key, 0) + 1

    modal_key = max(param_counts.items(), key=lambda x: x[1])[0]
    final_params = json.loads(modal_key)

    print("\n" + "=" * 80, flush=True)
    print("STEP 3: PERSISTING EXPERIMENTAL MODEL UNDER models/experiments/", flush=True)
    print("=" * 80, flush=True)
    print(f"Selected final Extra Trees hyperparameters (modal inner CV choice): {final_params}", flush=True)

    final_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", ExtraTreesClassifier(
            **final_params,
            random_state=RANDOM_STATE,
            n_jobs=1
        ))
    ])
    final_pipeline.fit(X, y)

    save_path = EXPERIMENTS_DIR / "phase_9_extra_trees_final.joblib"
    joblib.dump(final_pipeline, save_path)
    print(f"  * Persisted experimental pipeline -> {save_path.name}", flush=True)

    # Save metadata info
    info_path = EXPERIMENTS_DIR / "phase_9_validation_info.json"
    info = {
        "model_architecture": "ExtraTreesClassifier",
        "validation_strategy": "5-Fold Outer / 3-Fold Inner Nested Stratified CV",
        "fixed_threshold": FIXED_THRESHOLD,
        "final_hyperparameters": final_params,
        "fold_selections": best_params_per_fold,
        "training_samples": len(X),
        "status": "experimental_challenger"
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
    print(f"  * Persisted validation metadata -> {info_path.name}", flush=True)


def main():
    """Main execution flow for Phase 9."""
    ensure_dir(ARTIFACTS_DIR)
    ensure_dir(EXPERIMENTS_DIR)

    # 1. Load training data only
    X_train, y_train = load_training_data()

    # 2. Run nested cross-validation
    fold_metrics, oof_probas, best_params_per_fold, y_arr = run_nested_cross_validation(X_train, y_train)

    # 3. Compile and save results
    summary_df = summarize_results(fold_metrics, oof_probas, y_arr)
    csv_path = ARTIFACTS_DIR / "phase_9_nested_cv_results.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  [SAVED] Nested CV results table -> {csv_path.name}", flush=True)

    # 4. Generate diagnostic plots
    plot_path = ARTIFACTS_DIR / "phase_9_model_comparison.png"
    plot_model_comparison(summary_df, oof_probas, y_arr, plot_path)

    cm_path = ARTIFACTS_DIR / "phase_9_confusion_matrices.png"
    plot_confusion_matrices(summary_df, cm_path)

    # 5. Fit and persist experimental model
    fit_and_save_final_experimental_model(X_train, y_train, best_params_per_fold)

    print("\n" + "=" * 80, flush=True)
    print("                     PHASE 9 EXECUTION COMPLETE", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()

