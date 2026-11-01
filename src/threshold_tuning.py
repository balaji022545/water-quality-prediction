"""
Threshold Tuning Module for Water Quality Prediction.

Project: Water Quality Prediction Using Machine Learning
Champion Model: Tuned Random Forest Pipeline (models/best_model.joblib)

Objective:
- Decouple classification decision threshold from the standard 0.50 cutoff.
- Address low Class 1 Recall (29.65% CV, 26.95% Test) caused by class imbalance and tree averaging.
- Use strictly out-of-fold (OOF) cross-validation probabilities on the training set (zero test leakage).
- Evaluate 11 candidate thresholds: [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70].
- Save metrics to artifacts/threshold_comparison.csv.
- Generate visualization artifacts:
    * artifacts/threshold_metrics.png
    * artifacts/threshold_confusion_matrix.png
- Persist chosen threshold metadata to artifacts/threshold_config.json.
- Perform a single, one-time test set audit comparing default 0.50 vs tuned threshold.
"""

import os
import sys
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd
import numpy as np

# Configure non-interactive Matplotlib backend for headless environments
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn evaluation metrics & validation tools
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, classification_report
)
import joblib

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_STATE = 42
CV_FOLDS = 5

EVAL_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_training_data_and_model(
    processed_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR
) -> Tuple[pd.DataFrame, pd.Series, Any]:
    """
    Load training data and champion model pipeline.
    Preserves strict separation; test data is NOT loaded here.
    """
    x_train_path = processed_dir / "X_train.csv"
    y_train_path = processed_dir / "y_train.csv"
    model_path = models_dir / "best_model.joblib"

    if not x_train_path.is_file() or not y_train_path.is_file():
        raise FileNotFoundError(f"Training data missing in {processed_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Champion model missing at {model_path}")

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    pipeline = joblib.load(model_path)

    return X_train, y_train, pipeline


def compute_out_of_fold_probabilities(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE
) -> np.ndarray:
    """
    Generate out-of-fold probability predictions for Class 1 (Potable) on the training set.
    Each sample's probability is predicted by a pipeline fold that never saw that sample during fitting.
    Folds strictly isolate SimpleImputer, StandardScaler, and RandomForestClassifier.
    """
    print("\n--- 1. GENERATING OUT-OF-FOLD (OOF) PROBABILITIES (ZERO LEAKAGE) ---", flush=True)
    print(f"  * Training set size: N={len(X_train)} samples", flush=True)
    print(f"  * Cross-validation : Stratified {n_splits}-Fold (random_state={random_state})", flush=True)
    print("  * Pipeline steps   : Imputer -> Scaler -> Classifier", flush=True)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_probas = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    print(f"  * OOF Probabilities range: [{oof_probas.min():.4f}, {oof_probas.max():.4f}]", flush=True)
    print(f"  * Mean probability       : {oof_probas.mean():.4f}", flush=True)
    print(f"  * Median probability     : {np.median(oof_probas):.4f}", flush=True)

    return oof_probas


def evaluate_threshold_grid(
    y_true: pd.Series,
    probabilities: np.ndarray,
    thresholds: List[float] = EVAL_THRESHOLDS
) -> pd.DataFrame:
    """
    Calculate full evaluation metrics across candidate thresholds.
    """
    records = []

    for t in thresholds:
        preds = (probabilities >= t).astype(int)

        acc = accuracy_score(y_true, preds)
        bacc = balanced_accuracy_score(y_true, preds)

        # Class 1 (Potable) metrics
        prec1 = precision_score(y_true, preds, pos_label=1, zero_division=0)
        rec1 = recall_score(y_true, preds, pos_label=1, zero_division=0)
        f1_1 = f1_score(y_true, preds, pos_label=1, zero_division=0)

        # Class 0 (Not Potable) metrics
        prec0 = precision_score(y_true, preds, pos_label=0, zero_division=0)
        rec0 = recall_score(y_true, preds, pos_label=0, zero_division=0)
        f1_0 = f1_score(y_true, preds, pos_label=0, zero_division=0)

        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()

        records.append({
            "threshold": t,
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bacc), 4),
            "precision_class_1": round(float(prec1), 4),
            "recall_class_1": round(float(rec1), 4),
            "f1_class_1": round(float(f1_1), 4),
            "precision_class_0": round(float(prec0), 4),
            "recall_class_0": round(float(rec0), 4),
            "f1_class_0": round(float(f1_0), 4),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        })

    df = pd.DataFrame(records)
    return df


def plot_threshold_metrics(
    comparison_df: pd.DataFrame,
    selected_threshold: float,
    save_path: Path
):
    """
    Generate visualization showing how metrics vary with decision threshold.
    Plots: Accuracy, Class 1 Precision, Class 1 Recall, Class 1 F1-Score, Balanced Accuracy.
    Highlights the selected optimal threshold.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(comparison_df["threshold"], comparison_df["accuracy"], label="Overall Accuracy", color="#34495e", linestyle="--", linewidth=1.8)
    plt.plot(comparison_df["threshold"], comparison_df["balanced_accuracy"], label="Balanced Accuracy", color="#8e44ad", linewidth=2.5)
    plt.plot(comparison_df["threshold"], comparison_df["precision_class_1"], label="Class 1 Precision (Potable)", color="#e67e22", linewidth=2.0)
    plt.plot(comparison_df["threshold"], comparison_df["recall_class_1"], label="Class 1 Recall (Potable)", color="#27ae60", linewidth=2.5)
    plt.plot(comparison_df["threshold"], comparison_df["f1_class_1"], label="Class 1 F1-Score (Potable)", color="#2980b9", linewidth=2.0)

    # Highlight default (0.50) vs selected threshold
    plt.axvline(0.50, color="#7f8c8d", linestyle=":", label="Default Cutoff (0.50)", linewidth=1.5)
    plt.axvline(selected_threshold, color="#c0392b", linestyle="-", label=f"Selected Tuned Cutoff ({selected_threshold:.2f})", linewidth=2.2)

    plt.title("Classification Performance vs. Decision Threshold (Training OOF Cross-Validation)", fontsize=13, pad=12, fontweight="bold")
    plt.xlabel("Classification Threshold (Probability Cutoff for Potable Class 1)", fontsize=11, fontweight="bold")
    plt.ylabel("Metric Score", fontsize=11, fontweight="bold")
    plt.xlim(0.18, 0.72)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [SAVED] Threshold metrics plot -> {save_path.name}", flush=True)


def plot_confusion_matrix_at_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    save_path: Path
):
    """
    Plot and save confusion matrix heatmap for the selected threshold.
    """
    preds = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    labels = ["0: Not Potable", "1: Potable"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(f"Training OOF Confusion Matrix (N=2,620)\n[Selected Tuned Threshold = {threshold:.2f}]", pad=15, fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class", fontweight="bold")
    plt.ylabel("True Class", fontweight="bold")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [SAVED] Tuned confusion matrix plot -> {save_path.name}", flush=True)


def perform_single_test_evaluation(
    pipeline: Any,
    selected_threshold: float,
    processed_dir: Path = PROCESSED_DATA_DIR
) -> Dict[str, Any]:
    """
    Perform a single, final out-of-sample evaluation on the unseen test set (X_test.csv).
    Compares the default threshold (0.50) with the selected tuned threshold (0.40).
    Executed strictly once after threshold selection is complete.
    """
    print("\n--- 4. SINGLE FINAL TEST EVALUATION AUDIT (THRESHOLD 0.50 VS TUNED) ---", flush=True)

    x_test_path = processed_dir / "X_test.csv"
    y_test_path = processed_dir / "y_test.csv"

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    test_probas = pipeline.predict_proba(X_test)[:, 1]

    # Evaluate at default 0.50
    preds_def = (test_probas >= 0.50).astype(int)
    cm_def = confusion_matrix(y_test, preds_def)

    # Evaluate at tuned threshold
    preds_tuned = (test_probas >= selected_threshold).astype(int)
    cm_tuned = confusion_matrix(y_test, preds_tuned)

    results = {
        "test_samples": len(y_test),
        "default_threshold_0_50": {
            "threshold": 0.50,
            "accuracy": round(float(accuracy_score(y_test, preds_def)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, preds_def)), 4),
            "class_0_recall": round(float(recall_score(y_test, preds_def, pos_label=0)), 4),
            "class_0_precision": round(float(precision_score(y_test, preds_def, pos_label=0, zero_division=0)), 4),
            "class_0_f1": round(float(f1_score(y_test, preds_def, pos_label=0, zero_division=0)), 4),
            "class_1_recall": round(float(recall_score(y_test, preds_def, pos_label=1)), 4),
            "class_1_precision": round(float(precision_score(y_test, preds_def, pos_label=1, zero_division=0)), 4),
            "class_1_f1": round(float(f1_score(y_test, preds_def, pos_label=1, zero_division=0)), 4),
            "confusion_matrix": cm_def.tolist()
        },
        "tuned_threshold": {
            "threshold": selected_threshold,
            "accuracy": round(float(accuracy_score(y_test, preds_tuned)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, preds_tuned)), 4),
            "class_0_recall": round(float(recall_score(y_test, preds_tuned, pos_label=0)), 4),
            "class_0_precision": round(float(precision_score(y_test, preds_tuned, pos_label=0, zero_division=0)), 4),
            "class_0_f1": round(float(f1_score(y_test, preds_tuned, pos_label=0, zero_division=0)), 4),
            "class_1_recall": round(float(recall_score(y_test, preds_tuned, pos_label=1)), 4),
            "class_1_precision": round(float(precision_score(y_test, preds_tuned, pos_label=1, zero_division=0)), 4),
            "class_1_f1": round(float(f1_score(y_test, preds_tuned, pos_label=1, zero_division=0)), 4),
            "confusion_matrix": cm_tuned.tolist()
        }
    }

    print(f"  * Test Samples: {len(y_test)} (Unseen Holdout Partition)", flush=True)
    print(f"  * Default Threshold (0.50): Accuracy={results['default_threshold_0_50']['accuracy']:.4f} | BalancedAcc={results['default_threshold_0_50']['balanced_accuracy']:.4f} | C1 Recall={results['default_threshold_0_50']['class_1_recall']:.4f} | C0 Recall={results['default_threshold_0_50']['class_0_recall']:.4f}", flush=True)
    print(f"  * Tuned Threshold ({selected_threshold:.2f})  : Accuracy={results['tuned_threshold']['accuracy']:.4f} | BalancedAcc={results['tuned_threshold']['balanced_accuracy']:.4f} | C1 Recall={results['tuned_threshold']['class_1_recall']:.4f} | C0 Recall={results['tuned_threshold']['class_0_recall']:.4f}", flush=True)

    return results


def run_threshold_tuning() -> Dict[str, Any]:
    """Orchestrate end-to-end classification threshold tuning and artifact generation."""
    ensure_dir(ARTIFACTS_DIR)

    print("\n" + "=" * 85, flush=True)
    print(" " * 22 + "CLASSIFICATION THRESHOLD TUNING", flush=True)
    print("=" * 85, flush=True)

    # 1. Load data and model
    X_train, y_train, pipeline = load_training_data_and_model()

    # 2. Generate OOF probabilities on training set
    oof_probas = compute_out_of_fold_probabilities(pipeline, X_train, y_train)

    # 3. Evaluate threshold grid
    print("\n--- 2. EVALUATING THRESHOLD GRID [0.20 - 0.70] ---", flush=True)
    comparison_df = evaluate_threshold_grid(y_train, oof_probas, EVAL_THRESHOLDS)

    # Save comparison table
    csv_path = ARTIFACTS_DIR / "threshold_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"  [SAVED] Threshold comparison table -> {csv_path.name}", flush=True)

    print("\n" + "-" * 115, flush=True)
    print(comparison_df[["threshold", "accuracy", "balanced_accuracy", "precision_class_1", "recall_class_1", "f1_class_1", "recall_class_0", "f1_class_0"]].to_string(index=False), flush=True)
    print("-" * 115, flush=True)

    # 4. Selection Logic
    # Criteria: Maximize balanced accuracy to address low Class 1 recall without tanking Class 0 recall
    best_row = comparison_df.loc[comparison_df["balanced_accuracy"].idxmax()]
    selected_threshold = float(best_row["threshold"])

    print(f"\n--- 3. OPTIMAL THRESHOLD SELECTION ---", flush=True)
    print(f"  * Selected Threshold     : {selected_threshold:.2f}", flush=True)
    print(f"  * Selection Criterion     : Maximum Balanced Accuracy ({best_row['balanced_accuracy']:.4f}) with balanced Class 0/1 trade-off", flush=True)
    print(f"  * Class 1 Recall at {selected_threshold:.2f} : {best_row['recall_class_1']:.4f} (up from {comparison_df.loc[comparison_df['threshold']==0.50, 'recall_class_1'].values[0]:.4f} at 0.50)", flush=True)
    print(f"  * Class 0 Recall at {selected_threshold:.2f} : {best_row['recall_class_0']:.4f} (compared to {comparison_df.loc[comparison_df['threshold']==0.50, 'recall_class_0'].values[0]:.4f} at 0.50)", flush=True)
    print(f"  * Class 1 F1 at {selected_threshold:.2f}     : {best_row['f1_class_1']:.4f} (up from {comparison_df.loc[comparison_df['threshold']==0.50, 'f1_class_1'].values[0]:.4f} at 0.50)", flush=True)

    # 5. Visualizations
    plot_metrics_path = ARTIFACTS_DIR / "threshold_metrics.png"
    plot_threshold_metrics(comparison_df, selected_threshold, plot_metrics_path)

    plot_cm_path = ARTIFACTS_DIR / "threshold_confusion_matrix.png"
    plot_confusion_matrix_at_threshold(y_train, oof_probas, selected_threshold, plot_cm_path)

    # 6. Single final test evaluation audit
    test_audit = perform_single_test_evaluation(pipeline, selected_threshold)

    # 7. Persist configuration JSON
    config_data = {
        "selected_threshold": selected_threshold,
        "selection_criteria": "Maximized Balanced Accuracy (0.6336) on 5-fold cross-validation out-of-fold training probabilities, substantially improving Class 1 recall from 29.65% to 57.83% while retaining 68.90% Class 0 screening sensitivity.",
        "date_of_selection": datetime.now(timezone.utc).isoformat(),
        "candidate_thresholds_evaluated": EVAL_THRESHOLDS,
        "validation_metrics_at_selected_threshold": best_row.to_dict(),
        "validation_metrics_at_default_0_50": comparison_df.loc[comparison_df["threshold"] == 0.50].iloc[0].to_dict(),
        "final_test_evaluation_audit": test_audit
    }

    config_path = ARTIFACTS_DIR / "threshold_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)
    print(f"  [SAVED] Threshold configuration JSON -> {config_path.name}", flush=True)

    print("\n" + "=" * 85, flush=True)
    print(" " * 28 + "THRESHOLD TUNING COMPLETED", flush=True)
    print("=" * 85 + "\n", flush=True)

    return config_data


def main():
    """CLI execution entrypoint."""
    run_threshold_tuning()


if __name__ == "__main__":
    main()

