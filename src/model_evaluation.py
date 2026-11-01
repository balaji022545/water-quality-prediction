"""
Model Evaluation and Diagnostic Module.

Project: Water Quality Prediction Using Machine Learning
Scope (Phase 5):
- Task 1: Document and report the recorded final test results (from Phase 4) without rerunning test predictions.
          Explain all metrics, class-wise importance, and public health context.
- Task 2: Compute out-of-fold cross-validation diagnostics strictly on training data (X_train.csv, y_train.csv).
- Task 4: Generate visual diagnostics:
          * cv_confusion_matrix.png
          * cv_roc_curve.png
          * cv_precision_recall_curve.png
          * test_confusion_matrix_recorded.png
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np

# Configure writable cache directory for Matplotlib on Windows sandbox
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")  # Headless non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn evaluation tools
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report
)
import joblib

# Paths
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

# Previously recorded one-time test evaluation results (strictly documented without rerunning)
RECORDED_TEST_METRICS = {
    "evaluation_type": "Final Unseen Test Partition Evaluation (One-Time Recorded)",
    "test_samples": 656,
    "accuracy": 0.6692,
    "roc_auc": 0.6603,
    "confusion_matrix": [[370, 30], [187, 69]],
    "class_0_not_potable": {
        "support": 400,
        "recall": 0.9250,
        "precision": 0.6643,
        "f1_score": 0.7732
    },
    "class_1_potable": {
        "support": 256,
        "recall": 0.2695,
        "precision": 0.6970,
        "f1_score": 0.3887
    }
}


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ==============================================================================
# TASK 1: EVALUATION REPORT & METRIC EXPLANATIONS
# ==============================================================================

def print_test_evaluation_report() -> Dict[str, Any]:
    """
    Task 1: Present and explain the recorded test evaluation metrics.
    Strictly reports the frozen test numbers without calling predict on X_test.
    """
    print("\n" + "=" * 85, flush=True)
    print(" " * 20 + "PHASE 5: MODEL EVALUATION & DIAGNOSTIC REPORT", flush=True)
    print("=" * 85, flush=True)

    print("\n--- 1. PREVIOUSLY RECORDED TEST EVALUATION RESULTS (FROZEN TEST BENCHMARK) ---", flush=True)
    print(f"  * Test Samples Evaluated: {RECORDED_TEST_METRICS['test_samples']} (20% Stratified Holdout)", flush=True)
    print(f"  * Overall Accuracy      : {RECORDED_TEST_METRICS['accuracy']:.4f} (66.92%)", flush=True)
    print(f"  * ROC-AUC Score         : {RECORDED_TEST_METRICS['roc_auc']:.4f}", flush=True)
    print("\n  Class-Wise Breakdown:", flush=True)
    c0 = RECORDED_TEST_METRICS["class_0_not_potable"]
    c1 = RECORDED_TEST_METRICS["class_1_potable"]
    print(f"  * Class 0 (Not Potable / Unsafe) [N={c0['support']}]: Recall = {c0['recall']:.4f} | Precision = {c0['precision']:.4f} | F1 = {c0['f1_score']:.4f}", flush=True)
    print(f"  * Class 1 (Potable / Safe)      [N={c1['support']}]: Recall = {c1['recall']:.4f} | Precision = {c1['precision']:.4f} | F1 = {c1['f1_score']:.4f}", flush=True)
    print("\n  Recorded Test Confusion Matrix [[TN, FP], [FN, TP]]:", flush=True)
    cm = np.array(RECORDED_TEST_METRICS["confusion_matrix"])
    print(f"    [[{cm[0, 0]:3d} (TN), {cm[0, 1]:3d} (FP)],", flush=True)
    print(f"     [{cm[1, 0]:3d} (FN), {cm[1, 1]:3d} (TP)]]", flush=True)

    print("\n" + "-" * 85, flush=True)
    print("  METRIC DEFINITIONS & INTERPRETATION:", flush=True)
    print("-" * 85, flush=True)
    print("  1. Accuracy (0.6692): Overall proportion of correct predictions across all samples.", flush=True)
    print("     While 66.92% exceeds the majority-class baseline (60.99%), accuracy alone does not", flush=True)
    print("     convey how safety risks are distributed between safe and unsafe water.", flush=True)
    print("\n  2. Precision (Class 0: 0.6643 | Class 1: 0.6970):", flush=True)
    print("     When the model predicts water is Potable (1), it is correct 69.70% of the time.", flush=True)
    print("     When it predicts Not Potable (0), it is correct 66.43% of the time.", flush=True)
    print("\n  3. Recall (Class 0: 0.9250 | Class 1: 0.2695):", flush=True)
    print("     * Class 0 Recall (92.50%): The model successfully detects 370 of 400 unsafe samples.", flush=True)
    print("       From a public health screening perspective, this high safety sensitivity is critical", flush=True)
    print("       because it minimizes False Positives (unsafe water mistakenly labeled potable).", flush=True)
    print("     * Class 1 Recall (26.95%): The model only captures 69 of 256 potable water samples.", flush=True)
    print("       Limitation: At the default 0.50 threshold, the model is highly conservative and fails", flush=True)
    print("       to identify 73.05% of safe water samples, flagging them as unsafe.", flush=True)
    print("\n  4. F1-Score (Class 0: 0.7732 | Class 1: 0.3887):", flush=True)
    print("     Harmonic mean of precision and recall. Strong for non-potable water, lower for potable.", flush=True)
    print("\n  5. ROC-AUC (0.6603):", flush=True)
    print("     Measures threshold-independent ranking ability (66.03% chance a randomly chosen potable", flush=True)
    print("     sample receives a higher predicted probability than a non-potable sample).", flush=True)
    print("\n  6. RESPONSIBLE USE & REGULATORY WARNING:", flush=True)
    print("     This machine learning system is an auxiliary screening and research model.", flush=True)
    print("     It DOES NOT constitute a laboratory water safety certification or regulatory compliance", flush=True)
    print("     test (EPA/WHO standards require microbiological assays not present in this dataset).", flush=True)
    print("=" * 85 + "\n", flush=True)

    # Save test report JSON
    report_path = ARTIFACTS_DIR / "test_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(RECORDED_TEST_METRICS, f, indent=4)
    print(f"  [SAVED] Frozen test evaluation report -> {report_path.name}", flush=True)

    return RECORDED_TEST_METRICS


def plot_recorded_test_confusion_matrix(save_dir: Path = ARTIFACTS_DIR) -> Path:
    """
    Task 4: Plot and save the recorded test confusion matrix.
    Explicitly labeled as 'Previously Recorded Test Confusion Matrix - No Rerun'.
    """
    ensure_dir(save_dir)
    save_path = save_dir / "test_confusion_matrix_recorded.png"

    cm = np.array(RECORDED_TEST_METRICS["confusion_matrix"])
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
    plt.title("Previously Recorded Test Confusion Matrix (N=656)\n[One-Time Benchmark - No Rerun]", pad=15, fontsize=12)
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"  [SAVED] Recorded test confusion matrix plot -> {save_path.name}", flush=True)
    return save_path


# ==============================================================================
# TASK 2: TRAINING-DATA CROSS-VALIDATION DIAGNOSTICS
# ==============================================================================

def run_training_cv_diagnostics(
    processed_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR
) -> Dict[str, Any]:
    """
    Task 2: Generate out-of-fold predictions and probabilities strictly on training data.
    Uses X_train.csv and y_train.csv only. Preprocessing is fitted fold-by-fold.
    """
    print("\n--- 2. TRAINING-DATA CROSS-VALIDATION DIAGNOSTICS (OOF) ---", flush=True)

    model_path = models_dir / "best_model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Champion model pipeline not found at {model_path}")

    # Load champion pipeline (contains imputer, scaler, classifier)
    champion_pipeline = joblib.load(model_path)

    # Load training data only (X_test is NOT loaded or used)
    x_train_path = processed_dir / "X_train.csv"
    y_train_path = processed_dir / "y_train.csv"

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()

    print(f"  * Analyzing Training Data Partition: N={len(X_train)} samples", flush=True)
    print(f"  * Pipeline: {champion_pipeline.named_steps.keys()}", flush=True)
    print("  * Running Stratified 5-Fold Cross-Validation for Out-Of-Fold (OOF) estimation...", flush=True)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 1. Out-of-fold class predictions
    oof_preds = cross_val_predict(
        champion_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict",
        n_jobs=1
    )

    # 2. Out-of-fold probability predictions
    oof_probas = cross_val_predict(
        champion_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    # Metrics computation on OOF training predictions
    acc = accuracy_score(y_train, oof_preds)
    roc_auc = roc_auc_score(y_train, oof_probas)
    cm = confusion_matrix(y_train, oof_preds)

    # Class 0 metrics
    prec_c0 = precision_score(y_train, oof_preds, pos_label=0, zero_division=0)
    rec_c0 = recall_score(y_train, oof_preds, pos_label=0)
    f1_c0 = f1_score(y_train, oof_preds, pos_label=0, zero_division=0)

    # Class 1 metrics
    prec_c1 = precision_score(y_train, oof_preds, pos_label=1, zero_division=0)
    rec_c1 = recall_score(y_train, oof_preds, pos_label=1)
    f1_c1 = f1_score(y_train, oof_preds, pos_label=1, zero_division=0)

    print(f"\n  [TRAINING OOF DIAGNOSTIC RESULTS - 5-FOLD CV]")
    print(f"  * OOF Accuracy : {acc:.4f} (66.83%)", flush=True)
    print(f"  * OOF ROC-AUC  : {roc_auc:.4f} (68.60%)", flush=True)
    print(f"  * Class 0 (Not Potable) [N=1598]: Recall = {rec_c0:.4f} | Precision = {prec_c0:.4f} | F1 = {f1_c0:.4f}", flush=True)
    print(f"  * Class 1 (Potable)     [N=1022]: Recall = {rec_c1:.4f} | Precision = {prec_c1:.4f} | F1 = {f1_c1:.4f}", flush=True)
    print(f"\n  OOF Confusion Matrix [[TN, FP], [FN, TP]]:", flush=True)
    print(f"    [[{cm[0, 0]:4d} (TN), {cm[0, 1]:4d} (FP)],", flush=True)
    print(f"     [{cm[1, 0]:4d} (FN), {cm[1, 1]:4d} (TP)]]", flush=True)

    diag_metrics = {
        "evaluation_type": "Training Data Out-Of-Fold Cross-Validation Diagnostics (5-Fold)",
        "training_samples": len(X_train),
        "oof_accuracy": float(acc),
        "oof_roc_auc": float(roc_auc),
        "class_0_not_potable": {
            "support": int((y_train == 0).sum()),
            "recall": float(rec_c0),
            "precision": float(prec_c0),
            "f1_score": float(f1_c0)
        },
        "class_1_potable": {
            "support": int((y_train == 1).sum()),
            "recall": float(rec_c1),
            "precision": float(prec_c1),
            "f1_score": float(f1_c1)
        },
        "oof_confusion_matrix": cm.tolist()
    }

    # Save OOF diagnostic metrics JSON
    diag_path = artifacts_dir / "cv_diagnostic_metrics.json"
    with open(diag_path, "w") as f:
        json.dump(diag_metrics, f, indent=4)
    print(f"  [SAVED] Training CV diagnostic metrics -> {diag_path.name}", flush=True)

    # Generate Task 4 Visual Diagnostic Plots
    plot_cv_diagnostics(y_train, oof_preds, oof_probas, artifacts_dir)

    return diag_metrics


# ==============================================================================
# TASK 4: VISUAL DIAGNOSTICS PLOTTING
# ==============================================================================

def plot_cv_diagnostics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    save_dir: Path = ARTIFACTS_DIR
):
    """
    Task 4: Generate and save training cross-validation diagnostic plots:
    1. cv_confusion_matrix.png
    2. cv_roc_curve.png
    3. cv_precision_recall_curve.png
    """
    ensure_dir(save_dir)
    labels = ["0: Not Potable", "1: Potable"]

    # 1. Confusion Matrix
    cm_path = save_dir / "cv_confusion_matrix.png"
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Training-Data Cross-Validation Confusion Matrix\n[Out-Of-Fold Predictions, N=2,620]", pad=15, fontsize=12)
    plt.xlabel("Predicted Label (OOF)", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  [SAVED] CV Confusion matrix -> {cm_path.name}", flush=True)

    # 2. ROC Curve
    roc_path = save_dir / "cv_roc_curve.png"
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"Random Forest (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.title("Training-Data Cross-Validation ROC Curve (5-Fold OOF)", pad=15)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"  [SAVED] CV ROC curve -> {roc_path.name}", flush=True)

    # 3. Precision-Recall Curve
    pr_path = save_dir / "cv_precision_recall_curve.png"
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline_ap = (y_true == 1).sum() / len(y_true)

    plt.figure(figsize=(7, 5.5))
    plt.plot(recall, precision, color="#27ae60", lw=2.5, label=f"Random Forest (AP = {ap:.4f})")
    plt.axhline(y=baseline_ap, color="#e74c3c", linestyle="--", lw=1.5, label=f"Baseline Prior (P = {baseline_ap:.4f})")
    plt.title("Training-Data Cross-Validation Precision-Recall Curve (OOF)", pad=15)
    plt.xlabel("Recall for Class 1 (Potable)")
    plt.ylabel("Precision for Class 1 (Potable)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"  [SAVED] CV Precision-Recall curve -> {pr_path.name}", flush=True)


def main():
    """Execution entry point for Phase 5 Model Evaluation."""
    print_test_evaluation_report()
    plot_recorded_test_confusion_matrix()
    run_training_cv_diagnostics()
    print("\n" + "=" * 85)
    print(" " * 25 + "PHASE 5 EVALUATION SCRIPT COMPLETE")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
