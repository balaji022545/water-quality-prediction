"""
Phase 7: Class Imbalance Handling and Model Optimization Module.

Project: Water Quality Prediction Using Machine Learning
Champion Model: Tuned Random Forest Pipeline
Scope:
- Step 1: Analyze training set class distribution and generate artifacts/class_distribution.png.
- Step 2 & 3: Implement 6 distinct imbalance handling techniques with identical Stratified 5-Fold CV:
    * Exp A: No Resampling (Baseline Tuned Random Forest)
    * Exp B: Random Oversampling (RandomOverSampler)
    * Exp C: Random Undersampling (RandomUnderSampler)
    * Exp D: SMOTE (Synthetic Minority Over-sampling)
    * Exp E: Algorithmic Class Weighting (class_weight='balanced')
    * Exp F: Bootstrap Balanced Subsampling (class_weight='balanced_subsample')
- Step 4: Evaluate Out-Of-Fold (OOF) predictions across 10 metrics at threshold 0.50:
    * Accuracy, Balanced Accuracy, Precision C0/C1, Recall C0/C1, F1 C0/C1, ROC-AUC, PR-AUC.
    * Save artifacts/imbalance_comparison.csv.
    * Generate comparative visualizations:
        - artifacts/imbalance_accuracy_comparison.png
        - artifacts/imbalance_recall_comparison.png
        - artifacts/imbalance_f1_comparison.png
        - artifacts/imbalance_confusion_matrices.png
- Step 5: Select the best-performing imbalance approach.
- Step 6: Threshold analysis for the winning approach across [0.30 - 0.60].
    * Save artifacts/best_imbalance_threshold_comparison.csv.
- Step 7: Persist experimental models under models/experiments/ without touching models/best_model.joblib.
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

# Scikit-learn & Imbalanced-learn modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_STATE = 42
CV_FOLDS = 5

# Shared tuned Random Forest parameters
RF_BASE_PARAMS = {
    "n_estimators": 150,
    "max_depth": 15,
    "random_state": RANDOM_STATE,
    "n_jobs": 1
}


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ==============================================================================
# STEP 1: ANALYZE CLASS DISTRIBUTION
# ==============================================================================
def analyze_class_distribution(
    y_train: pd.Series,
    save_path: Path = ARTIFACTS_DIR / "class_distribution.png"
) -> Dict[str, Any]:
    """
    Step 1: Compute training target distribution and generate visualization.
    """
    counts = y_train.value_counts().sort_index()
    total = len(y_train)
    c0_count = int(counts[0])
    c1_count = int(counts[1])
    c0_pct = (c0_count / total) * 100
    c1_pct = (c1_count / total) * 100
    ratio = c0_count / c1_count

    # Determine degree of imbalance
    if ratio < 1.5:
        imbalance_degree = "Mild (Near Balanced)"
    elif ratio <= 3.0:
        imbalance_degree = "Moderate"
    else:
        imbalance_degree = "Severe"

    print("\n" + "=" * 80, flush=True)
    print(" " * 20 + "PHASE 7: STEP 1 — CLASS DISTRIBUTION ANALYSIS", flush=True)
    print("=" * 80, flush=True)
    print(f"  * Total Training Samples : {total:,}", flush=True)
    print(f"  * Class 0 (Not Potable)  : {c0_count:,} ({c0_pct:.2f}%)", flush=True)
    print(f"  * Class 1 (Potable)      : {c1_count:,} ({c1_pct:.2f}%)", flush=True)
    print(f"  * Majority:Minority Ratio: {ratio:.3f} : 1 ({c0_count}:{c1_count})", flush=True)
    print(f"  * Imbalance Assessment   : {imbalance_degree} Imbalance", flush=True)
    print("=" * 80, flush=True)

    # Plot
    ensure_dir(save_path.parent)
    plt.figure(figsize=(7, 5))
    palette = ["#e74c3c", "#2ecc71"]
    bars = plt.bar(
        ["Class 0\n(Not Potable)", "Class 1\n(Potable)"],
        [c0_count, c1_count],
        color=palette,
        edgecolor="#2c3e50",
        linewidth=1.2,
        width=0.45
    )

    for bar, pct in zip(bars, [c0_pct, c1_pct]):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 35,
            f"{int(yval):,} samples\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11
        )

    plt.title(
        f"Training Data Class Distribution (N={total:,})\nMajority:Minority Ratio = {ratio:.2f}:1 ({imbalance_degree})",
        fontsize=12,
        fontweight="bold",
        pad=15
    )
    plt.ylabel("Sample Count", fontweight="bold")
    plt.ylim(0, max(c0_count, c1_count) * 1.18)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [SAVED] Class distribution plot -> {save_path.name}", flush=True)

    return {
        "total_samples": total,
        "class_0_count": c0_count,
        "class_1_count": c1_count,
        "class_0_pct": round(c0_pct, 2),
        "class_1_pct": round(c1_pct, 2),
        "imbalance_ratio": round(ratio, 3),
        "imbalance_degree": imbalance_degree
    }


# ==============================================================================
# STEP 2 & 3: CONSTRUCT EXPERIMENTAL PIPELINES
# ==============================================================================
def get_imbalance_experiment_pipelines() -> Dict[str, ImbPipeline]:
    """
    Step 3: Define 6 experimental pipelines using identical tuned Random Forest.
    All preprocessing and resampling steps are strictly embedded inside ImbPipeline.
    """
    pipelines = {
        "Exp A: No Resampling (Baseline)": ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(**RF_BASE_PARAMS))
        ]),
        "Exp B: Random Oversampling": ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("sampler", RandomOverSampler(sampling_strategy="auto", random_state=RANDOM_STATE)),
            ("classifier", RandomForestClassifier(**RF_BASE_PARAMS))
        ]),
        "Exp C: Random Undersampling": ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("sampler", RandomUnderSampler(sampling_strategy="auto", random_state=RANDOM_STATE)),
            ("classifier", RandomForestClassifier(**RF_BASE_PARAMS))
        ]),
        "Exp D: SMOTE": ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("sampler", SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=RANDOM_STATE)),
            ("classifier", RandomForestClassifier(**RF_BASE_PARAMS))
        ]),
        "Exp E: Class Weight (balanced)": ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(**RF_BASE_PARAMS, class_weight="balanced"))
        ]),
        "Exp F: Balanced Subsample": ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(**RF_BASE_PARAMS, class_weight="balanced_subsample"))
        ]),
    }
    return pipelines


# ==============================================================================
# STEP 4: EVALUATE ALL EXPERIMENTS VIA STRATIFIED 5-FOLD CV
# ==============================================================================
def evaluate_all_imbalance_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipelines: Dict[str, ImbPipeline]
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Step 4: Generate out-of-fold probability predictions for all 6 experiments.
    Evaluates 10 metrics at threshold 0.50.
    """
    print("\n" + "=" * 80, flush=True)
    print(" " * 15 + "PHASE 7: STEP 4 — EVALUATING 6 IMBALANCE TECHNIQUES", flush=True)
    print("=" * 80, flush=True)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    records = []
    oof_probabilities = {}
    confusion_matrices = {}

    for exp_name, pipe in pipelines.items():
        print(f"\nEvaluating: {exp_name}...", flush=True)

        # Generate out-of-fold probability predictions
        probas = cross_val_predict(
            pipe,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1
        )[:, 1]

        oof_probabilities[exp_name] = probas

        # Classify at threshold 0.50
        preds = (probas >= 0.50).astype(int)

        # Metrics
        acc = accuracy_score(y_train, preds)
        bacc = balanced_accuracy_score(y_train, preds)

        # Class 0 metrics
        prec0 = precision_score(y_train, preds, pos_label=0, zero_division=0)
        rec0 = recall_score(y_train, preds, pos_label=0, zero_division=0)
        f1_0 = f1_score(y_train, preds, pos_label=0, zero_division=0)

        # Class 1 metrics
        prec1 = precision_score(y_train, preds, pos_label=1, zero_division=0)
        rec1 = recall_score(y_train, preds, pos_label=1, zero_division=0)
        f1_1 = f1_score(y_train, preds, pos_label=1, zero_division=0)

        roc_auc = roc_auc_score(y_train, probas)
        pr_auc = average_precision_score(y_train, probas)

        cm = confusion_matrix(y_train, preds)
        confusion_matrices[exp_name] = cm
        tn, fp, fn, tp = cm.ravel()

        records.append({
            "Experiment": exp_name,
            "Accuracy": round(float(acc), 4),
            "Balanced_Accuracy": round(float(bacc), 4),
            "Precision_C0": round(float(prec0), 4),
            "Recall_C0": round(float(rec0), 4),
            "F1_C0": round(float(f1_0), 4),
            "Precision_C1": round(float(prec1), 4),
            "Recall_C1": round(float(rec1), 4),
            "F1_C1": round(float(f1_1), 4),
            "ROC_AUC": round(float(roc_auc), 4),
            "PR_AUC": round(float(pr_auc), 4),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        })

        print(f"  * Acc: {acc:.4f} | BalancedAcc: {bacc:.4f} | C1 Recall: {rec1:.4f} | C1 Prec: {prec1:.4f} | C1 F1: {f1_1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}", flush=True)

    comparison_df = pd.DataFrame(records)

    # Save CSV
    csv_path = ARTIFACTS_DIR / "imbalance_comparison.csv"
    ensure_dir(csv_path.parent)
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n  [SAVED] Imbalance comparison table -> {csv_path.name}", flush=True)

    return comparison_df, oof_probabilities, confusion_matrices


# ==============================================================================
# VISUALIZATIONS FOR STEP 4
# ==============================================================================
def generate_imbalance_visualizations(
    comparison_df: pd.DataFrame,
    confusion_matrices: Dict[str, np.ndarray],
    artifacts_dir: Path = ARTIFACTS_DIR
):
    """
    Generate all 4 required visual artifacts:
    1. artifacts/imbalance_accuracy_comparison.png
    2. artifacts/imbalance_recall_comparison.png
    3. artifacts/imbalance_f1_comparison.png
    4. artifacts/imbalance_confusion_matrices.png
    """
    ensure_dir(artifacts_dir)
    labels = [
        "Baseline\n(No Resample)",
        "Random\nOversampling",
        "Random\nUndersampling",
        "SMOTE\nSynthetic",
        "Class Weight\n(Balanced)",
        "Balanced\nSubsample"
    ]

    # 1. Accuracy & Balanced Accuracy Comparison
    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width/2, comparison_df["Accuracy"], width, label="Overall Accuracy", color="#3498db", edgecolor="#1b4f72")
    plt.bar(x + width/2, comparison_df["Balanced_Accuracy"], width, label="Balanced Accuracy", color="#9b59b6", edgecolor="#4a235a")

    for i in x:
        acc_v = comparison_df["Accuracy"].iloc[i]
        bacc_v = comparison_df["Balanced_Accuracy"].iloc[i]
        plt.text(i - width/2, acc_v + 0.012, f"{acc_v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
        plt.text(i + width/2, bacc_v + 0.012, f"{bacc_v:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    plt.xticks(x, labels, fontsize=9.5)
    plt.ylabel("Score", fontweight="bold")
    plt.title("Phase 7: Overall Accuracy vs. Balanced Accuracy Across Imbalance Techniques (Threshold 0.50)", fontsize=11, fontweight="bold", pad=12)
    plt.ylim(0.40, 0.78)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p1 = artifacts_dir / "imbalance_accuracy_comparison.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"  [SAVED] Plot 1 -> {p1.name}", flush=True)

    # 2. Recall Comparison (Class 0 vs Class 1)
    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width/2, comparison_df["Recall_C0"], width, label="Class 0 Recall (Unsafe)", color="#e74c3c", edgecolor="#78281f")
    plt.bar(x + width/2, comparison_df["Recall_C1"], width, label="Class 1 Recall (Potable)", color="#2ecc71", edgecolor="#145a32")

    for i in x:
        r0_v = comparison_df["Recall_C0"].iloc[i]
        r1_v = comparison_df["Recall_C1"].iloc[i]
        plt.text(i - width/2, r0_v + 0.015, f"{r0_v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
        plt.text(i + width/2, r1_v + 0.015, f"{r1_v:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    plt.xticks(x, labels, fontsize=9.5)
    plt.ylabel("Recall (Sensitivity)", fontweight="bold")
    plt.title("Phase 7: Class-Wise Recall Comparison (Class 0 Unsafe vs. Class 1 Potable)", fontsize=11, fontweight="bold", pad=12)
    plt.ylim(0.15, 1.05)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p2 = artifacts_dir / "imbalance_recall_comparison.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"  [SAVED] Plot 2 -> {p2.name}", flush=True)

    # 3. F1-Score Comparison
    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width/2, comparison_df["F1_C0"], width, label="Class 0 F1 (Unsafe)", color="#e67e22", edgecolor="#6e2c00")
    plt.bar(x + width/2, comparison_df["F1_C1"], width, label="Class 1 F1 (Potable)", color="#16a085", edgecolor="#0e6251")

    for i in x:
        f0_v = comparison_df["F1_C0"].iloc[i]
        f1_v = comparison_df["F1_C1"].iloc[i]
        plt.text(i - width/2, f0_v + 0.015, f"{f0_v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
        plt.text(i + width/2, f1_v + 0.015, f"{f1_v:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    plt.xticks(x, labels, fontsize=9.5)
    plt.ylabel("F1-Score (Harmonic Mean)", fontweight="bold")
    plt.title("Phase 7: F1-Score Comparison Across Classes and Imbalance Techniques", fontsize=11, fontweight="bold", pad=12)
    plt.ylim(0.20, 0.90)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p3 = artifacts_dir / "imbalance_f1_comparison.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"  [SAVED] Plot 3 -> {p3.name}", flush=True)

    # 4. Multi-panel Confusion Matrices (2x3 Grid)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    cm_keys = list(confusion_matrices.keys())
    cm_labels = ["0: Not Potable", "1: Potable"]

    for idx, (ax, exp_k) in enumerate(zip(axes.flat, cm_keys)):
        cm = confusion_matrices[exp_k]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=cm_labels,
            yticklabels=cm_labels,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"}
        )
        short_title = exp_k.split(":")[1].strip() if ":" in exp_k else exp_k
        bacc = comparison_df.loc[comparison_df["Experiment"] == exp_k, "Balanced_Accuracy"].values[0]
        c1_rec = comparison_df.loc[comparison_df["Experiment"] == exp_k, "Recall_C1"].values[0]
        ax.set_title(f"{short_title}\nBalAcc: {bacc:.3f} | C1 Rec: {c1_rec:.3f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=9, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=9, fontweight="bold")

    fig.suptitle("Phase 7: Confusion Matrices Across 6 Imbalance Experiments (Threshold 0.50, N=2,620 OOF)", fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    p4 = artifacts_dir / "imbalance_confusion_matrices.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"  [SAVED] Plot 4 -> {p4.name}", flush=True)


# ==============================================================================
# STEP 5 & 6: SELECT BEST APPROACH & THRESHOLD ANALYSIS
# ==============================================================================
def analyze_best_candidate_thresholds(
    winning_exp_name: str,
    y_train: pd.Series,
    probabilities: np.ndarray,
    candidate_thresholds: List[float] = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    artifacts_dir: Path = ARTIFACTS_DIR
) -> pd.DataFrame:
    """
    Step 6: Threshold analysis for the strongest candidate across [0.30 - 0.60].
    Saves artifacts/best_imbalance_threshold_comparison.csv.
    """
    print("\n" + "=" * 80, flush=True)
    print(f"PHASE 7: STEP 6 — THRESHOLD ANALYSIS FOR WINNING CANDIDATE ({winning_exp_name})", flush=True)
    print("=" * 80, flush=True)

    records = []
    for t in candidate_thresholds:
        preds = (probabilities >= t).astype(int)

        acc = accuracy_score(y_train, preds)
        bacc = balanced_accuracy_score(y_train, preds)
        prec1 = precision_score(y_train, preds, pos_label=1, zero_division=0)
        rec1 = recall_score(y_train, preds, pos_label=1, zero_division=0)
        f1_1 = f1_score(y_train, preds, pos_label=1, zero_division=0)
        rec0 = recall_score(y_train, preds, pos_label=0, zero_division=0)
        prec0 = precision_score(y_train, preds, pos_label=0, zero_division=0)
        f1_0 = f1_score(y_train, preds, pos_label=0, zero_division=0)

        cm = confusion_matrix(y_train, preds)
        tn, fp, fn, tp = cm.ravel()

        records.append({
            "threshold": round(float(t), 2),
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

    thresh_df = pd.DataFrame(records)

    csv_path = artifacts_dir / "best_imbalance_threshold_comparison.csv"
    ensure_dir(csv_path.parent)
    thresh_df.to_csv(csv_path, index=False)
    print(f"  [SAVED] Best imbalance threshold table -> {csv_path.name}", flush=True)

    print("\n" + "-" * 105, flush=True)
    print(thresh_df[["threshold", "accuracy", "balanced_accuracy", "precision_class_1", "recall_class_1", "f1_class_1", "recall_class_0"]].to_string(index=False), flush=True)
    print("-" * 105, flush=True)

    return thresh_df


# ==============================================================================
# STEP 7: MODEL PERSISTENCE UNDER models/experiments/
# ==============================================================================
def save_experimental_models(
    pipelines: Dict[str, ImbPipeline],
    winning_exp_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    experiments_dir: Path = EXPERIMENTS_DIR
):
    """
    Step 7: Fit and save experimental models under models/experiments/.
    models/best_model.joblib is strictly PRESERVED and NOT modified.
    """
    ensure_dir(experiments_dir)
    print("\n" + "=" * 80, flush=True)
    print(f"PHASE 7: STEP 7 — PERSISTING EXPERIMENTAL MODELS TO {experiments_dir}", flush=True)
    print("=" * 80, flush=True)

    # 1. Fit winning experimental model on full training set
    print(f"Fitting winning experiment ({winning_exp_name}) on full training set...", flush=True)
    winning_pipe = pipelines[winning_exp_name]
    winning_pipe.fit(X_train, y_train)

    best_exp_path = experiments_dir / "best_imbalance_model.joblib"
    joblib.dump(winning_pipe, best_exp_path)
    print(f"  * Saved winning experimental model -> {best_exp_path.name}", flush=True)

    # 2. Also save an experiment summary json
    summary_path = experiments_dir / "experiment_info.json"
    exp_summary = {
        "winning_experiment": winning_exp_name,
        "model_file": str(best_exp_path.name),
        "notes": "Experimental model saved separately. Champion model models/best_model.joblib remains active and protected."
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(exp_summary, f, indent=4)
    print(f"  * Saved experiment info metadata -> {summary_path.name}", flush=True)


# ==============================================================================
# MAIN EXECUTION ORCHESTRATION
# ==============================================================================
def run_phase_7_pipeline() -> Dict[str, Any]:
    """Execute complete Phase 7 pipeline."""
    # Load training data only (test set is NOT loaded or accessed)
    x_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"

    if not x_train_path.is_file() or not y_train_path.is_file():
        raise FileNotFoundError(f"Processed training data missing in {PROCESSED_DATA_DIR}")

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()

    # Step 1: Class Distribution Analysis
    dist_info = analyze_class_distribution(y_train)

    # Step 2 & 3: Pipelines setup
    pipelines = get_imbalance_experiment_pipelines()

    # Step 4: Evaluate all 6 experiments via 5-Fold CV
    comparison_df, oof_probabilities, confusion_matrices = evaluate_all_imbalance_experiments(
        X_train, y_train, pipelines
    )

    # Visualizations
    generate_imbalance_visualizations(comparison_df, confusion_matrices)

    # Step 5: Select best approach
    # Ranked primarily by Balanced Accuracy, secondarily by F1_C1 and Accuracy
    ranked_df = comparison_df.sort_values(
        by=["Balanced_Accuracy", "F1_C1", "Accuracy"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    best_exp_name = ranked_df.iloc[0]["Experiment"]
    best_exp_row = ranked_df.iloc[0].to_dict()

    print("\n" + "=" * 80, flush=True)
    print(" " * 22 + "PHASE 7: STEP 5 — BEST APPROACH SELECTION", flush=True)
    print("=" * 80, flush=True)
    print(f"  * Ranked #1 Imbalance Technique: {best_exp_name}", flush=True)
    print(f"  * Balanced Accuracy            : {best_exp_row['Balanced_Accuracy']:.4f}", flush=True)
    print(f"  * Overall Accuracy             : {best_exp_row['Accuracy']:.4f}", flush=True)
    print(f"  * Class 1 Recall               : {best_exp_row['Recall_C1']:.4f}", flush=True)
    print(f"  * Class 1 Precision            : {best_exp_row['Precision_C1']:.4f}", flush=True)
    print(f"  * Class 1 F1-Score             : {best_exp_row['F1_C1']:.4f}", flush=True)
    print(f"  * Class 0 Recall               : {best_exp_row['Recall_C0']:.4f}", flush=True)
    print("=" * 80, flush=True)

    # Step 6: Threshold analysis for winning approach
    best_oof_probas = oof_probabilities[best_exp_name]
    thresh_df = analyze_best_candidate_thresholds(best_exp_name, y_train, best_oof_probas)

    # Step 7: Persist experimental models under models/experiments/
    save_experimental_models(pipelines, best_exp_name, X_train, y_train)

    return {
        "distribution_info": dist_info,
        "comparison_table": comparison_df.to_dict(orient="records"),
        "best_experiment_name": best_exp_name,
        "best_experiment_metrics": best_exp_row,
        "threshold_analysis": thresh_df.to_dict(orient="records")
    }


def main():
    run_phase_7_pipeline()


if __name__ == "__main__":
    main()

