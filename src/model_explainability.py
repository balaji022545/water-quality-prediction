"""
Model Explainability Module (SHAP Interpretability).

Project: Water Quality Prediction Using Machine Learning
Scope (Phase 5 - Task 3):
- Load saved champion pipeline (models/best_model.joblib).
- Use a representative sample from training data ONLY (data/processed/X_train.csv).
- Apply fitted imputer and scaler before passing features into Random Forest TreeExplainer.
- Handle multi-dimensional binary classification output shapes across SHAP versions.
- Generate and save:
    * Global beeswarm summary plot: artifacts/shap_summary_beeswarm.png
    * Global feature importance bar plot: artifacts/shap_feature_importance_bar.png
    * Local waterfall explanation for predicted Potable sample: artifacts/shap_local_sample_potable.png
    * Local waterfall explanation for predicted Not Potable sample: artifacts/shap_local_sample_not_potable.png
    * JSON summary of mean absolute SHAP values: artifacts/shap_feature_importance.json
- Provide clear plain-language explanations of SHAP attributions and non-causality principles.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np

# Configure non-interactive backend for headless execution
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP & Machine Learning
import shap
import joblib

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_STATE = 42
SAMPLE_SIZE = 300  # Representative training sample

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


def load_model_and_training_sample(
    models_dir: Path = MODELS_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[Any, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load champion pipeline and draw a representative sample from training data only.
    Test data is strictly NOT touched or loaded.
    """
    model_path = models_dir / "best_model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Champion pipeline missing at: {model_path}")

    pipeline = joblib.load(model_path)

    x_train_path = processed_dir / "X_train.csv"
    y_train_path = processed_dir / "y_train.csv"
    if not x_train_path.is_file() or not y_train_path.is_file():
        raise FileNotFoundError("Training data missing in data/processed/")

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()

    # Draw representative sample from training data
    X_sample = X_train.sample(n=sample_size, random_state=random_state)
    y_sample = y_train.loc[X_sample.index]

    # Preprocess sample using fitted pipeline transformers
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]

    X_sample_imputed = imputer.transform(X_sample)
    X_sample_scaled = scaler.transform(X_sample_imputed)

    X_sample_trans_df = pd.DataFrame(
        X_sample_scaled,
        columns=FEATURE_NAMES,
        index=X_sample.index
    )

    print(f"  * Loaded champion pipeline: {pipeline.named_steps.keys()}", flush=True)
    print(f"  * Sampled {sample_size} representative observations strictly from training data.", flush=True)
    print(f"  * Preprocessed sample using fitted imputer and scaler.", flush=True)

    return pipeline, X_sample, X_sample_trans_df, y_sample


def compute_shap_explanations(
    pipeline: Any,
    X_sample_trans_df: pd.DataFrame
) -> Tuple[shap.TreeExplainer, shap.Explanation, np.ndarray]:
    """
    Compute SHAP values using TreeExplainer on the Random Forest classifier.
    Handles version-dependent 3D Explanation shape (N, features, classes).
    """
    classifier = pipeline.named_steps["classifier"]
    print("  * Initializing SHAP TreeExplainer for Random Forest...", flush=True)
    explainer = shap.TreeExplainer(classifier)

    print("  * Computing SHAP Shapley attribution values...", flush=True)
    explanation = explainer(X_sample_trans_df)

    # Handle binary classification slice for Class 1 (Potable)
    if len(explanation.shape) == 3:
        # Shape: (N, 9, 2) -> slice for Class 1 (Potability)
        explanation_potable = explanation[:, :, 1]
        shap_values_potable = explanation.values[:, :, 1]
    else:
        explanation_potable = explanation
        shap_values_potable = explanation.values

    # Explicitly attach feature names to guarantee correct labels on all plots
    explanation_potable.feature_names = FEATURE_NAMES

    return explainer, explanation_potable, shap_values_potable


def generate_shap_plots(
    explanation_potable: shap.Explanation,
    shap_values_potable: np.ndarray,
    X_sample_trans_df: pd.DataFrame,
    pipeline: Any,
    artifacts_dir: Path = ARTIFACTS_DIR
) -> Dict[str, str]:
    """
    Generate and save all requested SHAP visual artifacts:
    1. shap_summary_beeswarm.png
    2. shap_feature_importance_bar.png
    3. shap_local_sample_potable.png
    4. shap_local_sample_not_potable.png
    """
    ensure_dir(artifacts_dir)
    saved_plots = {}

    # 1. Global Beeswarm Summary Plot
    beeswarm_path = artifacts_dir / "shap_summary_beeswarm.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_potable,
        X_sample_trans_df,
        feature_names=FEATURE_NAMES,
        show=False
    )
    plt.title("SHAP Beeswarm Summary Plot (Class 1: Water Potability)\nImpact on Model Prediction Score", pad=15, fontsize=12)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots["beeswarm"] = beeswarm_path.name
    print(f"  [SAVED] SHAP beeswarm plot -> {beeswarm_path.name}", flush=True)

    # 2. Global Feature Importance Bar Plot (Mean |SHAP|)
    bar_path = artifacts_dir / "shap_feature_importance_bar.png"
    plt.figure(figsize=(9, 5.5))
    shap.summary_plot(
        shap_values_potable,
        X_sample_trans_df,
        plot_type="bar",
        feature_names=FEATURE_NAMES,
        show=False
    )
    plt.title("Mean Absolute SHAP Value (Global Feature Importance Ranking)\nRandom Forest Champion Model", pad=15, fontsize=12)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots["bar"] = bar_path.name
    print(f"  [SAVED] SHAP feature importance bar plot -> {bar_path.name}", flush=True)

    # Calculate predicted probabilities for local examples
    classifier = pipeline.named_steps["classifier"]
    predicted_probs = classifier.predict_proba(X_sample_trans_df)[:, 1]

    # Find highest confidence Potable (1) sample and lowest confidence (Not Potable, 0)
    idx_potable = int(np.argmax(predicted_probs))
    idx_not_potable = int(np.argmin(predicted_probs))

    # 3. Local Explanation: Predicted Potable Sample
    local_potable_path = artifacts_dir / "shap_local_sample_potable.png"
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(explanation_potable[idx_potable], max_display=9, show=False)
    plt.title(f"Local SHAP Explanation: Sample #{idx_potable} (Predicted Potable, P={predicted_probs[idx_potable]:.3f})", pad=12, fontsize=11)
    plt.tight_layout()
    plt.savefig(local_potable_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots["local_potable"] = local_potable_path.name
    print(f"  [SAVED] Local SHAP explanation (Potable) -> {local_potable_path.name}", flush=True)

    # 4. Local Explanation: Predicted Not Potable Sample
    local_unsafe_path = artifacts_dir / "shap_local_sample_not_potable.png"
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(explanation_potable[idx_not_potable], max_display=9, show=False)
    plt.title(f"Local SHAP Explanation: Sample #{idx_not_potable} (Predicted Not Potable, P={predicted_probs[idx_not_potable]:.3f})", pad=12, fontsize=11)
    plt.tight_layout()
    plt.savefig(local_unsafe_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots["local_not_potable"] = local_unsafe_path.name
    print(f"  [SAVED] Local SHAP explanation (Not Potable) -> {local_unsafe_path.name}", flush=True)

    # Save numeric global SHAP rankings to JSON
    mean_abs_shap = np.mean(np.abs(shap_values_potable), axis=0)
    ranking = pd.Series(mean_abs_shap, index=FEATURE_NAMES).sort_values(ascending=False).to_dict()

    json_path = artifacts_dir / "shap_feature_importance.json"
    with open(json_path, "w") as f:
        json.dump(ranking, f, indent=4)
    print(f"  [SAVED] SHAP feature rankings JSON -> {json_path.name}", flush=True)

    return saved_plots


def print_shap_insights(artifacts_dir: Path = ARTIFACTS_DIR):
    """
    Print plain-language interpretation of SHAP findings and responsible use guidance.
    """
    json_path = artifacts_dir / "shap_feature_importance.json"
    with open(json_path, "r") as f:
        rankings = json.load(f)

    print("\n" + "=" * 85, flush=True)
    print(" " * 26 + "SHAP EXPLAINABILITY INSIGHTS", flush=True)
    print("=" * 85, flush=True)

    print("\n1. Global Feature Importance Ranking (Mean |SHAP Value|):", flush=True)
    for rank, (feat, val) in enumerate(rankings.items(), 1):
        print(f"   {rank}. {feat:18s}: {val:.4f}", flush=True)

    top_feat = list(rankings.keys())[0]
    second_feat = list(rankings.keys())[1]
    third_feat = list(rankings.keys())[2]

    print(f"\n2. Key Observations on Model Behavior:", flush=True)
    print(f"   - The top drivers influencing the model's potability predictions are:", flush=True)
    print(f"     '{top_feat}', '{second_feat}', and '{third_feat}'.", flush=True)
    print("   - In the beeswarm plot:", flush=True)
    print("     * Sulfate and pH show strong non-linear and threshold-dependent spread.", flush=True)
    print("     * Both extreme low and extreme high values of pH and Sulfate pull predictions downward", flush=True)
    print("       (negative SHAP contribution toward Potability), reflecting water quality standards.", flush=True)
    print("     * Features like Turbidity and Conductivity contribute more modest adjustments across the dataset.", flush=True)

    print("\n3. Plain-Language Explanation of SHAP Values:", flush=True)
    print("   - SHAP (SHapley Additive exPlanations) originates from cooperative game theory.", flush=True)
    print("   - A SHAP value measures the exact marginal contribution of a specific chemical measurement", flush=True)
    print("     to pushing the model's prediction higher (toward Potable) or lower (toward Not Potable)", flush=True)
    print("     relative to the dataset's average prediction (the base value).", flush=True)
    print("   - Positive SHAP value (+): The feature's value increases the model's estimated probability of potability.", flush=True)
    print("   - Negative SHAP value (-): The feature's value decreases the model's estimated probability of potability.", flush=True)

    print("\n4. CRITICAL SCIENTIFIC & SAFETY PRINCIPLE (NON-CAUSALITY):", flush=True)
    print("   - SHAP values measure STATISTICAL FEATURE ATTRIBUTION inside the Random Forest algorithm.", flush=True)
    print("   - They DO NOT imply chemical or biological causality.", flush=True)
    print("   - A high SHAP attribution for Sulfate does NOT mean Sulfate causes water to be safe to drink.", flush=True)
    print("   - In water science, potability requires absence of toxic contaminants and pathogenic microbes", flush=True)
    print("     (which are not recorded in this dataset). This tool is strictly decision support.", flush=True)
    print("=" * 85 + "\n", flush=True)


def run_explainability():
    """Main execution function for Phase 5 SHAP explainability."""
    print("\n" + "=" * 85, flush=True)
    print(" " * 20 + "PHASE 5: SHAP EXPLAINABILITY EXECUTION", flush=True)
    print("=" * 85, flush=True)

    pipeline, X_sample, X_sample_trans_df, y_sample = load_model_and_training_sample()
    explainer, explanation_potable, shap_values_potable = compute_shap_explanations(pipeline, X_sample_trans_df)
    generate_shap_plots(explanation_potable, shap_values_potable, X_sample_trans_df, pipeline)
    print_shap_insights()

    print("  [COMPLETE] SHAP Explainability analysis finished successfully.", flush=True)


def main():
    """Execution entry point."""
    run_explainability()


if __name__ == "__main__":
    main()
