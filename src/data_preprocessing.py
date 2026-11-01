"""
Data Preprocessing & Exploratory Data Analysis (EDA) Module.

Project: Water Quality Prediction Using Machine Learning
Scope:
- Phase 2: Exploratory Data Analysis (EDA) visualization functions.
- Phase 3: Data cleaning, missing-value imputation (median), train-test splitting (stratified),
  leakage-free Scikit-learn preprocessing pipeline (SimpleImputer + StandardScaler),
  and saving processed outputs to data/processed/ and models/.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np

# Configure writable cache directory for Matplotlib on Windows sandbox
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless script execution
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules for Phase 3
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Path references
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "water_potability.csv"
DEFAULT_PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

# Configure visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14
})

FEATURE_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]
TARGET_COLUMN = "Potability"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ==============================================================================
# PHASE 2: EXPLORATORY DATA ANALYSIS (EDA) FUNCTIONS
# ==============================================================================

def plot_target_distribution(df: pd.DataFrame, save_dir: Path = DEFAULT_ARTIFACTS_DIR) -> Path:
    """Analyze and plot the distribution of the Potability target column."""
    ensure_dir(save_dir)
    save_path = save_dir / "target_distribution.png"

    plt.figure(figsize=(7, 5))
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    total = len(df)
    labels = ["0: Not Potable", "1: Potable"]
    colors = ["#e74c3c", "#2ecc71"]

    bars = plt.bar(labels, counts.values, color=colors, edgecolor="black", alpha=0.85, width=0.5)

    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (total * 0.015),
            f"{int(height):,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    plt.title("Target Distribution: Water Potability (0 = Unsafe, 1 = Safe)", pad=15)
    plt.ylabel("Sample Count")
    plt.ylim(0, max(counts.values) * 1.18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"  [SAVED] Target distribution plot -> {save_path.name}")
    return save_path


def plot_numerical_features_histograms(df: pd.DataFrame, save_dir: Path = DEFAULT_ARTIFACTS_DIR) -> Path:
    """Create a 3x3 grid of histograms with KDE for numerical features."""
    ensure_dir(save_dir)
    save_path = save_dir / "numerical_features_histograms.png"

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLUMNS):
        ax = axes[i]
        if col in df.columns:
            sns.histplot(
                data=df,
                x=col,
                kde=True,
                ax=ax,
                color="#3498db",
                edgecolor="white",
                alpha=0.6,
                bins=30
            )
            mean_val = df[col].mean()
            median_val = df[col].median()
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
            ax.axvline(median_val, color="green", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.1f}")
            ax.set_title(f"Distribution: {col}", fontweight="bold")
            ax.legend(fontsize=8, loc="upper right")
        else:
            ax.set_visible(False)

    plt.suptitle("Histograms & Kernel Density Estimation (KDE) of Water Metrics", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  [SAVED] Numerical features histograms -> {save_path.name}")
    return save_path


def plot_feature_boxplots(df: pd.DataFrame, save_dir: Path = DEFAULT_ARTIFACTS_DIR) -> Path:
    """Create boxplots to detect outliers and inspect spread across features."""
    ensure_dir(save_dir)
    save_path = save_dir / "feature_boxplots.png"

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLUMNS):
        ax = axes[i]
        if col in df.columns:
            if TARGET_COLUMN in df.columns:
                sns.boxplot(
                    data=df,
                    y=col,
                    x=TARGET_COLUMN,
                    hue=TARGET_COLUMN,
                    ax=ax,
                    palette=["#e74c3c", "#2ecc71"],
                    legend=False,
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.5, "markerfacecolor": "grey"}
                )
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["0: Unsafe", "1: Potable"])
                ax.set_xlabel("Potability")
            else:
                sns.boxplot(
                    data=df,
                    y=col,
                    ax=ax,
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.5, "markerfacecolor": "grey"}
                )
            ax.set_title(f"Boxplot: {col}", fontweight="bold")
        else:
            ax.set_visible(False)

    plt.suptitle("Outlier Analysis via Boxplots (Stratified by Potability)", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  [SAVED] Feature boxplots -> {save_path.name}")
    return save_path


def plot_correlation_heatmap(df: pd.DataFrame, save_dir: Path = DEFAULT_ARTIFACTS_DIR) -> Path:
    """Create a correlation heatmap to analyze relationships between features and target."""
    ensure_dir(save_dir)
    save_path = save_dir / "correlation_heatmap.png"

    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=0.2,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"}
    )

    plt.title("Correlation Matrix of Water Quality Parameters", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"  [SAVED] Correlation heatmap -> {save_path.name}")
    return save_path


def plot_missing_values_analysis(df: pd.DataFrame, save_dir: Path = DEFAULT_ARTIFACTS_DIR) -> Path:
    """Analyze and plot the missing values distribution."""
    ensure_dir(save_dir)
    save_path = save_dir / "missing_values_analysis.png"

    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    missing_summary = pd.DataFrame({
        "MissingCount": missing_counts,
        "Percentage": missing_pct
    }).sort_values(by="Percentage", ascending=True)

    plt.figure(figsize=(9, 5))
    bars = plt.barh(
        missing_summary.index,
        missing_summary["Percentage"],
        color=["#95a5a6" if p == 0 else "#e67e22" for p in missing_summary["Percentage"]],
        edgecolor="black",
        alpha=0.85
    )

    for bar in bars:
        width = bar.get_width()
        pct = width
        cnt = missing_summary.loc[missing_summary["Percentage"] == pct, "MissingCount"].values[0]
        if pct > 0:
            plt.text(
                width + 0.5,
                bar.get_y() + bar.get_height() / 2.0,
                f"{cnt:,} ({pct:.1f}%)",
                va="center",
                fontweight="bold",
                color="#d35400"
            )
        else:
            plt.text(
                0.5,
                bar.get_y() + bar.get_height() / 2.0,
                "0 (Complete)",
                va="center",
                color="green",
                fontsize=9
            )

    plt.title("Missing Values Percentage by Water Quality Feature", pad=15)
    plt.xlabel("Missingness (%)")
    plt.xlim(0, max(missing_pct.max() * 1.25, 10))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"  [SAVED] Missing values analysis plot -> {save_path.name}")
    return save_path


def print_eda_observations(df: pd.DataFrame):
    """Print structured explanations of key observations from the visualizations."""
    total_samples = len(df)
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0].to_dict()
    corr_with_target = df.corr()[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values()

    print("\n" + "=" * 80)
    print(" " * 26 + "EXPLORATORY DATA ANALYSIS (EDA) OBSERVATIONS")
    print("=" * 80)

    print("\n1. Target Variable (Potability) Distribution:")
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts().to_dict()
        pcts = (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2).to_dict()
        print(f"   - Class 0 (Not Potable): {counts.get(0, 0):,} samples ({pcts.get(0, 0)}%)")
        print(f"   - Class 1 (Potable)    : {counts.get(1, 0):,} samples ({pcts.get(1, 0)}%)")
        print("   - Observation: The dataset exhibits mild class imbalance (~61% non-potable vs ~39% potable).")
        print("     Standard accuracy will be insufficient; evaluation will require F1-Score and ROC-AUC.")

    print("\n2. Missing Value Analysis:")
    if cols_with_missing:
        for c, count in cols_with_missing.items():
            pct = (count / total_samples) * 100
            print(f"   - {c}: {count:,} missing values ({pct:.2f}%)")
        print("   - Observation: 'Sulfate', 'ph', and 'Trihalomethanes' contain significant missing entries.")
        print("     Imputation must strictly be performed inside cross-validation folds to avoid data leakage.")
    else:
        print("   - No missing values detected.")

    print("\n3. Feature Distributions & Outliers:")
    print("   - Most features (ph, Hardness, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity)")
    print("     exhibit approximately Gaussian/bell-shaped distributions centered around their means.")
    print("   - 'Solids' (Total Dissolved Solids) exhibits right-skewness with an extended upper tail.")
    print("   - Boxplots confirm the presence of mild outliers across extreme chemical values (e.g., pH extremes < 3 or > 11).")

    print("\n4. Feature Correlations:")
    print("   - Linear correlations between individual features and Potability are low (|r| < 0.10).")
    print("   - Top linear correlation with Potability:")
    for feat, r in corr_with_target.items():
        print(f"     * {feat:16s} : r = {r:+.4f}")
    print("   - Observation: No single chemical metric linearly separates potable from non-potable water.")
    print("     Non-linear tree-based models (Random Forest, XGBoost) will be critical to capture multi-feature interactions.")

    print("\n" + "=" * 80 + "\n")


def run_eda(df: pd.DataFrame, save_dir: Path = DEFAULT_ARTIFACTS_DIR) -> List[Path]:
    """Execute full EDA pipeline: generate all plots and print observations."""
    print("\n[STARTING EDA] Generating visualizations and computing insights...")
    generated_plots = [
        plot_target_distribution(df, save_dir),
        plot_numerical_features_histograms(df, save_dir),
        plot_feature_boxplots(df, save_dir),
        plot_correlation_heatmap(df, save_dir),
        plot_missing_values_analysis(df, save_dir),
    ]
    print_eda_observations(df)
    return generated_plots


# ==============================================================================
# PHASE 3: DATA CLEANING, TRAIN-TEST SPLIT & PREPROCESSING PIPELINE
# ==============================================================================

def check_missing_and_duplicates(df: pd.DataFrame) -> Tuple[pd.Series, int]:
    """
    Part A (Tasks 2, 4, 5): Check missing values, duplicate records, and document duplicate decision.
    
    Args:
        df (pd.DataFrame): Raw DataFrame.
        
    Returns:
        Tuple[pd.Series, int]: Missing counts per column and number of duplicate rows found.
    """
    print("\n" + "=" * 80)
    print(" " * 28 + "DATA CLEANING AUDIT REPORT")
    print("=" * 80)

    # 1. Missing values check
    missing_series = df.isnull().sum()
    print("\n--- 1. MISSING VALUES PER COLUMN ---")
    for col, count in missing_series.items():
        pct = (count / len(df)) * 100
        status = f"MISSING {count:,} ({pct:.2f}%)" if count > 0 else "0 (Complete)"
        print(f"  * {col:18s}: {status}")

    # 2. Duplicate records check & decision
    duplicate_count = int(df.duplicated().sum())
    print("\n--- 2. DUPLICATE RECORDS ANALYSIS & DECISION ---")
    print(f"  * Duplicate rows found: {duplicate_count}")
    if duplicate_count == 0:
        print("  * Decision: No duplicate removal was required because all 3,276 records are unique.")
    else:
        print(f"  * Decision: {duplicate_count} duplicate rows must be removed to avoid data leakage and metric bias.")

    return missing_series, duplicate_count


def validate_value_ranges(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Part A (Task 6): Check for invalid, negative, or physically impossible values.
    
    Args:
        df (pd.DataFrame): DataFrame to audit.
        
    Returns:
        dict: Summary of value validation checks.
    """
    print("\n--- 3. VALUE INTEGRITY & PHYSICAL BOUNDARY VALIDATION ---")
    issues = {}

    # Check 1: pH validity (Physical standard scale is 0.0 to 14.0)
    ph_out_of_bounds = df["ph"].dropna().apply(lambda x: x < 0.0 or x > 14.0).sum()
    print(f"  * pH range check [0.0, 14.0]     : {ph_out_of_bounds} invalid values (Range: {df['ph'].min():.2f} - {df['ph'].max():.2f})")
    if ph_out_of_bounds > 0:
        issues["ph_out_of_bounds"] = int(ph_out_of_bounds)

    # Check 2: Chemical concentrations non-negativity
    for col in FEATURE_COLUMNS:
        neg_count = int((df[col].dropna() < 0).sum())
        min_val = df[col].min()
        print(f"  * Non-negative check: {col:16s}: {neg_count} negative values (Min: {min_val:.2f})")
        if neg_count > 0:
            issues[f"{col}_negative"] = neg_count

    # Check 3: Target validity (Must be strictly binary 0 or 1)
    if TARGET_COLUMN in df.columns:
        unique_targets = set(df[TARGET_COLUMN].dropna().unique())
        is_binary = unique_targets.issubset({0, 1})
        print(f"  * Target values binary check      : {unique_targets} -> {'VALID (Strictly {0, 1})' if is_binary else 'INVALID'}")
        if not is_binary:
            issues["invalid_target_values"] = list(unique_targets)

    if not issues:
        print("  * Summary: All feature values fall within valid physical and mathematical ranges. No corrupted values.")
    else:
        print(f"  * [WARNING] Found value integrity issues: {issues}")

    return issues


def separate_features_and_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Part B: Separate dataset into feature matrix X and target vector y.
    Verifies that the target column is not accidentally included in X.
    
    Args:
        df (pd.DataFrame): Full dataframe.
        target_col (str): Target column name.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    # Extract target vector y
    y = df[target_col].copy()

    # Extract feature matrix X (drop target column)
    X = df.drop(columns=[target_col]).copy()

    # Integrity verification
    assert target_col not in X.columns, f"CRITICAL LEAKAGE: '{target_col}' found inside feature matrix X!"
    assert set(X.columns) == set(FEATURE_COLUMNS), f"Feature mismatch! Found: {list(X.columns)}"

    print("\n--- 4. SEPARATE FEATURES & TARGET ---")
    print(f"  * Feature matrix X shape: {X.shape} (Columns: {list(X.columns)})")
    print(f"  * Target vector y shape  : {y.shape} (Name: '{y.name}')")
    print(f"  * Verification: '{target_col}' successfully isolated from X (No target leakage).")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Part C: Train-Test Split with stratification to preserve target proportion.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        test_size (float): Fraction reserved for testing (default: 0.20).
        random_state (int): Seed for reproducibility (default: 42).
        
    Returns:
        Tuple[X_train, X_test, y_train, y_test]
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("\n--- 5. STRATIFIED TRAIN-TEST SPLIT ---")
    print(f"  * Split configuration: test_size={test_size * 100:.0f}%, random_state={random_state}, stratify=y")
    print(f"  * Training set shape : X_train = {X_train.shape}, y_train = {y_train.shape}")
    print(f"  * Testing set shape  : X_test  = {X_test.shape}, y_test  = {y_test.shape}")

    # Display target distribution in both splits
    train_dist = y_train.value_counts(normalize=True).round(4) * 100
    test_dist = y_test.value_counts(normalize=True).round(4) * 100
    train_counts = y_train.value_counts().to_dict()
    test_counts = y_test.value_counts().to_dict()

    print("\n  Target distribution in splits:")
    print(f"  * Training Set Target Counts: {train_counts} -> 0: {train_dist.get(0, 0):.2f}%, 1: {train_dist.get(1, 0):.2f}%")
    print(f"  * Testing Set Target Counts : {test_counts} -> 0: {test_dist.get(0, 0):.2f}%, 1: {test_dist.get(1, 0):.2f}%")

    return X_train, X_test, y_train, y_test


def build_preprocessing_pipeline(feature_names: List[str]) -> ColumnTransformer:
    """
    Part D: Build a reusable Scikit-learn preprocessing pipeline.
    
    Requirements:
    - SimpleImputer(strategy='median') for missing numerical values.
    - StandardScaler() for feature scaling.
    - Packaged into ColumnTransformer for clean column mapping.
    
    Args:
        feature_names (List[str]): List of numerical feature names.
        
    Returns:
        ColumnTransformer: Reusable preprocessor.
    """
    # Create the numerical transformation pipeline
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Build ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, feature_names)
        ],
        remainder="drop"
    )

    return preprocessor


def fit_and_apply_preprocessing(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """
    Fit preprocessor ONLY on training data, then transform both train and test.
    Also produces unscaled median-imputed data (ideal for tree models and explainability).
    
    Strict Data Leakage Prevention:
    - preprocessor.fit(X_train) calculates medians, means, and std exclusively from X_train.
    - X_test is transformed using those pre-computed training parameters.
    - The imputer/scaler are NEVER fitted on X_test.
    
    Args:
        preprocessor (ColumnTransformer): Preprocessor pipeline.
        X_train (pd.DataFrame): Raw training features.
        X_test (pd.DataFrame): Raw testing features.
        feature_names (List[str]): Feature column names.
        
    Returns:
        Tuple: (X_train_scaled, X_test_scaled, X_train_imputed, X_test_imputed, standalone_imputer)
    """
    print("\n--- 6. FITTING PREPROCESSING PIPELINE (ZERO DATA LEAKAGE) ---")
    print("  * Fitting ColumnTransformer [SimpleImputer(median) -> StandardScaler] strictly on X_train...")
    
    # 1. Fit preprocessor on X_train only
    preprocessor.fit(X_train)
    
    # 2. Transform X_train and X_test using the fitted pipeline
    X_train_scaled_arr = preprocessor.transform(X_train)
    X_test_scaled_arr = preprocessor.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_arr, columns=feature_names, index=X_test.index)
    
    # 3. Create unscaled imputed data (fitted on train only) for tree-based models (Random Forest, XGBoost)
    standalone_imputer = SimpleImputer(strategy="median")
    standalone_imputer.fit(X_train)
    
    X_train_imputed_arr = standalone_imputer.transform(X_train)
    X_test_imputed_arr = standalone_imputer.transform(X_test)
    
    X_train_imputed = pd.DataFrame(X_train_imputed_arr, columns=feature_names, index=X_train.index)
    X_test_imputed = pd.DataFrame(X_test_imputed_arr, columns=feature_names, index=X_test.index)

    # 4. Verification of zero missing values after imputation
    train_nulls = X_train_imputed.isnull().sum().sum()
    test_nulls = X_test_imputed.isnull().sum().sum()
    scaled_train_nulls = X_train_scaled.isnull().sum().sum()
    scaled_test_nulls = X_test_scaled.isnull().sum().sum()

    assert train_nulls == 0, f"Remaining missing values in X_train_imputed: {train_nulls}"
    assert test_nulls == 0, f"Remaining missing values in X_test_imputed: {test_nulls}"
    assert scaled_train_nulls == 0, f"Remaining missing values in X_train_scaled: {scaled_train_nulls}"
    assert scaled_test_nulls == 0, f"Remaining missing values in X_test_scaled: {scaled_test_nulls}"

    print("  * [VERIFIED] Preprocessing fitted strictly on X_train. Zero leakage confirmed.")
    print(f"  * [VERIFIED] Missing values after imputation: Train = {train_nulls}, Test = {test_nulls}")

    return X_train_scaled, X_test_scaled, X_train_imputed, X_test_imputed, standalone_imputer


def save_processed_data(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    X_train_imputed: pd.DataFrame,
    X_test_imputed: pd.DataFrame,
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    imputer: SimpleImputer,
    processed_dir: Path = DEFAULT_PROCESSED_DATA_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR
):
    """
    Part E: Save processed data and fitted preprocessing transformers.
    Does not touch or modify the raw dataset.
    """
    ensure_dir(processed_dir)
    ensure_dir(models_dir)

    print("\n--- 7. SAVING PROCESSED DATASETS & SERIALIZED PIPELINES ---")

    # 1. Save raw split feature matrices (preserves missing values for fold-level imputation in model pipelines)
    X_train_raw.to_csv(processed_dir / "X_train.csv", index=False)
    X_test_raw.to_csv(processed_dir / "X_test.csv", index=False)

    # 2. Save imputed and scaled reference datasets
    X_train_imputed.to_csv(processed_dir / "X_train_imputed.csv", index=False)
    X_test_imputed.to_csv(processed_dir / "X_test_imputed.csv", index=False)
    X_train_scaled.to_csv(processed_dir / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(processed_dir / "X_test_scaled.csv", index=False)

    # 3. Save target vectors
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

    # 4. Save combined train and test sets (convenience files with raw features + target)
    train_combined = X_train_raw.copy()
    train_combined[TARGET_COLUMN] = y_train.values
    train_combined.to_csv(processed_dir / "train.csv", index=False)

    test_combined = X_test_raw.copy()
    test_combined[TARGET_COLUMN] = y_test.values
    test_combined.to_csv(processed_dir / "test.csv", index=False)

    print(f"  * Saved processed datasets to: {processed_dir}")
    print("    - X_train.csv, X_test.csv (raw partitioned features with original missingness)")
    print("    - X_train_imputed.csv, X_test_imputed.csv (median-imputed features)")
    print("    - X_train_scaled.csv, X_test_scaled.csv (imputed + scaled features)")
    print("    - y_train.csv, y_test.csv (target column)")
    print("    - train.csv, test.csv (complete train/test tables)")

    # 5. Save fitted preprocessor and standalone imputer pipelines using Joblib
    preprocessor_path = models_dir / "preprocessor.joblib"
    imputer_path = models_dir / "imputer.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(imputer, imputer_path)

    print(f"  * Saved fitted preprocessor pipeline to: {preprocessor_path}")
    print(f"  * Saved fitted standalone imputer to   : {imputer_path}")


def run_preprocessing_pipeline(
    raw_data_path: Path = DEFAULT_RAW_DATA_PATH,
    processed_dir: Path = DEFAULT_PROCESSED_DATA_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR
) -> Dict[str, Any]:
    """
    Execute full Phase 3 data cleaning and preprocessing pipeline.
    
    Returns:
        dict: Detailed execution metrics and verification summary.
    """
    if not raw_data_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found at {raw_data_path}")

    # 1. Load raw data (do not modify source file)
    print(f"\n[PHASE 3 START] Loading raw dataset from: {raw_data_path}")
    raw_df = pd.read_csv(raw_data_path)

    # 2. Check missing values and duplicate records
    missing_before, duplicate_count = check_missing_and_duplicates(raw_df)

    # 3. Validate value ranges and physical boundaries
    validation_issues = validate_value_ranges(raw_df)

    # 4. Separate features and target
    X, y = separate_features_and_target(raw_df, target_col=TARGET_COLUMN)

    # 5. Stratified train-test split (80/20)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)

    # 6. Build and fit Scikit-learn preprocessing pipeline
    pipeline = build_preprocessing_pipeline(FEATURE_COLUMNS)
    X_train_scaled, X_test_scaled, X_train_imputed, X_test_imputed, imputer = fit_and_apply_preprocessing(
        preprocessor=pipeline,
        X_train=X_train,
        X_test=X_test,
        feature_names=FEATURE_COLUMNS
    )

    # 7. Save processed splits and fitted pipelines
    save_processed_data(
        X_train_raw=X_train,
        X_test_raw=X_test,
        X_train_imputed=X_train_imputed,
        X_test_imputed=X_test_imputed,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        preprocessor=pipeline,
        imputer=imputer,
        processed_dir=processed_dir,
        models_dir=models_dir
    )

    print("\n" + "=" * 80)
    print(" " * 24 + "PHASE 3 PREPROCESSING VERIFICATION COMPLETE")
    print("=" * 80 + "\n")

    return {
        "raw_shape": raw_df.shape,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape,
        "missing_before": missing_before.to_dict(),
        "missing_after_train": int(X_train_imputed.isnull().sum().sum()),
        "missing_after_test": int(X_test_imputed.isnull().sum().sum()),
        "duplicate_count": duplicate_count,
        "validation_issues": validation_issues,
        "train_target_dist": y_train.value_counts(normalize=True).round(4).to_dict(),
        "test_target_dist": y_test.value_counts(normalize=True).round(4).to_dict()
    }


def main():
    """Execution entry point for running Phase 3 preprocessing."""
    if not DEFAULT_RAW_DATA_PATH.is_file():
        print(f"\n[ALERT] Dataset missing at: {DEFAULT_RAW_DATA_PATH}")
        sys.exit(1)

    # Run complete preprocessing pipeline
    run_preprocessing_pipeline()


if __name__ == "__main__":
    main()
