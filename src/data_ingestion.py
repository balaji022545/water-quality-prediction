"""
Data Ingestion Module.

Project: Water Quality Prediction Using Machine Learning
Purpose:
- Check whether the raw dataset exists at 'data/raw/water_potability.csv'.
- Safely load raw dataset into Pandas DataFrame without modifying the source.
- Perform exploratory inspection: head, tail, shape, column names, data types,
  missing values, duplicate counts, statistical summaries, and target distribution.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

# Determine project root dynamically to ensure cross-platform path safety
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "water_potability.csv"


def check_dataset_exists(file_path: Path = DEFAULT_RAW_DATA_PATH) -> bool:
    """
    Check if the raw dataset file exists on disk.
    
    Args:
        file_path (Path): Path to the raw CSV file.
        
    Returns:
        bool: True if file exists and is a file, False otherwise.
    """
    return file_path.is_file()


def load_raw_data(file_path: Path = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw water potability dataset into a Pandas DataFrame.
    
    Args:
        file_path (Path): Path to raw CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
        
    Raises:
        FileNotFoundError: If the CSV file is missing.
        RuntimeError: If reading the CSV fails.
    """
    if not check_dataset_exists(file_path):
        missing_msg = (
            f"\n[DATASET MISSING ERROR]\n"
            f"The required dataset was not found at:\n  {file_path}\n\n"
            f"Action Required:\n"
            f"Please download 'water_potability.csv' from Kaggle and place it in:\n"
            f"  data/raw/water_potability.csv\n"
            f"No synthetic or placeholder data will be generated."
        )
        raise FileNotFoundError(missing_msg)
        
    try:
        # Load dataset preserving original types
        df = pd.read_csv(file_path)
        return df
    except Exception as exc:
        raise RuntimeError(f"Error while reading CSV file from {file_path}: {exc}") from exc


def inspect_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Display comprehensive dataset overview and summary statistics.
    
    Args:
        df (pd.DataFrame): Raw DataFrame to inspect.
        
    Returns:
        dict: Extracted summary information.
    """
    print("\n" + "=" * 80)
    print(" " * 24 + "DATA INGESTION & INSPECTION REPORT")
    print("=" * 80)

    # 1. First five rows
    print("\n--- 1. FIRST 5 ROWS ---")
    print(df.head())

    # 2. Last five rows
    print("\n--- 2. LAST 5 ROWS ---")
    print(df.tail())

    # 3. Shape
    rows, cols = df.shape
    print("\n--- 3. DATASET DIMENSIONS ---")
    print(f"Total Rows   : {rows}")
    print(f"Total Columns: {cols}")

    # 4. Column names
    print("\n--- 4. COLUMN NAMES ---")
    print(list(df.columns))

    # 5. Data types
    print("\n--- 5. DATA TYPES & MEMORY USAGE ---")
    print(df.dtypes)

    # 6. Missing value counts
    print("\n--- 6. MISSING VALUE COUNTS & PERCENTAGES ---")
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / rows) * 100
    missing_df = pd.DataFrame({
        "Missing Count": missing_counts,
        "Percentage (%)": missing_pct.round(2)
    })
    print(missing_df)

    # 7. Duplicate row count
    duplicate_count = int(df.duplicated().sum())
    print("\n--- 7. DUPLICATE ROWS ---")
    print(f"Total Duplicate Rows: {duplicate_count}")

    # 8. Statistical summary
    print("\n--- 8. STATISTICAL SUMMARY (describe) ---")
    print(df.describe().T.round(3))

    # 9. Target distribution
    target_col = "Potability"
    target_info = {}
    print("\n--- 9. TARGET VARIABLE DISTRIBUTION (Potability) ---")
    if target_col in df.columns:
        counts = df[target_col].value_counts().to_dict()
        proportions = (df[target_col].value_counts(normalize=True) * 100).round(2).to_dict()
        print(f"Counts     : {counts} (0 = Not Potable, 1 = Potable)")
        print(f"Percentages: {proportions} %")
        target_info = {"counts": counts, "proportions": proportions}
    else:
        print(f"[WARNING] Target column '{target_col}' not found in DataFrame.")

    print("\n" + "=" * 80)
    print(" " * 28 + "END OF INSPECTION REPORT")
    print("=" * 80 + "\n")

    return {
        "rows": rows,
        "cols": cols,
        "columns": list(df.columns),
        "missing_counts": missing_counts.to_dict(),
        "duplicate_count": duplicate_count,
        "target_distribution": target_info,
    }


def main():
    """Main execution entry point for testing data ingestion."""
    print(f"Checking dataset presence at: {DEFAULT_RAW_DATA_PATH}")
    if not check_dataset_exists(DEFAULT_RAW_DATA_PATH):
        print(f"\n[ALERT] Dataset file missing at: {DEFAULT_RAW_DATA_PATH}")
        print("Please place the Kaggle 'water_potability.csv' in 'data/raw/' and rerun.")
        sys.exit(1)
        
    print("[SUCCESS] Dataset found. Loading and inspecting...")
    df = load_raw_data(DEFAULT_RAW_DATA_PATH)
    inspect_raw_data(df)


if __name__ == "__main__":
    main()
