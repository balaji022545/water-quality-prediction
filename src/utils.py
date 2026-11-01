"""
Utility Module for Water Quality Prediction Project.

Contains shared helper functions for:
- Loading and validating threshold configurations
- Threshold application and probability conversion
- Safe file and artifact handling
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np


DEFAULT_THRESHOLD = 0.50
THRESHOLD_CONFIG_PATH = Path("artifacts") / "threshold_config.json"


def load_threshold_config(
    config_path: Path = THRESHOLD_CONFIG_PATH,
    fallback_threshold: float = DEFAULT_THRESHOLD
) -> Tuple[float, Dict[str, Any]]:
    """
    Load the selected classification threshold and associated metadata from JSON config.

    Parameters
    ----------
    config_path : Path
        Path to threshold_config.json
    fallback_threshold : float
        Default threshold to fall back on if config is missing or invalid.

    Returns
    -------
    threshold : float
        Classification threshold in range (0.0, 1.0).
    config : dict
        Full configuration dictionary containing metadata and validation scores.
    """
    config_path = Path(config_path)

    if not config_path.is_file():
        # Configuration file missing: log warning and use safe default
        return fallback_threshold, {
            "selected_threshold": fallback_threshold,
            "status": "warning_file_missing",
            "message": f"Config not found at {config_path}. Using fallback threshold {fallback_threshold:.2f}."
        }

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        threshold = float(data.get("selected_threshold", fallback_threshold))

        # Validate threshold range
        if not (0.0 < threshold < 1.0):
            return fallback_threshold, {
                "selected_threshold": fallback_threshold,
                "status": "warning_invalid_range",
                "message": f"Threshold {threshold} outside (0.0, 1.0). Using fallback {fallback_threshold:.2f}."
            }

        return threshold, data

    except (json.JSONDecodeError, ValueError, KeyError) as err:
        return fallback_threshold, {
            "selected_threshold": fallback_threshold,
            "status": "warning_corrupted_config",
            "message": f"Error loading threshold config ({err}). Using fallback {fallback_threshold:.2f}."
        }


def apply_classification_threshold(
    probabilities: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD
) -> np.ndarray:
    """
    Apply a decision threshold to Class 1 (Potable) probabilities.

    Parameters
    ----------
    probabilities : np.ndarray
        Array of Class 1 probabilities, shape (N,) or (N, 2).
    threshold : float
        Decision cutoff. Predicts 1 if prob >= threshold, else 0.

    Returns
    -------
    predictions : np.ndarray
        Binary integer predictions (0 or 1).
    """
    probas = np.asarray(probabilities)
    if probas.ndim == 2:
        # If passed 2D probabilities (N, 2), extract Class 1 (second column)
        probas = probas[:, 1]

    return (probas >= threshold).astype(int)
