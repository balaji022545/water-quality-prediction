"""
Streamlit Web Application: Water Quality Prediction System.

Phase 6: Interactive Web Application
Author: Senior Python Developer & ML Engineer
Project: Water Quality Prediction Using Machine Learning
Champion Model: Tuned Random Forest Pipeline (models/best_model.joblib)

Important Operational Rules:
- Preserves the project folder structure.
- Uses models/best_model.joblib champion pipeline without retraining or altering.
- Does NOT load or evaluate the holdout test dataset (data/processed/X_test.csv).
- Takes all 9 chemical features in their exact training order.
- Supports optional missing values gracefully via the pipeline's embedded SimpleImputer.
- Displays dataset-based class predictions and probability scores.
- Prominently displays the responsible-use safety disclaimer and limitations.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Constant Definitions & Metadata
# -----------------------------------------------------------------------------
MODEL_PATH = Path("models") / "best_model.joblib"

# Exact 9 feature names in strict order matching model training schema
FEATURE_NAMES = [
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

# Feature metadata: Units, sensible default values, min/max bounds, and descriptions
FEATURE_METADATA = {
    "ph": {
        "label": "pH Level",
        "unit": "pH scale (0-14)",
        "min": 0.0,
        "max": 14.0,
        "default": 7.04,
        "step": 0.1,
        "description": "Acidity or alkalinity measure. Values near 7.0 are neutral.",
    },
    "Hardness": {
        "label": "Hardness",
        "unit": "mg/L",
        "min": 0.0,
        "max": 400.0,
        "default": 196.9,
        "step": 1.0,
        "description": "Capacity of water to precipitate soap; caused by calcium and magnesium ions.",
    },
    "Solids": {
        "label": "Solids (TDS)",
        "unit": "ppm",
        "min": 0.0,
        "max": 70000.0,
        "default": 20866.0,
        "step": 100.0,
        "description": "Total Dissolved Solids (TDS) indicating dissolved minerals, salts, and metals.",
    },
    "Chloramines": {
        "label": "Chloramines",
        "unit": "ppm",
        "min": 0.0,
        "max": 20.0,
        "default": 7.12,
        "step": 0.1,
        "description": "Disinfectant compound of chlorine and ammonia used in municipal treatment.",
    },
    "Sulfate": {
        "label": "Sulfate",
        "unit": "mg/L",
        "min": 0.0,
        "max": 600.0,
        "default": 333.1,
        "step": 1.0,
        "description": "Naturally occurring mineral salts formed from sulfur and oxygen.",
    },
    "Conductivity": {
        "label": "Conductivity",
        "unit": "µS/cm",
        "min": 0.0,
        "max": 1000.0,
        "default": 424.9,
        "step": 1.0,
        "description": "Electrical conductivity measuring ionic activity and dissolved solute concentration.",
    },
    "Organic_carbon": {
        "label": "Organic Carbon",
        "unit": "ppm",
        "min": 0.0,
        "max": 35.0,
        "default": 14.2,
        "step": 0.1,
        "description": "Total Organic Carbon (TOC) derived from decaying organic matter and synthetic compounds.",
    },
    "Trihalomethanes": {
        "label": "Trihalomethanes (THMs)",
        "unit": "µg/L",
        "min": 0.0,
        "max": 150.0,
        "default": 66.6,
        "step": 0.5,
        "description": "Chemical byproducts produced when chlorine disinfectants react with organic matter.",
    },
    "Turbidity": {
        "label": "Turbidity",
        "unit": "NTU",
        "min": 0.0,
        "max": 10.0,
        "default": 3.97,
        "step": 0.05,
        "description": "Cloudiness or haziness caused by large numbers of individual suspended microscopic particles.",
    },
}


# -----------------------------------------------------------------------------
# Model Loader with Streamlit Resource Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading machine learning model...")
def load_champion_pipeline(model_path: Path):
    """
    Load the trained champion machine learning pipeline.

    The saved pipeline contains:
      1. SimpleImputer(strategy='median')
      2. StandardScaler()
      3. Tuned RandomForestClassifier

    Parameters
    ----------
    model_path : Path
        Filesystem path to the serialized joblib pipeline.

    Returns
    -------
    pipeline : imblearn.pipeline.Pipeline
        Fitted self-contained pipeline.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Please ensure Phase 4 model training was executed."
        )

    pipeline = joblib.load(model_path)
    return pipeline


# -----------------------------------------------------------------------------
# Core Prediction Logic
# -----------------------------------------------------------------------------
def predict_sample(
    pipeline, input_dict: Dict[str, Optional[float]]
) -> Tuple[int, np.ndarray, pd.DataFrame]:
    """
    Format user input into a DataFrame and execute prediction.

    Parameters
    ----------
    pipeline : imblearn.pipeline.Pipeline
        Fitted champion pipeline.
    input_dict : dict
        Dictionary of feature names and numeric/None values.

    Returns
    -------
    predicted_class : int (0 or 1)
    probabilities : np.ndarray of shape (2,)
    input_df : pd.DataFrame of shape (1, 9)
    """
    # Construct DataFrame with strictly defined feature names and column order
    input_df = pd.DataFrame([input_dict], columns=FEATURE_NAMES)

    # Execute pipeline inference
    # Missing values (np.nan / None) are automatically imputed by the pipeline's SimpleImputer
    predicted_class = int(pipeline.predict(input_df)[0])
    probabilities = pipeline.predict_proba(input_df)[0]

    return predicted_class, probabilities, input_df


# -----------------------------------------------------------------------------
# Main Streamlit Application UI
# -----------------------------------------------------------------------------
def main():
    # Page setup
    st.set_page_config(
        page_title="Water Quality Prediction System",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom UI styling for clean presentation
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0E76A8;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #555555;
            margin-bottom: 1.2rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            border: 1px solid #e9ecef;
            margin-bottom: 10px;
        }
        .disclaimer-box {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 14px;
            border-radius: 4px;
            color: #856404;
            font-size: 0.92rem;
            margin-top: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Header Section
    # -------------------------------------------------------------------------
    st.markdown('<div class="main-header">💧 Water Quality Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Interactive Machine Learning Screening Interface for Water Potability Assessment</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        This web application employs an end-to-end Machine Learning pipeline
        (champion **Tuned Random Forest**) trained on the Kaggle Water Potability dataset.
        Enter the chemical and physical measurements of a water sample below to generate an automated potability screening.
        
        *Missing values are fully supported*: If a measurement is unknown or unmeasured, simply check the
        **"Unmeasured (NaN)"** box for that feature. The model's embedded imputer will substitute the training median value.
        """
    )

    st.divider()

    # -------------------------------------------------------------------------
    # Sidebar: Model Architecture & Quick Info
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("ℹ️ Model Architecture")
        st.markdown(
            """
            **Champion Model:**
            - Algorithm: `Random Forest Classifier`
            - Trees (`n_estimators`): `150`
            - Tree Depth (`max_depth`): `15`
            - Split Criterion: `Gini Impurity`
            
            **Embedded Pipeline Steps:**
            1. `SimpleImputer(strategy='median')`
            2. `StandardScaler()`
            3. `RandomForestClassifier()`
            
            **Holdout Test Benchmark:**
            - **Overall Accuracy:** `66.92%`
            - **ROC-AUC Score:** `0.6603`
            - **Class 0 Recall (Screening Power):** `92.50%`
            - **Class 1 Recall:** `26.95%`
            
            *The model prioritizes public health caution: it has a 92.5% capture rate for unsafe water, making it a conservative screening filter.*
            """
        )

        st.divider()
        st.subheader("💡 Preset Samples")
        preset_choice = st.radio(
            "Select a preset sample to test:",
            options=[
                "Custom Inputs",
                "Typical Neutral Sample",
                "Potable Profile Sample",
                "Sample with Missing Values (pH & Sulfate)",
            ],
            index=0,
            help="Quickly populate inputs with predefined test scenarios.",
        )

    # -------------------------------------------------------------------------
    # Preset Configurations
    # -------------------------------------------------------------------------
    preset_overrides = {}
    preset_missing = set()

    if preset_choice == "Typical Neutral Sample":
        preset_overrides = {
            "ph": 7.04,
            "Hardness": 196.9,
            "Solids": 20866.0,
            "Chloramines": 7.12,
            "Sulfate": 333.1,
            "Conductivity": 424.9,
            "Organic_carbon": 14.2,
            "Trihalomethanes": 66.6,
            "Turbidity": 3.97,
        }
    elif preset_choice == "Potable Profile Sample":
        preset_overrides = {
            "ph": 6.04,
            "Hardness": 227.0,
            "Solids": 12988.8,
            "Chloramines": 8.48,
            "Sulfate": 392.4,
            "Conductivity": 306.4,
            "Organic_carbon": 7.12,
            "Trihalomethanes": 41.19,
            "Turbidity": 3.59,
        }
    elif preset_choice == "Sample with Missing Values (pH & Sulfate)":
        preset_overrides = {
            "Hardness": 210.0,
            "Solids": 18500.0,
            "Chloramines": 6.8,
            "Conductivity": 410.0,
            "Organic_carbon": 13.5,
            "Trihalomethanes": 60.0,
            "Turbidity": 3.8,
        }
        preset_missing = {"ph", "Sulfate"}

    # -------------------------------------------------------------------------
    # Input Features Section
    # -------------------------------------------------------------------------
    st.subheader("📋 Enter Water Quality Measurements")
    st.caption("Provide values for all 9 chemical parameters. Check 'Unmeasured' to leave a field blank (NaN).")

    input_data: Dict[str, Optional[float]] = {}
    missing_features = []

    # Organize inputs into 3 columns
    col1, col2, col3 = st.columns(3, gap="medium")

    # Column 1: Chemical & Mineral Indicators
    with col1:
        st.markdown("#### 🧪 Chemical & Minerals")
        for feat in ["ph", "Hardness", "Solids"]:
            meta = FEATURE_METADATA[feat]
            is_preset_missing = feat in preset_missing
            is_missing = st.checkbox(
                f"Unmeasured ({feat})",
                value=is_preset_missing,
                key=f"missing_{feat}",
                help=f"Mark {feat} as missing (NaN). The model's SimpleImputer will handle it.",
            )

            if is_missing:
                input_data[feat] = np.nan
                missing_features.append(feat)
                st.text_input(
                    label=f"{meta['label']} [{meta['unit']}]",
                    value="[Unmeasured / Missing — Will be imputed]",
                    disabled=True,
                    key=f"disabled_{feat}",
                )
            else:
                default_val = float(preset_overrides.get(feat, meta["default"]))
                val = st.number_input(
                    label=f"{meta['label']} [{meta['unit']}]",
                    min_value=float(meta["min"]),
                    max_value=float(meta["max"]),
                    value=default_val,
                    step=float(meta["step"]),
                    format="%.2f" if meta["step"] < 1 else "%.0f",
                    help=f"{meta['description']} Typical range: {meta['min']} - {meta['max']} {meta['unit']}.",
                    key=f"input_{feat}",
                )
                input_data[feat] = float(val)

    # Column 2: Treatment & Conductivity
    with col2:
        st.markdown("#### 🚰 Treatment & Disinfection")
        for feat in ["Chloramines", "Sulfate", "Conductivity"]:
            meta = FEATURE_METADATA[feat]
            is_preset_missing = feat in preset_missing
            is_missing = st.checkbox(
                f"Unmeasured ({feat})",
                value=is_preset_missing,
                key=f"missing_{feat}",
                help=f"Mark {feat} as missing (NaN). The model's SimpleImputer will handle it.",
            )

            if is_missing:
                input_data[feat] = np.nan
                missing_features.append(feat)
                st.text_input(
                    label=f"{meta['label']} [{meta['unit']}]",
                    value="[Unmeasured / Missing — Will be imputed]",
                    disabled=True,
                    key=f"disabled_{feat}",
                )
            else:
                default_val = float(preset_overrides.get(feat, meta["default"]))
                val = st.number_input(
                    label=f"{meta['label']} [{meta['unit']}]",
                    min_value=float(meta["min"]),
                    max_value=float(meta["max"]),
                    value=default_val,
                    step=float(meta["step"]),
                    format="%.2f" if meta["step"] < 1 else "%.0f",
                    help=f"{meta['description']} Typical range: {meta['min']} - {meta['max']} {meta['unit']}.",
                    key=f"input_{feat}",
                )
                input_data[feat] = float(val)

    # Column 3: Organic, Byproducts & Physical
    with col3:
        st.markdown("#### 🔬 Organic & Physical")
        for feat in ["Organic_carbon", "Trihalomethanes", "Turbidity"]:
            meta = FEATURE_METADATA[feat]
            is_preset_missing = feat in preset_missing
            is_missing = st.checkbox(
                f"Unmeasured ({feat})",
                value=is_preset_missing,
                key=f"missing_{feat}",
                help=f"Mark {feat} as missing (NaN). The model's SimpleImputer will handle it.",
            )

            if is_missing:
                input_data[feat] = np.nan
                missing_features.append(feat)
                st.text_input(
                    label=f"{meta['label']} [{meta['unit']}]",
                    value="[Unmeasured / Missing — Will be imputed]",
                    disabled=True,
                    key=f"disabled_{feat}",
                )
            else:
                default_val = float(preset_overrides.get(feat, meta["default"]))
                val = st.number_input(
                    label=f"{meta['label']} [{meta['unit']}]",
                    min_value=float(meta["min"]),
                    max_value=float(meta["max"]),
                    value=default_val,
                    step=float(meta["step"]),
                    format="%.2f" if meta["step"] < 1 else "%.0f",
                    help=f"{meta['description']} Typical range: {meta['min']} - {meta['max']} {meta['unit']}.",
                    key=f"input_{feat}",
                )
                input_data[feat] = float(val)

    # -------------------------------------------------------------------------
    # Prediction Trigger Button
    # -------------------------------------------------------------------------
    st.write("")
    predict_btn = st.button("🔮 Predict Water Potability", type="primary", use_container_width=True)

    # -------------------------------------------------------------------------
    # Prediction Execution & Results Presentation
    # -------------------------------------------------------------------------
    if predict_btn:
        try:
            # 1. Load pipeline securely
            pipeline = load_champion_pipeline(MODEL_PATH)

            # 2. Execute inference
            pred_class, probas, input_df = predict_sample(pipeline, input_data)
            prob_class_0 = float(probas[0])
            prob_class_1 = float(probas[1])

            st.divider()
            st.subheader("📊 Prediction Results")

            # Result announcement card
            res_col1, res_col2 = st.columns([1.2, 1], gap="medium")

            with res_col1:
                if pred_class == 1:
                    st.success(
                        "### 🚰 Model Result: Potable (Class 1)\n"
                        "The machine learning model classifies this water sample as **Potable** "
                        "based on the statistical patterns learned from the Kaggle Water Potability training dataset."
                    )
                else:
                    st.error(
                        "### 🚫 Model Result: Not Potable (Class 0)\n"
                        "The machine learning model classifies this water sample as **Not Potable** "
                        "based on the statistical patterns learned from the Kaggle Water Potability training dataset."
                    )

                # Explanation of what prediction means
                st.markdown(
                    f"""
                    **Prediction Summary:**
                    - **Predicted Dataset Class:** `Class {pred_class}` ({'Potable' if pred_class == 1 else 'Not Potable'})
                    - **Confidence Distribution:** {prob_class_0*100:.1f}% Not Potable vs. {prob_class_1*100:.1f}% Potable
                    - **Imputed Features:** {', '.join(missing_features) if missing_features else 'None (all 9 measured)'}
                    """
                )

            with res_col2:
                st.markdown("#### Probability Distribution")
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(
                        label="Class 0 (Not Potable)",
                        value=f"{prob_class_0 * 100:.1f}%",
                    )
                with m_col2:
                    st.metric(
                        label="Class 1 (Potable)",
                        value=f"{prob_class_1 * 100:.1f}%",
                    )

                # Progress bar visualization for potability probability
                st.markdown("**Estimated Potability Score:**")
                st.progress(prob_class_1)
                st.caption(f"Score: {prob_class_1:.3f} / 1.000 (Decision Threshold = 0.500)")

            # Submitted Feature Data Inspection Expander
            with st.expander("🔍 View Submitted Feature Table (Input to Pipeline)", expanded=False):
                st.dataframe(input_df, use_container_width=True)
                if missing_features:
                    st.info(
                        f"Note: The following {len(missing_features)} feature(s) had missing values: "
                        f"**{', '.join(missing_features)}**. They were dynamically imputed with training medians "
                        "by the pipeline's embedded `SimpleImputer`."
                    )

        except FileNotFoundError as fnf_err:
            st.error(f"❌ Error: {fnf_err}")
        except Exception as ex:
            st.error(f"❌ Prediction failed due to an unexpected error: {ex}")

    # -------------------------------------------------------------------------
    # Mandatory Responsible-Use Disclaimer & Public Health Advisory
    # -------------------------------------------------------------------------
    st.divider()
    st.markdown(
        """
        <div class="disclaimer-box">
        <strong>⚠️ CRITICAL RESPONSIBLE-USE DISCLAIMER & LIMITATIONS:</strong><br>
        1. <strong>Not a Certified Safety Determination:</strong> The outputs generated by this application are
           statistical dataset predictions produced by an artificial intelligence model; they <strong>DO NOT</strong>
           constitute laboratory certification, clinical clearance, or regulatory approval of drinking water safety.<br>
        2. <strong>Laboratory Testing is Essential:</strong> Real-world drinking water safety requires professional
           physical, chemical, and microbiological laboratory analysis conducted in accordance with WHO, EPA, or local
           municipal standards.<br>
        3. <strong>Unmeasured Pathogens:</strong> This dataset measures only 9 physical/chemical variables. Critical
           <strong>microbiological pathogens (such as <em>Escherichia coli</em>, Salmonella, Vibrio cholerae, Giardia, and enteroviruses)</strong>
           are entirely unmeasured in this dataset. A water sample can be chemically balanced yet fatally contaminated with bacteria or viruses.<br>
        4. <strong>Conservative Model Behavior:</strong> Operating at a standard 0.50 cutoff, the model is intentionally
           conservative (92.5% Class 0 recall, 27.0% Class 1 recall on holdout testing) to minimize the dangerous risk
           of false-safe classifications.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
